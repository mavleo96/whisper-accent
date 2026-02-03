import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from torchmetrics.text import WordErrorRate
from transformers import Seq2SeqTrainer

from src.constants import IGNORE_INDEX
from src.utils import repulsive_loss


class WhisperAccentTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        model,
        args,
        data_collator,
        train_dataset,
        eval_dataset,
        processing_class,
        callbacks=None,
        compute_metrics="none",
    ):
        assert args.batch_eval_metrics, "Batch eval metrics must be enabled"
        assert "inputs" in args.include_for_metrics, "Inputs must be included for metrics"
        assert compute_metrics in ["none", "all", "wer", "accent_accuracy"], (
            "Invalid compute_metrics value"
        )
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )

        if compute_metrics == "all":
            compute_metrics = ["wer", "accent_accuracy"]
        elif compute_metrics == "none":
            compute_metrics = []
        else:
            compute_metrics = [compute_metrics]

        # Metrics must be on the accelerator device so torchmetrics' distributed sync (all_gather)
        # works with NCCL
        device = self.accelerator.device
        self.metrics_accumulators = {}
        for metric in compute_metrics:
            if metric == "wer":
                self.metrics_accumulators["wer"] = WordErrorRate().to(device)
            if metric == "accent_accuracy":
                self.metrics_accumulators["accent_accuracy"] = Accuracy(
                    task="multiclass", num_classes=len(model.generation_config.accent_to_id)
                ).to(device)

    def create_optimizer(self):
        model = self.model
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and "trainable_tokens_delta" in n
                ],
                "lr": self.args.embedding_learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and "trainable_tokens_delta" not in n
                ],
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            },
        ]
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, model)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    @staticmethod
    def get_optimizer_cls_and_kwargs(args, model):
        optimizer_cls, optimizer_kwargs = Seq2SeqTrainer.get_optimizer_cls_and_kwargs(args, model)
        optimizer_kwargs.pop("lr")
        return optimizer_cls, optimizer_kwargs

    def log(self, logs, start_time=None):
        # Log both learning_rate (main) and embedding_learning_rate when using two param groups
        if self.optimizer is not None and len(self.optimizer.param_groups) >= 2:
            lr_main = self.optimizer.param_groups[1]["lr"]
            lr_embed = self.optimizer.param_groups[0]["lr"]
            logs["learning_rate"] = lr_main.item() if torch.is_tensor(lr_main) else lr_main
            logs["embedding_learning_rate"] = (
                lr_embed.item() if torch.is_tensor(lr_embed) else lr_embed
            )
        super().log(logs, start_time)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        if self.accelerator.unwrap_model(model).config.model_type == "whisper":
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # Transcription Loss
        transcription_loss = outputs.loss

        # Compute Accent Loss
        accent_loss = 0.0
        if self.args.lambda_accent_loss > 0:
            accent_loss = self.compute_accent_loss(outputs, inputs)

        # Compute diversity loss
        diversity_loss = 0.0
        if self.args.lambda_diversity_loss > 0:
            diversity_loss = self.compute_diversity_loss()

        # Compute total loss
        loss = (
            transcription_loss
            + self.args.lambda_accent_loss * accent_loss
            + self.args.lambda_diversity_loss * diversity_loss
        )

        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

        return (loss, outputs) if return_outputs else loss

    def compute_accent_loss(self, outputs, inputs):
        model = self.accelerator.unwrap_model(self.model)
        is_multilingual = model.generation_config.is_multilingual
        accent_token_position = 2 if is_multilingual else 0
        accent_token_indices = sorted(list(model.generation_config.accent_to_id.values()))
        min_label = min(model.generation_config.accent_to_id.values())

        # Get accent labels and logits
        accent_labels = (inputs["labels"][:, accent_token_position] - min_label).long()
        accent_logits = outputs.logits[:, accent_token_position, accent_token_indices]

        # Compute accent loss
        return F.cross_entropy(accent_logits, accent_labels, reduction="mean")

    def compute_diversity_loss(self):
        # Use unwrapped model so attribute access works under DDP (self.model may be wrapped)
        model = self.accelerator.unwrap_model(self.model)
        accent_token_indices = sorted(list(model.generation_config.accent_to_id.values()))
        embedding_layer = model.base_model.model.model.decoder.embed_tokens
        accent_embeddings = (
            embedding_layer.token_adapter.base_layer.weight[accent_token_indices]
            + embedding_layer.token_adapter.trainable_tokens_delta["default"]
        )

        # Compute diversity loss
        return repulsive_loss(accent_embeddings, temperature=0.1)

    def compute_metrics(self, eval_pred, compute_result=True):
        tokenizer = self.processing_class.tokenizer
        predictions = eval_pred.predictions
        label_ids = eval_pred.label_ids
        inputs = eval_pred.inputs

        # Update WER
        if "wer" in self.metrics_accumulators:
            label_ids_np = label_ids.cpu().numpy().copy()
            label_ids_np[label_ids_np == IGNORE_INDEX] = tokenizer.pad_token_id
            label_str = tokenizer.batch_decode(label_ids_np, skip_special_tokens=True)
            pred_ids = predictions.cpu().numpy()
            raw_pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            pred_str = [tokenizer.normalize(s) for s in raw_pred_str]
            self.metrics_accumulators["wer"].update(pred_str, label_str)

        # Update Accent Accuracy
        if "accent_accuracy" in self.metrics_accumulators:
            preds, labels = self._get_accent_labels_and_preds(inputs)
            self.metrics_accumulators["accent_accuracy"].update(preds, labels)

        # On final compute, return the result and reset the metric
        if compute_result:
            results = {}
            for name, metric in self.metrics_accumulators.items():
                results[name] = metric.compute().item()
                metric.reset()
            return results
        return {}

    def _get_accent_labels_and_preds(self, inputs):
        model = self.accelerator.unwrap_model(self.model)
        is_multilingual = model.generation_config.is_multilingual
        accent_token_position = 2 if is_multilingual else 0

        init_tokens = model._retrieve_init_tokens(
            inputs["input_features"],
            inputs["input_features"].shape[0],
            model.generation_config,
            model.config,
            3000,
            {},
        )
        preds = init_tokens[:, accent_token_position + 1]
        labels = inputs["labels"][:, accent_token_position]
        min_label = min(model.generation_config.accent_to_id.values())
        return (preds - min_label).long(), (labels - min_label).long()


__all__ = ["WhisperAccentTrainer"]
