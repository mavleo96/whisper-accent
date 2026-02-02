import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from torchmetrics.text import WordErrorRate
from transformers import Seq2SeqTrainer

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

        self.metrics_accumulators = {}
        for metric in compute_metrics:
            if metric == "wer":
                self.metrics_accumulators["wer"] = WordErrorRate()
            if metric == "accent_accuracy":
                self.metrics_accumulators["accent_accuracy"] = Accuracy(
                    task="multiclass", num_classes=len(model.generation_config.accent_to_id)
                ).to(self.model.device)

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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        if self.model.config.model_type == "whisper":
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

        return (loss, outputs) if return_outputs else loss

    def compute_accent_loss(self, outputs, inputs):
        # Get accent token position and indices
        is_multilingual = self.model.generation_config.is_multilingual
        accent_token_position = 2 if is_multilingual else 0
        accent_token_indices = sorted(list(self.model.generation_config.accent_to_id.values()))
        min_label = min(self.model.generation_config.accent_to_id.values())

        # Get accent labels and logits
        accent_labels = inputs["labels"][:, accent_token_position] - min_label
        accent_logits = outputs.logits[:, accent_token_position, accent_token_indices]

        # Compute accent loss
        return F.cross_entropy(accent_logits, accent_labels, reduction="mean")

    def compute_diversity_loss(self):
        accent_token_indices = sorted(list(self.model.generation_config.accent_to_id.values()))
        embedding_layer = self.model.base_model.model.model.decoder.embed_tokens
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
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            raw_pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
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

    def _get_accent_labels_and_preds(self, inputs):
        is_multilingual = self.model.generation_config.is_multilingual
        accent_token_position = 2 if is_multilingual else 0

        init_tokens = self.model._retrieve_init_tokens(
            inputs["input_features"],
            inputs["input_features"].shape[0],
            self.model.generation_config,
            self.model.config,
            3000,
            {},
        )
        preds = init_tokens[:, accent_token_position + 1]
        labels = inputs["labels"][:, accent_token_position]
        min_label = min(self.model.generation_config.accent_to_id.values())
        return preds - min_label, labels - min_label


__all__ = ["WhisperAccentTrainer"]
