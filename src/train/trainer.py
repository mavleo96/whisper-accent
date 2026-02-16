import evaluate
import torch
from transformers import Seq2SeqTrainer

from src.constants import IGNORE_INDEX


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

        self.metrics_accumulators = {}
        for metric in compute_metrics:
            if metric == "wer":
                self.metrics_accumulators["wer"] = evaluate.load("wer")
            if metric == "accent_accuracy":
                self.metrics_accumulators["accent_accuracy"] = evaluate.load("accuracy")

    def create_optimizer(self):
        model = self.model
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and "modulation.1.weight" in n
                ],
                "lr": self.args.learning_rate,
                "weight_decay": 0.0,  # No weight decay for modulation weights; they are zero-init
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and "embed_accents" in n
                ],
                "lr": self.args.embedding_learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and "accent_classifier" in n
                ],
                "lr": self.args.accent_classifier_learning_rate,
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
        # Note: This is a hack to prevent logging learning rate inside eval loop
        inside_eval_loop = False if "loss" in logs else True

        # Log both learning_rate (main) and embedding_learning_rate when using two param groups
        if self.optimizer is not None and not inside_eval_loop:
            lr_main = self.optimizer.param_groups[0]["lr"]
            lr_embed = self.optimizer.param_groups[1]["lr"]
            lr_accent_classifier = self.optimizer.param_groups[2]["lr"]
            logs["learning_rate"] = lr_main.item() if torch.is_tensor(lr_main) else lr_main
            logs["embedding_learning_rate"] = (
                lr_embed.item() if torch.is_tensor(lr_embed) else lr_embed
            )
            logs["accent_classifier_learning_rate"] = (
                lr_accent_classifier.item()
                if torch.is_tensor(lr_accent_classifier)
                else lr_accent_classifier
            )
        super().log(logs, start_time)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        if self.accelerator.unwrap_model(model).config.model_type == "whisper":
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # Transcription Loss
        transcription_loss = outputs.loss
        accent_loss = outputs.accent_loss

        # Compute total loss
        loss = transcription_loss + self.args.lambda_accent * accent_loss

        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

        return (loss, outputs) if return_outputs else loss

    # def compute_diversity_loss(self):
    #     # Use unwrapped model so attribute access works under DDP (self.model may be wrapped)
    #     model = self.accelerator.unwrap_model(self.model)
    #     embedding_layer = model.get_input_embeddings()
    #     accent_embeddings = embedding_layer.token_adapter.trainable_tokens_delta["default"]

    #     # Compute diversity loss
    #     return repulsive_loss(accent_embeddings, temperature=0.1)

    def compute_metrics(self, eval_pred, compute_result=True):
        predictions = eval_pred.predictions
        label_ids = eval_pred.label_ids
        inputs = eval_pred.inputs

        # Update metric accumulators
        if "wer" in self.metrics_accumulators:
            label_str, pred_str = self._get_normalized_labels_and_preds(label_ids, predictions)
            self.metrics_accumulators["wer"].add_batch(predictions=pred_str, references=label_str)

        if "accent_accuracy" in self.metrics_accumulators:
            preds, labels = self._get_accent_labels_and_preds(inputs)
            self.metrics_accumulators["accent_accuracy"].add_batch(
                predictions=preds, references=labels
            )

        # On final compute, return the result
        if compute_result:
            results = {}
            for name, metric in self.metrics_accumulators.items():
                if name == "wer":
                    results[name] = metric.compute()
                if name == "accent_accuracy":
                    results[name] = metric.compute()["accuracy"]
            return results
        return {}

    def _get_normalized_labels_and_preds(self, label_ids, predictions):
        tokenizer = self.processing_class.tokenizer

        # Decode and normalize labels
        label_ids_np = label_ids.cpu().numpy().copy()
        label_ids_np[label_ids_np == IGNORE_INDEX] = tokenizer.pad_token_id
        raw_label_str = tokenizer.batch_decode(label_ids_np, skip_special_tokens=True)
        label_str = [tokenizer.normalize(s) for s in raw_label_str]

        # Decode and normalize predictions
        pred_ids = predictions.cpu().numpy()
        raw_pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        pred_str = [tokenizer.normalize(s) for s in raw_pred_str]
        return label_str, pred_str

    def _get_accent_labels_and_preds(self, inputs):
        model = self.accelerator.unwrap_model(self.model)

        with torch.no_grad():
            encoder_outputs = model.get_encoder()(
                inputs["input_features"], inputs["attention_mask"], return_dict=True
            )

        labels = inputs["accent_labels"]
        preds = encoder_outputs.accent_logits.argmax(dim=-1)
        return preds, labels


__all__ = ["WhisperAccentTrainer"]
