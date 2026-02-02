from torch.utils.data import Dataset, IterableDataset
from torchmetrics.classification import Accuracy
from torchmetrics.text import WordErrorRate
from transformers import EvalPrediction, Seq2SeqTrainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.trainer_callback import TrainerCallback

from src.train.train import WhisperAccentTrainingArguments


class WhisperAccentTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args: WhisperAccentTrainingArguments,
        data_collator: DataCollator,
        train_dataset: Dataset | IterableDataset,
        eval_dataset: Dataset | dict[str, Dataset],
        processing_class: ProcessorMixin,
        callbacks: list[TrainerCallback] | None = None,
    ):
        assert args.batch_eval_metrics, "Batch eval metrics must be enabled"
        assert "inputs" in args.include_for_metrics, (
            "Inputs must be included for metrics"
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
        self.metrics_accumulators = {
            "wer": WordErrorRate(),
            "acc": Accuracy(
                task="multiclass", num_classes=len(model.generation_config.accent_to_id)
            ).to(self.model.device),
        }

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
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, model
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    @staticmethod
    def get_optimizer_cls_and_kwargs(args, model):
        optimizer_cls, optimizer_kwargs = Seq2SeqTrainer.get_optimizer_cls_and_kwargs(
            args, model
        )
        optimizer_kwargs.pop("lr")
        return optimizer_cls, optimizer_kwargs

    def compute_metrics(
        self, eval_pred: EvalPrediction, compute_result: bool = True
    ) -> dict[str, float]:
        tokenizer = self.processing_class.tokenizer
        predictions = eval_pred.predictions
        label_ids = eval_pred.label_ids
        inputs = eval_pred.inputs

        # Update WER
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        raw_pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        pred_str = [tokenizer.normalize(s) for s in raw_pred_str]
        self.metrics_accumulators["wer"].update(pred_str, label_str)

        # Update Accent Accuracy
        preds, labels = self._get_accent_labels_and_preds(inputs)
        self.metrics_accumulators["acc"].update(preds, labels)

        # On final compute, return the result and reset the metric
        if compute_result:
            results = {}
            for name, metric in self.metrics_accumulators.items():
                results[name] = metric.compute().item()
                metric.reset()
            return results

    def _get_accent_labels_and_preds(self, inputs):
        return_timestamps = self.model.generation_config.return_timestamps
        is_multilingual = self.model.generation_config.is_multilingual

        output = self.model._retrieve_init_tokens(
            inputs["input_features"],
            inputs["input_features"].shape[0],
            self.model.generation_config,
            self.model.config,
            3000,
            {},
        )
        preds = output[:, -1] if return_timestamps else output[:, -2]
        labels = inputs["labels"][:, 0] if is_multilingual else inputs["labels"][:, 1]
        min_label = min(self.model.generation_config.accent_to_id.values())
        return preds - min_label, labels - min_label


__all__ = ["WhisperAccentTrainer"]
