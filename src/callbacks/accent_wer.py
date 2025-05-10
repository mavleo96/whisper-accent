import pandas as pd
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torchmetrics.text import WordErrorRate

from src.constants import ACCENT_TO_ID_MAP


class AccentWERCallback(Callback):
    def __init__(self):
        super().__init__()
        self.id_to_accent_map = {value: key for key, value in ACCENT_TO_ID_MAP.items()}
        self.wer_metric = WordErrorRate()
        self._reset_buffers()

    def _reset_buffers(self):
        self.all_predictions = []
        self.all_targets = []
        self.all_accents = []

    def _process_batch(self, pl_module, outputs, batch):
        normalize = pl_module.processor.tokenizer.normalize
        preds = [normalize(p) for p in outputs["predictions"]]
        tgts = [normalize(t) for t in outputs["targets"]]

        self.all_predictions.extend(preds)
        self.all_targets.extend(tgts)
        self.all_accents.extend(batch["accent_id"].cpu().tolist())

    def _calculate_wer_metrics(self):
        self.wer_metric.reset()
        overall = self.wer_metric(self.all_predictions, self.all_targets).item()

        rows = []
        self.wer_metric.reset()
        for aid, accent in self.id_to_accent_map.items():
            idxs = [i for i, a in enumerate(self.all_accents) if a == aid]
            if not idxs:
                continue
            sub_preds = [self.all_predictions[i] for i in idxs]
            sub_tgts = [self.all_targets[i] for i in idxs]
            wer_val = self.wer_metric(sub_preds, sub_tgts).item()
            rows.append({"accent": accent, "wer": wer_val})

        return overall, rows

    def _log_wer_metrics(self, trainer, overall, rows, metric_name):
        df = pd.DataFrame(rows).sort_values("accent")
        overall_row = {"accent": "Overall", "wer": overall}
        df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)

        text = f"<pre>{df.to_csv(index=False)}</pre>"
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_text(metric_name, text, global_step=0)
            elif isinstance(logger, WandbLogger):
                logger.log_metrics({f"{metric_name}/overall": overall})
                for _, row in df.iterrows():
                    logger.log_metrics({f"{metric_name}/{row['accent']}": row["wer"]})
                logger.log_table(
                    key=f"{metric_name}_table",
                    dataframe=df,
                )

    # def on_validation_batch_end(
    #     self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    # ):
    #     self._process_batch(pl_module, outputs, batch)

    # def on_validation_epoch_end(self, trainer, pl_module):
    #     overall, rows = self._calculate_wer_metrics()
    #     self._log_wer_metrics(trainer, overall, rows, "val_accent_wer")
    #     self._reset_buffers()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._process_batch(pl_module, outputs, batch)

    def on_test_epoch_end(self, trainer, pl_module):
        overall, rows = self._calculate_wer_metrics()
        self._log_wer_metrics(trainer, overall, rows, "test_accent_wer")
        self._reset_buffers()
