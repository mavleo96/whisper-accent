import logging

import pandas as pd
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.text import WordErrorRate

from src.constants import ACCENT_TO_ID_MAP

logger = logging.getLogger(__name__)


class AccentWERCallback(Callback):
    def __init__(self):
        super().__init__()
        self.id_to_accent_map = dict()
        self.wer_metric = WordErrorRate()
        self._reset_buffers()

        for key, value in ACCENT_TO_ID_MAP.items():
            if not self.id_to_accent_map.get(value):
                self.id_to_accent_map[value] = key

    def _reset_buffers(self):
        self.all_predictions = []
        self.all_targets = []
        self.all_accents = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        normalize = pl_module.processor.tokenizer.normalize
        preds = [normalize(p) for p in outputs["predictions"]]
        tgts = [normalize(t) for t in outputs["targets"]]

        self.all_predictions.extend(preds)
        self.all_targets.extend(tgts)
        self.all_accents.extend(batch["accent_id"].cpu().tolist())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Overall WER
        self.wer_metric.reset()
        overall = self.wer_metric(self.all_predictions, self.all_targets).item()

        # Per-accent WER
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

        df = pd.DataFrame(rows).sort_values("accent")

        overall_row = {"accent": "Overall", "wer": overall}
        df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)

        text = f"<pre>{df.to_csv(index=False)}</pre>"
        for tb in trainer.loggers:
            if isinstance(tb, TensorBoardLogger):
                tb.experiment.add_text("val_accent_wer", text, global_step=0)

        self._reset_buffers()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        normalize = pl_module.processor.tokenizer.normalize
        preds = [normalize(p) for p in outputs["predictions"]]
        tgts = [normalize(t) for t in outputs["targets"]]

        self.all_predictions.extend(preds)
        self.all_targets.extend(tgts)
        self.all_accents.extend(batch["accent_id"].cpu().tolist())

    def on_test_epoch_end(self, trainer, pl_module):
        # Overall WER
        self.wer_metric.reset()
        overall = self.wer_metric(self.all_predictions, self.all_targets).item()

        # Per-accent WER
        self.wer_metric.reset()
        rows = []
        for aid, accent in self.id_to_accent_map.items():
            idxs = [i for i, a in enumerate(self.all_accents) if a == aid]
            if not idxs:
                continue
            sub_preds = [self.all_predictions[i] for i in idxs]
            sub_tgts = [self.all_targets[i] for i in idxs]
            wer_val = self.wer_metric(sub_preds, sub_tgts).item()
            rows.append({"accent": accent, "wer": wer_val})

        df = pd.DataFrame(rows).sort_values("accent")

        overall_row = {"accent": "Overall", "wer": overall}
        df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)

        text = f"<pre>{df.to_csv(index=False)}</pre>"
        for tb in trainer.loggers:
            if isinstance(tb, TensorBoardLogger):
                tb.experiment.add_text("test_accent_wer", text, global_step=0)

        self._reset_buffers()
