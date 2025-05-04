import logging

import pandas as pd
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger

logger = logging.getLogger(__name__)


class PredictionLogger(Callback):
    def __init__(self):
        super().__init__()
        self.all_predictions = []
        self.all_targets = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Accumulate predictions and targets in callback object
        self.all_predictions.extend(outputs["predictions"])
        self.all_targets.extend(outputs["targets"])

    def on_test_end(self, trainer, pl_module):
        df = pd.DataFrame(
            {
                "id": range(len(self.all_predictions)),
                "target": self.all_targets,
                "prediction": self.all_predictions,
            }
        )
        text = f"<pre>{df.to_csv(index=False)}</pre>"

        for tb_logger in trainer.loggers:
            if isinstance(tb_logger, TensorBoardLogger):
                tb_logger.experiment.add_text("predictions", text, global_step=0)