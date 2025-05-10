import pandas as pd
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger


class PredictionSaver(Callback):
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

        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_text("predictions", text, global_step=0)
                df.to_csv(logger.log_dir + "/predictions.csv", index=False)
            elif isinstance(logger, WandbLogger):
                logger.log_table(
                    key="predictions",
                    dataframe=df,
                )
