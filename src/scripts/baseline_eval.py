import logging

import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.data_module import EdaccDataModule
from src.models.base_model import BaseWhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="../../configs", config_name="baseline_eval.yaml")
def main(cfg):
    logger.info(f"Initializing model: {cfg.model.model_name}")
    model = BaseWhisperModel(**cfg.model)

    logger.info("Initializing data module")
    data_module = EdaccDataModule(**cfg.data)

    model_name = cfg.model.model_name.split("/")[-1]
    tensorboard_logger = TensorBoardLogger(
        save_dir=cfg.trainer.logger[0].save_dir,
        name=f"baseline_eval_{model_name}",
    )

    logger.info("Initializing trainer")
    trainer = L.Trainer(
        logger=[tensorboard_logger],
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        enable_progress_bar=True,
    )

    logger.info("Starting evaluation")
    results = trainer.test(model, data_module)

    logger.info("Evaluation results:")
    for metric, value in results[0].items():
        logger.info(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
