import logging

import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf

from src.callbacks import AccentWERCallback, PredictionSaver
from src.data.data_module import EdaccDataModule
from src.models.base_model import BaseWhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


@hydra.main(
    config_path="../../configs", config_name="baseline_eval.yaml", version_base=None
)
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

    # Convert config to dict for wandb
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    if "model" in config_dict and "optimizer_config" in config_dict["model"]:
        opt_config = config_dict["model"]["optimizer_config"]
        if hasattr(opt_config, "lr"):
            config_dict["model"]["optimizer_config"] = {
                "lr": float(opt_config.lr),
                "weight_decay": float(opt_config.weight_decay),
            }

    wandb_logger = WandbLogger(
        project="baseline-eval",
        name=f"baseline_eval_{model_name}",
        config=config_dict,
    )

    logger.info("Initializing trainer")
    trainer = L.Trainer(
        logger=[tensorboard_logger, wandb_logger],
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        enable_progress_bar=True,
        callbacks=[PredictionSaver(), AccentWERCallback()],
    )

    logger.info("Starting evaluation")
    results = trainer.test(model, data_module)

    logger.info("Evaluation results:")
    for metric, value in results[0].items():
        logger.info(f"{metric}: {value:.4f}")
        wandb_logger.experiment.log({f"test/{metric}": value})

    # Close wandb run
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
