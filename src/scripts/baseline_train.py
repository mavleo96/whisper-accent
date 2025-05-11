import logging
import os

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
    config_path="../../configs",
    config_name="baseline_train.yaml",
    version_base=None,
)
def main(cfg):
    logger.info(f"Initializing model: {cfg.model.model_name}")
    model = BaseWhisperModel(**cfg.model)

    logger.info("Initializing data module")
    data_module = EdaccDataModule(**cfg.data)

    model_name = cfg.model.model_name.split("/")[-1]
    tensorboard_logger = TensorBoardLogger(
        save_dir=cfg.trainer.logger[0].save_dir,
        name=f"baseline_train_{model_name}",
    )

    # Convert config to dict and handle optimizer config
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    if "model" in config_dict and "optimizer_config" in config_dict["model"]:
        opt_config = config_dict["model"]["optimizer_config"]
        if hasattr(opt_config, "lr"):
            config_dict["model"]["optimizer_config"] = {
                "lr": float(opt_config.lr),
                "weight_decay": float(opt_config.weight_decay),
            }

    wandb_logger = WandbLogger(
        project="baseline-train",
        name=f"baseline_train_{model_name}",
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
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
    )

    logger.info("Starting training")
    trainer.fit(model, data_module)

    logger.info("Starting testing")
    trainer.test(model, data_module)

    logger.info("Saving final model")
    save_dir = os.path.join(tensorboard_logger.log_dir, "final_model")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "final_model.ckpt")
    trainer.save_checkpoint(checkpoint_path)

    # Close wandb run
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
