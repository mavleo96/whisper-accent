import logging
import os
import sys
from datetime import datetime

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.callbacks.prediction_logger import PredictionLogger
from src.data.data_module import EdaccDataModule
from src.models.accent_aware_model import AccentAwareWhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


@hydra.main(
    config_path="../../configs/trainer",
    config_name="accent_train.yaml",
    version_base=None,
)
def main(cfg):
    # Create unique experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = cfg.model.model_name.split("/")[-1]
    experiment_name = f"{model_name}_{timestamp}"

    # Set up model save directory
    save_dir = os.path.join(cfg.trainer.save_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Model will be saved to: {save_dir}")

    # Initialize model
    logger.info(f"Initializing model: {cfg.model.model_name}")
    model = AccentAwareWhisperModel(**cfg.model)

    # Initialize data module
    logger.info("Initializing data module")
    data_module = EdaccDataModule(**cfg.data)

    # Set up TensorBoard logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=cfg.trainer.logger[0].save_dir,
        name=experiment_name,
    )

    # Set up callbacks
    callbacks = [
        # # Save best model based on validation WER
        # ModelCheckpoint(
        #     dirpath=save_dir,
        #     filename="{epoch}-{val_wer:.4f}",
        #     monitor="val_wer",
        #     mode="min",
        #     save_top_k=3,
        #     verbose=True,
        # ),
        # Save last model
        ModelCheckpoint(
            dirpath=save_dir,
            filename="last-{epoch}",
            save_last=True,
            verbose=True,
        ),
        # # Early stopping based on validation WER
        # EarlyStopping(
        #     monitor="val_wer",
        #     patience=cfg.trainer.early_stopping_patience,
        #     mode="min",
        #     verbose=True,
        # ),
        # # Learning rate monitor
        # LearningRateMonitor(logging_interval="step"),
        # Custom prediction logger
        PredictionLogger(),
    ]

    # Initialize trainer
    logger.info("Initializing trainer")
    trainer = L.Trainer(
        logger=[tensorboard_logger],
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        enable_progress_bar=True,
        callbacks=callbacks,
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
    )

    # Train the model
    logger.info("Starting training")
    trainer.fit(model, data_module)

    # Save the final model explicitly (in addition to checkpoints)
    final_model_path = os.path.join(save_dir, "final_model")
    model.model.save_pretrained(final_model_path)
    model.processor.save_pretrained(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    # Test the model on the test set
    logger.info("Starting evaluation on test set")
    results = trainer.test(model, data_module)

    # Log test results
    logger.info("Test results:")
    for metric, value in results[0].items():
        if isinstance(value, float):
            logger.info(f"{metric}: {value:.4f}")
        else:
            logger.info(f"{metric}: {value}")

    # Also save test results to a file
    results_path = os.path.join(save_dir, "test_results.txt")
    with open(results_path, "w") as f:
        for metric, value in results[0].items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: {value}\n")
    logger.info(f"Test results saved to: {results_path}")

    # Return best checkpoint path for potential further use
    best_checkpoint = None
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint) and not callback.save_last:
            best_checkpoint = callback.best_model_path

    if best_checkpoint:
        logger.info(f"Best checkpoint: {best_checkpoint}")
        return best_checkpoint


if __name__ == "__main__":
    main()
