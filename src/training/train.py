import os
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.loggers import TensorBoardLogger, WandbLogger
import wandb

from src.models.base_model import BaseModel


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print configuration
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize wandb
    wandb.init(
        project=cfg.logging.wandb.project,
        entity=cfg.logging.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # Initialize model
    model = BaseModel(cfg)
    
    # Initialize loggers
    tensorboard_logger = TensorBoardLogger(
        save_dir=cfg.logging.tensorboard.log_dir,
        name=cfg.model.name
    )
    wandb_logger = WandbLogger()
    
    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.hydra.run.dir, "checkpoints"),
        filename=f"{cfg.model.name}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=cfg.training.early_stopping_patience,
        mode='min'
    )
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        logger=[tensorboard_logger, wandb_logger],
        callbacks=[checkpoint_callback, early_stopping]
    )
    
    # Train model
    trainer.fit(model)
    
    # Close wandb run
    wandb.finish()


if __name__ == '__main__':
    main() 