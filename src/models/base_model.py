import lightning as L
import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import DictConfig


class BaseModel(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        
        # Initialize model components
        self.model = self._build_model()
        self.criterion = self._build_criterion()
        
    def _build_model(self):
        """Build the model architecture. Override this method in child classes."""
        raise NotImplementedError
        
    def _build_criterion(self):
        """Build the loss function. Override this method in child classes."""
        raise NotImplementedError
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Log training metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.current_epoch % 10 == 0:  # Log less frequently to avoid clutter
            wandb.log({"train_loss": loss})
            
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Log validation metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.current_epoch % 10 == 0:
            wandb.log({"val_loss": loss})
            
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.model.learning_rate,
            weight_decay=self.cfg.model.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.training.max_epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        } 