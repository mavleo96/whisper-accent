import lightning as L
from torch.utils.data import DataLoader, Dataset
import torch
from omegaconf import DictConfig


class BaseDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.data.batch_size
        self.num_workers = cfg.data.num_workers
        
    def prepare_data(self):
        """Download or prepare data. Override this method in child classes."""
        pass
        
    def setup(self, stage=None):
        """Set up datasets. Override this method in child classes."""
        pass
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        ) 