import pytest
import torch
from src.models.base_model import BaseModel


class TestModel(BaseModel):
    def _build_model(self):
        return torch.nn.Linear(10, 1)
        
    def _build_criterion(self):
        return torch.nn.MSELoss()


def test_model_initialization():
    config = {
        'model': {
            'learning_rate': 0.001,
            'weight_decay': 0.0001
        },
        'training': {
            'max_epochs': 100
        }
    }
    
    model = TestModel(config)
    assert isinstance(model, BaseModel)
    assert isinstance(model.model, torch.nn.Linear)
    assert isinstance(model.criterion, torch.nn.MSELoss)


def test_model_forward():
    config = {
        'model': {
            'learning_rate': 0.001,
            'weight_decay': 0.0001
        },
        'training': {
            'max_epochs': 100
        }
    }
    
    model = TestModel(config)
    x = torch.randn(32, 10)
    y = model(x)
    assert y.shape == (32, 1) 