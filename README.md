# Deep Learning Project Template

A template for deep learning projects with PyTorch Lightning, Hydra, and experiment tracking.

## Features

- PyTorch Lightning
- Weights & Biases and TensorBoard
- Hydra configuration
- Testing & linting

## Quick Start

```bash
# Setup
conda env create -f env.yml
conda activate dl-template
pre-commit install
wandb login

# Train
python src/training/train.py

# Override config
python src/training/train.py model.learning_rate=0.0001
```

## Project Structure

```
.
├── configs/          # Hydra configs
├── src/             # Source code
├── tests/           # Tests
└── outputs/         # Experiment outputs
```

## Configuration

- `configs/config.yaml`: Main config
- `configs/model/`: Model configs
- `configs/data/`: Data configs
- `configs/training/`: Training configs
- `configs/logging/`: Logging configs
