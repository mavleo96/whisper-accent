# Accent Aware Speech Recognition
### Authors: Vijayabharathi Murugan, Jelwin Rodrigues, Gautham Narendra

## Quick Start

Setup:
```bash
conda env create -f env.yml
conda activate accent-asr
pre-commit install
huggingface-cli login
wandb login
```

Commands to run scripts for baseline and our methods:
```bash
python -m src.scripts.baseline_eval
python -m src.scripts.baseline_train_peft
python -m src.scripts.accent_token_train
python -m src.scripts.accent_token_train_peft
```

Machine Details:
```
Operating System: Ubuntu 22.04.1 LTS
Kernel: Linux 6.8.0-59-generic
GPU: NVIDIA GeForce RTX 3090
CUDA Version: 12.2
```

## References to where NLP Concepts were used
- Syntax | Classification: Tokenization to prepare model inputs, Regularization in mitigating overfitting
- Semantics | Probabilistic Model: Vector semantics topic concepts in Accent Embeddings
- Language Modeling | Transformers: Encoder Decoder models topic in modifying whisper
- Applications | Custom Statistical or Symbolic: Automatic Speech Recognition

## Project Structure
```
.
├── .gitignore
├── README.md
├── env.yml                   # Conda environment
├── .pre-commit-config.yaml   # Pre-commit to enfore styling
├── configs/                  # Hydra based configs for training / evaluation
└── src                       # Model directory
    ├── constants.py
    ├── callbacks/            # Callback objects for logging
    ├── data/                 # Data module
    ├── models
    │   ├── accent_token/               # Accent-Token model base class
    │   ├── accent_embedding.py         # Accent-Codebook model (not used)
    │   ├── accent_token_model_peft.py  # Accent-Token model with LoRa
    │   ├── accent_token_model.py       # Accent-Token model
    │   ├── base_model_peft.py          # Whisper model with LoRa
    │   └── base_model.py               # Whisper model
    └── scripts/                        # Contain script file to run / evaluate
```
