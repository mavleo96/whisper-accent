#! /bin/bash
set -euo pipefail

conda create -n asr python=3.12 -y
conda init bash
eval "$(conda shell.bash hook)"
conda activate asr
pip install torch lightning tensorboard torchmetrics torchaudio torchcodec torchvision
pip install huggingface-hub transformers accelerate peft datasets
pip install hydra-core ipykernel wandb ruff tqdm
pip install scikit-learn scipy numpy pandas matplotlib seaborn
pip install pre-commit
