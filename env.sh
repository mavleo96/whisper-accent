#! /bin/bash
set -euo pipefail

conda create -n asr python=3.12 -y
conda init bash
eval "$(conda shell.bash hook)"
conda activate asr
pip install torch tensorboard torchmetrics torchaudio torchcodec torchvision
pip install huggingface-hub transformers accelerate peft datasets
pip install ipykernel wandb ruff tqdm pre-commit
pip install scikit-learn scipy numpy pandas matplotlib seaborn
