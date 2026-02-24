#!/bin/bash
set -euo pipefail

ENV_NAME="whisper-accent"

conda create -n $ENV_NAME python=3.12 -y && \
conda init bash && \
eval "$(conda shell.bash hook)" && \
conda activate $ENV_NAME && \
pip install -U torch \
 torchaudio \
 torchcodec \
 torchvision \
 huggingface-hub \
 transformers \
 accelerate \
 datasets \
 evaluate \
 jiwer \
 ipykernel \
 wandb \
 ruff \
 tqdm \
 pre-commit \
 tensorboard \
 scikit-learn \
 scipy \
 numpy \
 pandas \
 umap-learn \
 matplotlib \
 seaborn \
 adjustText

conda env export --no-builds | grep -v "^prefix: " > env.yml
