#!/bin/bash
set -euo pipefail

conda create -n whisper-accent python=3.12 -y && \
conda init bash && \
eval "$(conda shell.bash hook)" && \
conda activate whisper-accent && \
pip install -U torch \
 torchaudio \
 torchcodec \
 torchvision \
 huggingface-hub \
 transformers \
 accelerate \
 peft \
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
 matplotlib \
 seaborn

conda env export --no-builds | grep -v "^prefix: " > env.yml
