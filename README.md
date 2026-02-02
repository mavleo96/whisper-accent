# Accent Aware Speech Recognition

## Quick Start

### Setup
```bash
conda env create -f env.yml
conda activate whisper-accent
pre-commit install
hf auth login
wandb login
```

### Evaluate
```bash
python scripts/eval.py \
    --model_name openai/whisper-tiny.en \
    --dataset_name westbrook/English_Accent_DataSet \
    --split test \
    --batch_size 8 \
    --device cuda \
    --output results/whisper-tiny.en.json
```
