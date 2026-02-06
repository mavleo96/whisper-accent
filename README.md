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


```bash
python scripts/whisper_finetune.py \
    --dataset_name common_voice \
    --model_name openai/whisper-small \
    --output_dir /workspace/checkpoints/whisper-small-hi \
    --language hi
```

```bash
python scripts/whisper_finetune.py \
    --dataset_name westbrook \
    --model_name openai/whisper-small \
    --output_dir /workspace/checkpoints/whisper-small-accent \
    --language en
```
