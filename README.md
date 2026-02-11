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
CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
    --model_name /workspace/checkpoints/whisper-small-accent1/checkpoint-3000 \
    --dataset_name westbrook/English_Accent_DataSet \
    --split test \
    --batch_size 8 \
    --device cuda \
    --output results/whisper-small-accent1-3000.json


CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
    --model_name openai/whisper-small \
    --dataset_name westbrook/English_Accent_DataSet \
    --split test \
    --batch_size 8 \
    --device cuda \
    --output results/whisper-small.json

```


```bash
python scripts/whisper_finetune.py \
    --dataset_name common_voice \
    --model_name openai/whisper-small \
    --output_dir /workspace/checkpoints/whisper-small-hi \
    --language hi
```

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT="whisper-accent" WANDB_ENTITY="mavleo96-team" python scripts/whisper_finetune.py \
    --dataset_name westbrook \
    --model_name openai/whisper-small \
    --output_dir /workspace/checkpoints/whisper-small-accent1 \
    --language en \
    --device cuda \
    --run_name whisper-small-test-run-full-model-$(date +%Y%m%d-%H%M%S)

CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT="whisper-accent" WANDB_ENTITY="mavleo96-team" python scripts/whisper_finetune.py \
    --dataset_name westbrook \
    --model_name openai/whisper-small \
    --output_dir /workspace/checkpoints/whisper-small-accent2 \
    --language en \
    --device cuda \
    --freeze_encoder \
    --run_name whisper-small-test-run-decoder-only-$(date +%Y%m%d-%H%M%S)
```
