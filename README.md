# Accent Aware Speech Recognition

## Quick Start

Setup
```bash
conda env create -f env.yml
conda activate accent-asr
pre-commit install
huggingface-cli login
```

Evaluate baseline
```bash
python -m src.scripts.baseline_eval +data.subset_mode=$DEV_MODE_BOOL
python -m src.scripts.baseline_train +data.subset_mode=$DEV_MODE_BOOL
python -m src.scripts.accent_token_train +data.subset_mode=$DEV_MODE_BOOL
```

```bash
python -m src.scripts.baseline_eval --multirun model.model_name=openai/whisper-base,openai/whisper-base.en,openai/whisper-small,openai/whisper-small.en,openai/whisper-medium,openai/whisper-medium.en
```
```bash
python -m src.scripts.baseline_train model.optimizer_config.lr=1e-5 model.optimizer_config.weight_decay=0.1 trainer.max_epochs=2
```

python -m src.scripts.baseline_eval +data.subset_mode=True

python -m src.scripts.accent_token_train

python -m src.models.accent_embedding
