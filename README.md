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
```

Format predictions
```bash
python -m src.utils.format_predictions +log_dir=$LOG_DIR
```

```bash
python -m src.scripts.baseline_eval --multirun model.model_name=openai/whisper-base,openai/whisper-base.en,openai/whisper-small,openai/whisper-small.en,openai/whisper-medium,openai/whisper-medium.en
```
