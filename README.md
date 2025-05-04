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
python -m src.scripts.baseline_eval
```
