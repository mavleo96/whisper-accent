# Whisper Accent — Accent-Aware English Speech Recognition

**Make Whisper significantly better at transcribing diverse English accents**
by conditioning the decoder on predicted accent embeddings via **Adaptive Layer Normalization (AdaLN)**.

Built on top of [OpenAI Whisper](https://github.com/openai/whisper) using [Hugging Face Transformers](https://github.com/huggingface/transformers).

![Architecture overview](assets/architecture.png)
*Accent embeddings are predicted from encoder hidden states (layer-weighted fusion + multi-head attention pooling) and used to modulate decoder LayerNorms via AdaLN.*

---

## Why This Project?

English speech recognition still struggles with **accent diversity** — Indian, Scottish, Nigerian, Vietnamese-influenced, and many others — even in state-of-the-art models like Whisper.

This project demonstrates how to **lightweight-condition** a frozen Whisper model on accent identity, achieving better word error rate (WER) across accents **without full fine-tuning**.

Key advantages:

- Only <10% of parameters are trainable (AdaLN modulation weights, accent embeddings, accent classifier)
- Encoder & decoder remain completely frozen → preserves original generalization capability
- Self-contained at inference: accent is predicted automatically from the audio

---

## Features

- Extends Whisper with **per-accent conditioning** via AdaLN in every decoder layer
- Accent identity predicted end-to-end from encoder hidden states:
  - Learnable weighted sum across all layers + input embeddings
  - Projection layer
  - Multi-head attention pooling over time
- Supports **23 English-accent varieties** (see full list below)
- Two model modes: `whisper_accent` (conditioned) vs `whisper` (baseline with standard LayerNorm)
- Evaluation reports **WER overall + per accent** and **accent classification accuracy**
- Ready multi-GPU training with Accelerate + evaluation scripts

**Supported accents**:

- American, British, Scottish, Irish, Canadian, Northern Irish
- Indian, Spanish-influenced, Dutch, German, Czech, Polish
- French, Italian, Hungarian, Finnish
- Vietnamese, Romanian, Slovak, Estonian, Lithuanian, Croatian, Slovene

---

## Quick Start

### Install

```bash
# Recommended: use conda
conda env create -f env.yml
conda activate whisper-accent

# Install pre-commit hooks (optional but recommended)
pre-commit install

# Login to Hugging Face (needed for model weights & some datasets)
huggingface-cli login

# Optional: Weights & Biases for logging
wandb login
```

### Training

```bash
# Uses accelerate for multi-GPU / mixed precision
bash scripts/train.sh
```

Default config trains only the new components (AdaLN, accent embeddings, classifier) while keeping Whisper frozen.

### Evaluation

Single model:

```bash
python scripts/eval.py \
  --model_name_or_path your/checkpoint-or-hf-name \
  --dataset_name westbrook/English_Accent_DataSet \
  --split test \
  --batch_size 8 \
  --output_dir results/
```

Batch-evaluate multiple checkpoints:

```bash
bash scripts/eval_all.sh
```

Results include JSON with overall WER, per-accent WER, accent accuracy, and per-sample predictions.

---

## Project Structure

```
whisper-accent/
├── src/
│   ├── model/              # WhisperAccentModel, AccentClassifier, AdaLN layers
│   ├── train/              # Dataset, data collator, custom Trainer
│   ├── utils/
│   └── constants.py
├── scripts/
│   ├── train.sh            # Launch training with accelerate config
│   ├── eval.py             # Single-model evaluation
│   └── eval_all.sh         # Batch evaluation across models
├── assets/                 # architecture.png, sample results
├── env.yml
├── pyproject.toml
└── README.md
```

---

## Advanced Usage & Tips

- **Frozen backbone**: Encoder & decoder weights are never updated. It is recommended to set dropout in the frozen Whisper parts to 0.0 during training for cleaner, more deterministic features.
- **Hyperparameters**:
  - `lambda_accent`: Accent auxiliary loss weight (start with 0.1–1.0)
  - `max_grad_norm`: AdaLN layers are zero-initialized while you need a high learning rate for accent classifier, so you need to set this carefully (default is 5.0)
  - `learning_rate`: Classifier / Accent embedding learning rate (often higher than main LR)
- **Custom datasets**: Any dataset with audio + text + accent_label column works (modify data collator accordingly)

---

## License

Model weights: inherit from original Whisper (Apache 2.0)

---
