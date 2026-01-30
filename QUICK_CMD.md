### Evaluation

```bash
python scripts/eval.py \
    --model_name openai/whisper-tiny.en \
    --dataset_name westbrook/English_Accent_DataSet \
    --split test \
    --batch_size 4 \
    --device cuda \
    --output results/whisper-tiny.en.json
```

```bash
python scripts/create_initial_ckpt.py \
    --model_name openai/whisper-tiny.en \
    --output_dir checkpoints/whisper-accent-tiny.en
```
