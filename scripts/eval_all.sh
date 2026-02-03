#!/bin/bash

MODELS=(
    "openai/whisper-tiny"
    "openai/whisper-tiny.en"
    "openai/whisper-base"
    "openai/whisper-base.en"
    "openai/whisper-small"
    "openai/whisper-small.en"
    "openai/whisper-medium"
    "openai/whisper-medium.en"
    "openai/whisper-large-v3"
    "openai/whisper-large-v3-turbo"
)

for model in "${MODELS[@]}"; do
    name="${model#openai/}"
    output="results/${name}.json"
    echo "Evaluating $model -> $output"
    python scripts/eval.py \
        --model_name "$model" \
        --dataset_name westbrook/English_Accent_DataSet \
        --split test \
        --batch_size 4 \
        --device cuda \
        --output "$output"
done
