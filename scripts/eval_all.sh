#!/bin/bash

export DATASET_NAME="westbrook/English_Accent_DataSet"

MODELS=(
    "openai/whisper-small.en"
    "mavleo96/whisper-accent-small.en"
    "openai/whisper-medium.en"
    # "mavleo96/whisper-accent-medium.en"
    "openai/whisper-large-v3"
    "openai/whisper-large-v3-turbo"
)

for model in "${MODELS[@]}"; do
    name="${model//\//-}"
    output="results/${name}.json"
    echo "Evaluating $model -> $output"
    CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
        --model_name "$model" \
        --dataset_name $DATASET_NAME \
        --split test \
        --batch_size 4 \
        --output "$output"
done
