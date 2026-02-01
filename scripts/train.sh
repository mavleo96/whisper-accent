#!/bin/bash

# Model and Dataset Arguments
export MODEL_NAME="openai/whisper-tiny.en"
export IS_MULTILINGUAL="False"
export DATASET_NAME="westbrook/English_Accent_DataSet"

# Checkpoint Arguments
export OUTPUT_DIR="checkpoints/whisper-accent-tiny.en"
export RUN_NAME="whisper-accent-tiny.en-$(date +%Y%m%d-%H%M%S)"
export HUB_MODEL_ID="mavleo96/whisper-accent-tiny.en"

# Wandb Arguments
export WANDB_PROJECT="whisper-accent"
export WANDB_ENTITY="mavleo96-team"

# optim_args={"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01},

python -m src.train \
    --model_name_or_path $MODEL_NAME \
    --is_multilingual $IS_MULTILINGUAL \
    --train_data_path $DATASET_NAME \
    --eval_data_path $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing False \
    --tf32 False \
    --bf16 True \
    --fp16 False \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_steps 10000 \
    --warmup_steps 500 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --eval_strategy steps \
    --eval_steps 100 \
    --predict_with_generate True \
    --logging_steps 10 \
    --run_name $RUN_NAME \
    --report_to tensorboard wandb \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 100 \
    --push_to_hub True \
    --hub_model_id $HUB_MODEL_ID \
    --hub_strategy "all_checkpoints" \
    --remove_unused_columns False
