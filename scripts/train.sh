#!/bin/bash

# Model and Dataset Arguments
export MODEL_TYPE="whisper_accent"
export BASE_MODEL_NAME="openai/whisper-tiny.en"
export IS_MULTILINGUAL="False"
export DATASET_NAME="westbrook/English_Accent_DataSet"

# Checkpoint Arguments
export OUTPUT_DIR="checkpoints/whisper-accent-tiny.en"
export RUN_NAME="whisper-accent-tiny.en-$(date +%Y%m%d-%H%M%S)"
export HUB_MODEL_ID="mavleo96/whisper-accent-tiny.en"

# Wandb Arguments
export WANDB_PROJECT="whisper-accent"
export WANDB_ENTITY="mavleo96-team"

python -m src.train \
    --model_type $MODEL_TYPE \
    --base_model_name_or_path $BASE_MODEL_NAME \
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
    --lambda_accent_loss 0.01 \
    --lambda_diversity_loss 0.01 \
    --optim adamw_torch \
    --learning_rate 1e-5 \
    --embedding_learning_rate 1e-4 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --warmup_steps 0.05 \
    --max_steps 10000 \
    --max_grad_norm 1.0 \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_bias "none" \
    --use_rslora True \
    --task_type "SEQ_2_SEQ_LM" \
    --eval_strategy steps \
    --eval_steps 100 \
    --eval_on_start True \
    --predict_with_generate True \
    --logging_first_step True \
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
