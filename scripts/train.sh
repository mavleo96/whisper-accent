#!/bin/bash

# Model and Dataset Arguments
export MODEL_TYPE="whisper_accent"
export BASE_MODEL_NAME="openai/whisper-small.en"
export IS_MULTILINGUAL="False"
export DATASET_NAME="westbrook/English_Accent_DataSet"

# Checkpoint Arguments
export OUTPUT_DIR="/workspace/checkpoints/whisper-accent-small.en"
export RUN_NAME="whisper-accent-small.en-decoder-only-baserun-$(date +%Y%m%d-%H%M%S)"
export HUB_MODEL_ID="mavleo96/whisper-accent-small.en"

# Wandb Arguments
export WANDB_PROJECT="whisper-accent"
export WANDB_ENTITY="mavleo96-team"

accelerate launch \
    --num_processes 2 \
    --num_machines 1 \
    --mixed_precision fp16 \
    --gpu_ids 0,1 \
    -m src.train \
    --model_type $MODEL_TYPE \
    --base_model_name_or_path $BASE_MODEL_NAME \
    --is_multilingual $IS_MULTILINGUAL \
    --train_data_path $DATASET_NAME \
    --eval_data_path $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --lambda_accent_loss 0.0 \
    --lambda_diversity_loss 0.0 \
    --optim adamw_torch \
    --learning_rate 1e-5 \
    --embedding_learning_rate 5e-5 \
    --weight_decay 0.1 \
    --lr_scheduler_type linear \
    --warmup_steps 0.05 \
    --max_steps 5000 \
    --max_grad_norm 1.0 \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.2 \
    --lora_bias "none" \
    --use_rslora True \
    --task_type "SEQ_2_SEQ_LM" \
    --eval_strategy steps \
    --eval_steps 200 \
    --eval_on_start True \
    --predict_with_generate True \
    --logging_first_step True \
    --logging_steps 10 \
    --run_name $RUN_NAME \
    --report_to tensorboard wandb \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 100 \
    --push_to_hub True \
    --hub_model_id $HUB_MODEL_ID \
    --hub_strategy "all_checkpoints" \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False
