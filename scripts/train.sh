#!/bin/bash

# Multi-device CUDA training (GPUs 0 and 1)
export CUDA_VISIBLE_DEVICES=0,1
NPROC_PER_NODE=2

# Model and Dataset Arguments
export MODEL_TYPE="whisper"
export BASE_MODEL_NAME="openai/whisper-medium.en"
export IS_MULTILINGUAL="False"
export DATASET_NAME="westbrook/English_Accent_DataSet"

# Checkpoint Arguments
export OUTPUT_DIR="/workspace/checkpoints/whisper-medium.en"
export RUN_NAME="whisper-medium.en-test-run-$(date +%Y%m%d-%H%M%S)"
export HUB_MODEL_ID="mavleo96/whisper-medium.en"

# Wandb Arguments
export WANDB_PROJECT="whisper-accent"
export WANDB_ENTITY="mavleo96-team"

torchrun --nproc_per_node=$NPROC_PER_NODE -m src.train \
    --model_type $MODEL_TYPE \
    --base_model_name_or_path $BASE_MODEL_NAME \
    --is_multilingual $IS_MULTILINGUAL \
    --train_data_path $DATASET_NAME \
    --eval_data_path $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --tf32 False \
    --bf16 True \
    --fp16 False \
    --lambda_accent_loss 0.0 \
    --lambda_diversity_loss 0.0 \
    --optim adamw_torch \
    --learning_rate 3e-6 \
    --embedding_learning_rate 0.0 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --warmup_steps 0.05 \
    --max_steps 1000 \
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
