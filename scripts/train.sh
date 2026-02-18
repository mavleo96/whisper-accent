#!/bin/bash

# Model and Dataset Arguments
export MODEL_NAME="whisper-accent-small.en"
export MODEL_TYPE="whisper_accent"
export BASE_MODEL_NAME="openai/whisper-small.en"
export IS_MULTILINGUAL="False"
export DATASET_NAME="westbrook/English_Accent_DataSet"

# Checkpoint Arguments
export OUTPUT_DIR="/workspace/checkpoints/$MODEL_NAME"
export RUN_NAME="$MODEL_NAME-$(date +%Y%m%d-%H%M%S)"
export HUB_MODEL_ID="mavleo96/$MODEL_NAME"

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
    --optim adamw_torch \
    --learning_rate 5e-5 \
    --embedding_learning_rate 5e-4 \
    --accent_classifier_learning_rate 1e-3 \
    --lambda_accent 1.0 \
    --weight_decay 0.01 \
    --lr_scheduler_type linear \
    --warmup_steps 0.05 \
    --num_train_epochs 2 \
    --max_grad_norm 5.0 \
    --eval_strategy steps \
    --eval_steps 200 \
    --eval_on_start True \
    --predict_with_generate True \
    --logging_first_step True \
    --logging_steps 10 \
    --run_name $RUN_NAME \
    --report_to tensorboard wandb \
    --load_best_model_at_end True \
    --metric_for_best_model "wer" \
    --greater_is_better False \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 100 \
    --push_to_hub True \
    --hub_model_id $HUB_MODEL_ID \
    --hub_strategy "all_checkpoints" \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False
