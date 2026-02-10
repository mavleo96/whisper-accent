import logging
import os
import random

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from transformers import (
    HfArgumentParser,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from .dataset import DataCollatorSpeechSeq2SeqWithPadding, WhisperDataset
from .train import (
    DatasetArguments,
    LoraArguments,
    ModelArguments,
    WhisperAccentTrainingArguments,
    model_init,
    processor_init,
)
from .trainer import WhisperAccentTrainer


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Multi-device: use CUDA_VISIBLE_DEVICES (e.g. 0,1) and launch with torchrun --nproc_per_node=N
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        logger.info(f"Distributed training: local_rank={local_rank}, world_size={world_size}")
    else:
        logger.info(f"Training on {torch.cuda.device_count()} GPU(s)")

    parser = HfArgumentParser(
        [
            ModelArguments,
            DatasetArguments,
            WhisperAccentTrainingArguments,
            LoraArguments,
        ]
    )
    model_args, dataset_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # Load model and processor; whisper / whisper_accent models are supported
    logger.info(f"Loading {model_args.model_type} model from {model_args.base_model_name_or_path}")
    if model_args.model_type == "whisper_accent":
        processor = processor_init(model_args.base_model_name_or_path)
        model = model_init(model_args.base_model_name_or_path, processor)
    elif model_args.model_type == "whisper":
        processor = WhisperProcessor.from_pretrained(model_args.base_model_name_or_path)
        model = WhisperForConditionalGeneration.from_pretrained(model_args.base_model_name_or_path)
        # Update generation config; https://github.com/openai/whisper/discussions/2094
        if model.generation_config.is_multilingual:
            model.generation_config.language = "en"
            model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None
    else:
        raise ValueError(f"Invalid model type: {model_args.model_type}")

    # Add LoRA layers
    # Note: Non-LoRA training is not implemented yet
    if lora_args.lora_enable:
        # Target linear layers
        target_modules = []
        m_list = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        for name, _ in model.named_modules():
            if any(suffix in name for suffix in m_list):
                target_modules.append(name)

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            use_rslora=lora_args.use_rslora,
            target_modules=target_modules,
            task_type=lora_args.task_type,
            ensure_weight_tying=True,
        )

        # Trainable token indices for new accent tokens
        if model_args.model_type == "whisper_accent":
            accent_token_indices = sorted(list(model.generation_config.accent_to_id.values()))
            lora_config.trainable_token_indices = accent_token_indices

        model = get_peft_model(model, lora_config)
        logger.info("Lora enabled")
        model.print_trainable_parameters()
    else:
        raise NotImplementedError("Non-LoRA training is not implemented yet")

    # Initialize datasets and data collator
    logger.info("Initializing datasets and data collator")
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    train_dataset = WhisperDataset(
        dataset_args.train_data_path,
        "train",
        processor,
        multilingual_model=model_args.is_multilingual,
        num_proc=dataset_args.num_proc,
    )
    eval_dataset = WhisperDataset(
        dataset_args.eval_data_path,
        "validation",
        processor,
        multilingual_model=model_args.is_multilingual,
        num_proc=dataset_args.num_proc,
    )

    # Initialize trainer
    logger.info("Initializing trainer")
    trainer = WhisperAccentTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        compute_metrics="all" if model_args.model_type == "whisper_accent" else "wer",
    )

    trainer.train()
    trainer.evaluate()
    trainer.push_to_hub(
        language="en",
        license="apache-2.0",
        dataset=dataset_args.train_data_path,
    )

    # Ensure distributed process group is properly destroyed to avoid NCCL warnings on exit
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
