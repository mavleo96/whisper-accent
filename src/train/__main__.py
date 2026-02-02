import logging

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
    # Suppress HTTP request logging from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = HfArgumentParser(
        [
            ModelArguments,
            DatasetArguments,
            WhisperAccentTrainingArguments,
            LoraArguments,
        ]
    )
    model_args, dataset_args, training_args, lora_args = (
        parser.parse_args_into_dataclasses()
    )

    # Load model and processor; whisper / whisper_accent models are supported
    # Note: Non-LoRA training is not implemented yet
    logger.info(
        f"Loading {model_args.model_type} model from"
        f" {model_args.base_model_name_or_path}"
    )
    if model_args.model_type == "whisper":
        processor = WhisperProcessor.from_pretrained(model_args.base_model_name_or_path)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_args.base_model_name_or_path
        )
        if lora_args.lora_enable:
            target_modules = []
            for name, _ in model.named_modules():
                m_list = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
                if "model.decoder" in name and any(suffix in name for suffix in m_list):
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
            model = get_peft_model(model, lora_config)
        else:
            raise NotImplementedError("Non-LoRA training is not implemented yet")
    elif model_args.model_type == "whisper_accent":
        processor = processor_init(model_args)
        model = model_init(model_args, lora_args, processor)
    else:
        raise ValueError(f"Invalid model type: {model_args.model_type}")

    if lora_args.lora_enable:
        logger.info("Lora enabled")
        model.print_trainable_parameters()

    # Initialize datasets and data collator
    logger.info("Initializing datasets and data collator")
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    train_dataset = WhisperDataset(
        dataset_args.train_data_path,
        processor,
        multilingual_model=model_args.is_multilingual,
        split="train",
        shuffle=True,
        num_proc=dataset_args.num_proc,
    )
    eval_dataset = WhisperDataset(
        dataset_args.eval_data_path,
        processor,
        multilingual_model=model_args.is_multilingual,
        split="validation",
        shuffle=False,
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
    )

    # trainer.train()
    print(trainer.model)


if __name__ == "__main__":
    main()
