import logging
from functools import partial

from transformers import HfArgumentParser

from .dataset import DataCollatorSpeechSeq2SeqWithPadding, WhisperDataset
from .train import (
    DatasetArguments,
    LoraArguments,
    ModelArguments,
    WhisperAccentTrainingArguments,
    model_init,
    processor_init,
)
from .trainer import WhisperAccentTrainer, compute_metrics


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
    print(training_args)
    print(dataset_args)
    print(model_args)
    print(lora_args)

    logger.info(
        f"Initializing processor and model from {model_args.model_name_or_path}"
    )
    processor = processor_init(model_args)
    model = model_init(model_args, lora_args, processor)
    if lora_args.lora_enable:
        logger.info("Lora enabled")
        model.print_trainable_parameters()

    logger.info("Initializing datasets and data collator")
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    train_dataset = WhisperDataset(
        dataset_args.train_data_path,
        processor,
        split="train",
        shuffle=True,
        num_proc=dataset_args.num_proc,
    )
    eval_dataset = WhisperDataset(
        dataset_args.eval_data_path,
        processor,
        split="validation",
        shuffle=False,
        num_proc=dataset_args.num_proc,
    )

    logger.info("Initializing trainer")
    compute_metrics_fn = partial(compute_metrics, processor=processor)
    trainer = WhisperAccentTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        compute_metrics=compute_metrics_fn,
    )

    # trainer.train()
    print(trainer.model)


if __name__ == "__main__":
    main()
