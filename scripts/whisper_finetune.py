import argparse
from dataclasses import dataclass
from functools import partial
from typing import Any

import evaluate
import torch
from datasets import Audio, DatasetDict, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

NUM_PROC = 4


def prepare_dataset(batch, processor):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    if "raw_text" in batch:
        batch["labels"] = processor.tokenizer(batch["raw_text"]).input_ids
    else:
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


# Create dataset
def create_common_voice_dataset(processor):
    dataset = DatasetDict()
    # `verification_mode="no_checks"` avoids NonMatchingSplitsSizesError when the remote
    # dataset has changed since the local cached metadata was created.
    dataset["train"] = load_dataset(
        "fixie-ai/common_voice_17_0",
        "hi",
        split="train+validation",
        verification_mode="no_checks",
    )
    dataset["validation"] = load_dataset(
        "fixie-ai/common_voice_17_0",
        "hi",
        split="test",
        verification_mode="no_checks",
    )
    print(dataset)
    dataset = dataset.remove_columns(
        [
            "accent",
            "age",
            "client_id",
            "down_votes",
            "gender",
            "locale",
            "path",
            "segment",
            "up_votes",
        ]
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    prepare_dataset_fn = partial(prepare_dataset, processor=processor)
    dataset = dataset.map(prepare_dataset_fn, remove_columns=dataset.column_names["train"])
    return dataset


# Create dataset
def create_westbrook_dataset(processor):
    dataset = DatasetDict()
    dataset["train"] = load_dataset("westbrook/English_Accent_DataSet", split="train")
    dataset["validation"] = load_dataset("westbrook/English_Accent_DataSet", split="validation")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def is_valid_audio(item):
        try:
            audio = item["audio"]["array"]
            return audio is not None and len(audio) > 0
        except Exception:
            return False

    dataset = dataset.filter(is_valid_audio, num_proc=NUM_PROC)

    prepare_dataset_fn = partial(prepare_dataset, processor=processor)
    dataset = dataset.map(prepare_dataset_fn, remove_columns=dataset.column_names["train"])
    return dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: list[dict[str, list[int] | torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        labels = labels[:, :448]

        batch["labels"] = labels

        return batch


def compute_metrics(pred, compute_result=False, processor=None, metric=None):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [processor.tokenizer.normalize(s) for s in pred_str]
    label_str = [processor.tokenizer.normalize(s) for s in label_str]

    metric.add_batch(predictions=pred_str, references=label_str)

    if compute_result:
        wer = 100 * metric.compute()
        return {"wer": wer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="common_voice")
    parser.add_argument("--model_name", type=str, default="openai/whisper-small")
    parser.add_argument("--output_dir", type=str, default="./whisper-small-hi")
    parser.add_argument("--language", type=str, default="hi")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default="whisper-small-hi")
    args = parser.parse_args()

    # Initialize processor
    processor = WhisperProcessor.from_pretrained(
        args.model_name, language=args.language, task="transcribe"
    )

    # Load model; let Transformers/Accelerate handle device placement
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
    )
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.to(args.device)

    if args.freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    if args.dataset_name == "common_voice":
        dataset = create_common_voice_dataset(processor)
    elif args.dataset_name == "westbrook":
        dataset = create_westbrook_dataset(processor)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    metric = evaluate.load("wer")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=3000,
        eval_on_start=True,
        gradient_checkpointing=True,
        bf16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=250,
        logging_steps=10,
        report_to=["tensorboard", "wandb"],
        run_name=args.run_name,
        load_best_model_at_end=True,
        batch_eval_metrics=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    compute_metrics_fn = partial(compute_metrics, processor=processor, metric=metric)
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        processing_class=processor,
    )

    trainer.train()


if __name__ == "__main__":
    main()
