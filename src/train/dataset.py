import logging
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Audio, load_dataset
from torch.utils.data import Dataset
from transformers import WhisperProcessor

from src.constants import IGNORE_INDEX, SAMPLING_RATE, WESTBROOK_DATASET_ACCENT_MAP
from src.model.tokenization import ACCENTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        processor: WhisperProcessor,
        split: str = "train",
        shuffle: bool = False,
    ):
        super().__init__()

        self.tokenizer = processor.tokenizer
        self.feature_extractor = processor.feature_extractor

        # Load data
        self.raw_dataset = load_dataset(data_path, split=split).cast_column(
            "audio", Audio(sampling_rate=SAMPLING_RATE)
        )

        # Token ids
        self.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|startoftranscript|>"
        )
        self.translate_token_id = self.tokenizer.convert_tokens_to_ids("<|translate|>")
        self.transcribe_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|transcribe|>"
        )
        self.no_timestamps_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|notimestamps|>"
        )
        self.eos_token_id = self.tokenizer.eos_token_id

        if shuffle:
            self.raw_dataset.shuffle()

        logger.info(f"Loaded {len(self.raw_dataset)} examples for Whisper fine-tuning")

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        item = self.raw_dataset[i]

        # Extract audio and text
        text = item["raw_text"]
        audio_array = item["audio"]["array"]
        sampling_rate = item["audio"]["sampling_rate"]

        # Process audio with feature extractor
        features = self.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_attention_mask=True,
        )
        input_features = features["input_features"][0]
        attention_mask = features["attention_mask"][0]

        # Process text with tokenizer
        # Normalize text
        text = self.tokenizer.normalize(text)

        # Tokenize text labels
        accent_token_id = self._convert_accent_to_token_id(item["accent"])
        prefix_tokens = [accent_token_id, self.no_timestamps_token_id]
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.tokenizer.model_max_length - len(prefix_tokens),
        ).input_ids
        labels = prefix_tokens + tokens
        if len(labels) < self.tokenizer.model_max_length:
            labels.append(self.eos_token_id)

        return {
            "labels": labels,
            "input_features": input_features,
            "attention_mask": attention_mask,
        }

    def _convert_accent_to_token_id(self, accent_idx: int) -> int:
        return self.tokenizer.convert_tokens_to_ids(
            ACCENTS[WESTBROOK_DATASET_ACCENT_MAP[accent_idx]]
        )


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int | None

    def __call__(self, features):
        batch = {}
        # Pad audio input features
        input_features_list = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch["input_features"] = self.processor.feature_extractor.pad(
            input_features_list, return_tensors="pt"
        )["input_features"]

        # Merge attention masks
        batch["attention_mask"] = torch.tensor(
            np.array([i["attention_mask"] for i in features])
        )

        # Pad labels
        labels_list = [{"input_ids": feature["labels"]} for feature in features]
        labels = self.processor.tokenizer.pad(
            labels_list, return_tensors="pt", return_attention_mask=True
        )

        # Replace padding tokens in labels with -100 to ignore in loss computation
        batch["labels"] = labels["input_ids"].masked_fill(
            labels["attention_mask"].ne(1), IGNORE_INDEX
        )

        # # If decoder_start_token_id is provided and all sequences start with it,
        # # remove it since it will be added during forward pass
        # if self.decoder_start_token_id is not None:
        #     check_bos_token = (
        #         batch["labels"].eq(self.decoder_start_token_id)[:, 0].all().item()
        #     )
        #     if batch["labels"].shape[0] > 0 and check_bos_token:
        #         batch["labels"] = batch["labels"][:, 1:]
        #         batch["attention_mask"] = batch["attention_mask"][:, 1:]

        return batch
