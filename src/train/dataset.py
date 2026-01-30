from typing import Dict

import torch
from datasets import Audio, load_dataset
from torch.utils.data import Dataset
from transformers import WhisperProcessor

from src.constants import SAMPLING_RATE, WESTBROOK_DATASET_ACCENT_MAP
from src.model.tokenization import ACCENTS


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

        print(f"Loaded {len(self.raw_dataset)} examples for Whisper fine-tuning")

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
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
