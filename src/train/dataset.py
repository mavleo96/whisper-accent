import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import Audio, Value, load_dataset
from torch.utils.data import Dataset
from transformers import WhisperProcessor

from src.constants import (
    IGNORE_INDEX,
    MAX_LENGTH,
    SAMPLING_RATE,
    WESTBROOK_DATASET_ACCENT_MAP,
)
from src.model.tokenization import ACCENTS, WhisperAccentTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Suppress HTTP request logging from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)


class WhisperDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        processor: WhisperProcessor,
        multilingual_model: bool = False,
        shuffle: bool = False,
        num_proc: int = 16,
    ):
        super().__init__()

        self.tokenizer = processor.tokenizer
        self.feature_extractor = processor.feature_extractor
        self.multilingual_model = multilingual_model

        # Load data
        self.raw_dataset = load_dataset(data_path, split=split, num_proc=num_proc)
        self.raw_dataset = self.raw_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        self.raw_dataset = self.raw_dataset.cast_column("accent", Value("string"))
        self.raw_dataset = self.raw_dataset.map(
            lambda x: {"accent": WESTBROOK_DATASET_ACCENT_MAP[int(x["accent"])]}
        )

        def is_valid_audio(item):
            try:
                audio = item["audio"]["array"]
                return audio is not None and len(audio) > 0
            except Exception:
                return False

        self.raw_dataset = self.raw_dataset.filter(is_valid_audio, num_proc=num_proc)

        # Token ids
        self.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self.transcribe_token_id = self.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        self.no_timestamps_token_id = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self.eos_token_id = self.tokenizer.eos_token_id
        self.en_lang_token_id = self.tokenizer.convert_tokens_to_ids("<|en|>")

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
        prefix_tokens = self.get_prefix_tokens(item)
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH - len(prefix_tokens),
        ).input_ids
        labels = prefix_tokens + tokens
        if len(labels) < MAX_LENGTH:
            labels.append(self.eos_token_id)

        return {
            "labels": labels,
            "input_features": input_features,
            "attention_mask": attention_mask,
        }

    def get_prefix_tokens(self, item: dict[str, Any]) -> list[int]:
        prefix_tokens = []
        if self.multilingual_model:
            prefix_tokens.extend([self.en_lang_token_id, self.transcribe_token_id])
        if isinstance(self.tokenizer, WhisperAccentTokenizer):
            accent_token_id = self.tokenizer.convert_tokens_to_ids(ACCENTS[item["accent"]])
            prefix_tokens.append(accent_token_id)
        prefix_tokens.append(self.no_timestamps_token_id)
        return prefix_tokens


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

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
        batch["attention_mask"] = torch.tensor(np.array([i["attention_mask"] for i in features]))

        # Pad labels
        labels_list = [{"input_ids": feature["labels"]} for feature in features]
        labels = self.processor.tokenizer.pad(
            labels_list, return_tensors="pt", return_attention_mask=True
        )

        # Replace padding tokens in labels with -100 to ignore in loss computation
        batch["labels"] = labels["input_ids"].masked_fill(
            labels["attention_mask"].ne(1), IGNORE_INDEX
        )

        # # Remove bos token since it will be added during forward pass
        # bos_token_id = self.processor.tokenizer.bos_token_id
        # check_bos_token = batch["labels"].eq(bos_token_id)[:, 0].all().item()
        # if batch["labels"].shape[0] > 0 and check_bos_token:
        #     batch["labels"] = batch["labels"][:, 1:]
        #     batch["attention_mask"] = batch["attention_mask"][:, 1:]

        return batch
