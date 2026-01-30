from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import WhisperProcessor

from src.constants import IGNORE_INDEX


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: Optional[int]

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
        #     if batch["labels"].shape[0] > 0 and (batch["labels"][:, 0] == self.decoder_start_token_id).all().cpu().item():
        #         batch["labels"] = batch["labels"][:, 1:]
        #         batch["attention_mask"] = batch["attention_mask"][:, 1:]

        return batch
