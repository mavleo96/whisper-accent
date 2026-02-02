import json
import os

from transformers import WhisperProcessor

from .tokenization import WhisperAccentTokenizer


class WhisperAccentProcessor(WhisperProcessor):
    @classmethod
    def _load_tokenizer_from_pretrained(
        cls, sub_processor_type, pretrained_model_name_or_path, subfolder="", **kwargs
    ):
        return WhisperAccentTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        super().save_pretrained(save_directory, push_to_hub, **kwargs)
        if self.tokenizer.english_spelling_normalizer is not None:
            with open(os.path.join(save_directory, "normalizer.json"), "w") as f:
                json.dump(self.tokenizer.english_spelling_normalizer, f)


__all__ = ["WhisperAccentProcessor"]
