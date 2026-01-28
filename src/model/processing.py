from transformers import WhisperProcessor

from .tokenization import WhisperAccentTokenizer


class WhisperAccentProcessor(WhisperProcessor):
    @classmethod
    def _load_tokenizer_from_pretrained(
        cls, sub_processor_type, pretrained_model_name_or_path, subfolder="", **kwargs
    ):
        return WhisperAccentTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )


__all__ = ["WhisperAccentProcessor"]
