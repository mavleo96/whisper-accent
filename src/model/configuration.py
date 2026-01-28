from transformers import WhisperConfig


class WhisperAccentConfig(WhisperConfig):
    model_type = "whisper_accent"


__all__ = ["WhisperAccentConfig"]
