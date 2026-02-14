from transformers import WhisperConfig

from ..constants import ACCENTS


class WhisperAccentConfig(WhisperConfig):
    model_type = "whisper_accent"

    def __init__(self, **kwargs):
        self.accent_embed_dim = kwargs.pop("accent_embed_dim", kwargs.get("d_model", 384) // 2)
        self.num_accents = kwargs.pop("num_accents", len(ACCENTS))
        super().__init__(**kwargs)


__all__ = ["WhisperAccentConfig"]
