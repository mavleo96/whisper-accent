from .configuration import WhisperAccentConfig
from .model import WhisperAccentForConditionalGeneration, WhisperAccentModel


def register_whisper_accent():
    """Registers your custom WhisperAccent classes with Transformers Auto* classes.
    Call this once before using any Auto* with your model.
    """
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForSpeechSeq2Seq,
    )

    AutoConfig.register("whisper_accent", WhisperAccentConfig)
    AutoModel.register(WhisperAccentConfig, WhisperAccentModel)
    AutoModelForSpeechSeq2Seq.register(WhisperAccentConfig, WhisperAccentForConditionalGeneration)


__all__ = [
    "WhisperAccentConfig",
    "WhisperAccentModel",
    "WhisperAccentForConditionalGeneration",
    "register_whisper_accent",
]
