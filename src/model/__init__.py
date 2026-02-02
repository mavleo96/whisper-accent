from .configuration import WhisperAccentConfig
from .model import WhisperAccentForConditionalGeneration, WhisperAccentModel
from .processing import WhisperAccentProcessor
from .tokenization import WhisperAccentTokenizer


def register_whisper_accent():
    """Registers your custom WhisperAccent classes with Transformers Auto* classes.
    Call this once before using any Auto* with your model.
    """
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        AutoTokenizer,
    )

    AutoConfig.register("whisper_accent", WhisperAccentConfig)
    AutoModel.register(WhisperAccentConfig, WhisperAccentModel)
    AutoModelForSpeechSeq2Seq.register(WhisperAccentConfig, WhisperAccentForConditionalGeneration)
    AutoTokenizer.register(WhisperAccentConfig, WhisperAccentTokenizer)
    AutoProcessor.register(WhisperAccentConfig, WhisperAccentProcessor)


__all__ = [
    "WhisperAccentConfig",
    "WhisperAccentModel",
    "WhisperAccentForConditionalGeneration",
    "WhisperAccentTokenizer",
    "WhisperAccentProcessor",
    "register_whisper_accent",
]
