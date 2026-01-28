from .configuration import *
from .model import *
from .processing import *
from .tokenization import *


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
    AutoModelForSpeechSeq2Seq.register(
        WhisperAccentConfig, WhisperAccentForConditionalGeneration
    )
    AutoTokenizer.register(
        WhisperAccentConfig, fast_tokenizer_class=WhisperAccentTokenizer
    )
    AutoProcessor.register(WhisperAccentConfig, processor_class=WhisperAccentProcessor)


__all__ = [
    "WhisperAccentConfig",
    "WhisperAccentModel",
    "WhisperAccentForConditionalGeneration",
    "WhisperAccentTokenizer",
    "WhisperAccentProcessor",
    "register_whisper_accent",
]
