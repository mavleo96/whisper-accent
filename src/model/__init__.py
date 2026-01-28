from .model import *
from .tokenization import *

# from transformers import AutoConfig, AutoTokenizer, AutoProcessor, AutoModel

__all__ = [
    "WhisperAccentConfig",
    "WhisperAccentModel",
    "WhisperAccentForConditionalGeneration",
    "WhisperAccentTokenizerFast",
    "WhisperAccentProcessor",
]

# # Register model and config on import only once
# _registered = False
# if not _registered:
#     AutoConfig.register("whisper_accent", config=WhisperAccentConfig)
#     AutoTokenizer.register(WhisperAccentConfig, fast_tokenizer_class=WhisperAccentTokenizerFast)
#     AutoProcessor.register(WhisperAccentConfig, processor_class=WhisperAccentProcessor)
#     AutoModel.register(WhisperAccentConfig, model_class=WhisperAccentModel)
#     _registered = True
