from .model import *

__all__ = ["WhisperAccentConfig", "WhisperAccentModel", "WhisperAccentForConditionalGeneration"]

# # Register model and config on import only once
# _registered = False
# if not _registered:
#     WhisperAccentConfig.register_for_auto_class()
#     WhisperAccentModel.register_for_auto_class()
#     WhisperAccentForConditionalGeneration.register_for_auto_class()
#     _registered = True
