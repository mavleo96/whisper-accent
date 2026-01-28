from transformers import WhisperProcessor, WhisperTokenizerFast


class WhisperAccentTokenizerFast(WhisperTokenizerFast):
    def __init__(self, *args, **kwargs):
        self.accent = kwargs.pop("accent", None)
        super().__init__(*args, **kwargs)

    def set_prefix_tokens(
        self, language=None, task=None, accent=None, predict_timestamps=None
    ):
        self.accent = accent if accent is not None else self.accent
        super().set_prefix_tokens(language, task, predict_timestamps)

    @property
    def prefix_tokens(self):
        bos_sequence = super().prefix_tokens
        # if accent is not set, return original prefix tokens
        if self.accent is None:
            return bos_sequence

        # If accent is set, insert accent token id
        accent_token_id = self.convert_tokens_to_ids(self.accent)
        if accent_token_id == self.eos_token_id:
            # TODO: Add supported accents
            raise ValueError(
                f"Unsupported accent: {self.accent}. Accent should be one of: {['<|accent0|>', '<|accent1|>']}."
            )

        # if predict_timestamps is not set, insert accent token id before timestamps token
        if not self.predict_timestamps:
            bos_sequence.insert(-1, accent_token_id)
        else:
            bos_sequence.append(accent_token_id)
        return bos_sequence


class WhisperAccentProcessor(WhisperProcessor):
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "WhisperAccentTokenizerFast"


__all__ = ["WhisperAccentTokenizerFast", "WhisperAccentProcessor"]
