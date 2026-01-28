from transformers import WhisperProcessor, WhisperTokenizer

ACCENTS = {
    "<|a_us|>": "mainstream us",
    "<|a_southern_british|>": "southern british",
    "<|a_irish|>": "irish",
    "<|a_eastern_european|>": "eastern european",
    "<|a_italian|>": "italian",
    "<|a_egyptian|>": "egyptian",
    "<|a_vietnamese|>": "vietnamese",
    "<|a_chinese|>": "chinese",
    "<|a_indian|>": "indian",
    "<|a_indonesian|>": "indonesian",
    "<|a_unknown|>": "unknown",
}


class WhisperAccentTokenizer(WhisperTokenizer):
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
        self.accent = self.accent.lower()
        if self.accent not in ACCENTS.values():
            raise ValueError(
                f"Unsupported accent: {self.accent}. Accent should be one of: {list(ACCENTS.values())}."
            )
        accent_token_id = self.convert_tokens_to_ids(ACCENTS[self.accent])
        if accent_token_id == self.eos_token_id:
            raise ValueError(
                f"Accent token {ACCENTS[self.accent]} was not found in the tokenizer."
            )

        # if predict_timestamps is not set, insert accent token id before timestamps token
        if not self.predict_timestamps:
            bos_sequence.insert(-1, accent_token_id)
        else:
            bos_sequence.append(accent_token_id)
        return bos_sequence


__all__ = ["WhisperAccentTokenizer"]
