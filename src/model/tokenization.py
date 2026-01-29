from transformers import WhisperTokenizer

ACCENTS = {
    "english": "<|a_english|>",
    "american": "<|a_american|>",
    "scottish": "<|a_scottish|>",
    "irish": "<|a_irish|>",
    "canadian": "<|a_canadian|>",
    "northernirish": "<|a_northernirish|>",
    "indian": "<|a_indian|>",
    "spanish": "<|a_spanish|>",
    "dutch": "<|a_dutch|>",
    "german": "<|a_german|>",
    "czech": "<|a_czech|>",
    "polish": "<|a_polish|>",
    "french": "<|a_french|>",
    "italian": "<|a_italian|>",
    "hungarian": "<|a_hungarian|>",
    "finnish": "<|a_finnish|>",
    "vietnamese": "<|a_vietnamese|>",
    "romanian": "<|a_romanian|>",
    "slovak": "<|a_slovak|>",
    "estonian": "<|a_estonian|>",
    "lithuanian": "<|a_lithuanian|>",
    "croatian": "<|a_croatian|>",
    "slovene": "<|a_slovene|>",
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
        if self.accent not in ACCENTS:
            raise ValueError(
                f"Unsupported accent: {self.accent}. Accent should be one of: {list(ACCENTS.keys())}."
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
