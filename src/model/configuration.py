from transformers import WhisperConfig

ACCENTS = {
    "english": 0,
    "american": 1,
    "scottish": 2,
    "irish": 3,
    "canadian": 4,
    "northernirish": 5,
    "indian": 6,
    "spanish": 7,
    "dutch": 8,
    "german": 9,
    "czech": 10,
    "polish": 11,
    "french": 12,
    "italian": 13,
    "hungarian": 14,
    "finnish": 15,
    "vietnamese": 16,
    "romanian": 17,
    "slovak": 18,
    "estonian": 19,
    "lithuanian": 20,
    "croatian": 21,
    "slovene": 22,
}


class WhisperAccentConfig(WhisperConfig):
    model_type = "whisper_accent"

    def __init__(self, **kwargs):
        self.base_model = kwargs.pop("base_model", None)

        self.num_accents = kwargs.pop("num_accents", len(ACCENTS))
        self.accent_to_id = kwargs.pop("accent_to_id", ACCENTS)

        # Decoder conditioning dimension; default half of d_model.
        self.accent_embed_dim = kwargs.pop("accent_embed_dim", kwargs.get("d_model", 384) // 2)

        self.accent_proj_size = kwargs.pop("accent_proj_size", 256)
        self.accent_attention_heads = kwargs.pop("accent_attention_heads", 8)
        # Optional class weights for imbalanced accent classification loss.
        self.accent_class_weights = kwargs.pop("accent_class_weights", None)

        super().__init__(**kwargs)


__all__ = ["WhisperAccentConfig"]
