from transformers import Seq2SeqTrainer


class WhisperAccentSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


__all__ = ["WhisperAccentSeq2SeqTrainer"]
