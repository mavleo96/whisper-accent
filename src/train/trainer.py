from transformers import EvalPrediction, Seq2SeqTrainer

from src.model import WhisperAccentProcessor
from src.utils.metrics import compute_wer


def compute_metrics(
    eval_pred: EvalPrediction, processor: WhisperAccentProcessor
) -> dict[str, float]:
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    raw_pred_str = processor.tokenizer.batch_decode(
        predictions, skip_special_tokens=True
    )
    pred_str = [processor.tokenizer.normalize(s) for s in raw_pred_str]

    wer, _ = compute_wer(pred_str, label_str)
    return {"wer": wer}


class WhisperAccentSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


__all__ = ["WhisperAccentSeq2SeqTrainer"]
