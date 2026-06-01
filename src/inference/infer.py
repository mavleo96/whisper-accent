import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer

from ..constants import SAMPLING_RATE
from ..model import register_whisper_accent
from ..model.configuration import ACCENTS

register_whisper_accent()

ACCENT_MAP: dict[int, str] = {v: k for k, v in ACCENTS.items()}


def load_model(model_id: str, device: str, dtype: torch.dtype, **kwargs):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id, **kwargs)
    processor.tokenizer = WhisperTokenizer.from_pretrained(model_id, **kwargs)

    model.to(device).to(dtype)
    model.eval()

    if model.generation_config.is_multilingual:
        model.generation_config.language = "en"
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None

    return model, processor


def run_inference(
    audio_arrays: list[np.ndarray],
    model,
    processor,
) -> list[dict]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    inputs = processor.feature_extractor(
        audio_arrays,
        sampling_rate=SAMPLING_RATE,
        return_attention_mask=True,
        return_tensors="pt",
    )
    feats = inputs.input_features.to(device).to(dtype)
    mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        pred_ids = model.generate(feats, attention_mask=mask)
        raw_predictions = [
            t.strip() for t in processor.batch_decode(pred_ids, skip_special_tokens=True)
        ]

        accent_ids = model.predict_accent(feats, mask).tolist()
        accent_predictions = [ACCENT_MAP.get(aid, "Unknown") for aid in accent_ids]

    return [
        {
            "raw_prediction": raw,
            "prediction": processor.tokenizer.normalize(raw),
            "accent_prediction": accent,
        }
        for raw, accent in zip(raw_predictions, accent_predictions, strict=True)
    ]
