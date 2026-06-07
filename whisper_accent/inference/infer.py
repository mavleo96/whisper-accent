import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer

from ..constants import SAMPLING_RATE
from ..model import register_whisper_accent

register_whisper_accent()


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

    accent_map: dict[int, str] = {v: k for k, v in model.config.accent_to_id.items()}

    with torch.no_grad():
        encoder_outputs = model.model.encoder(feats, attention_mask=mask, return_dict=True)
        accent_ids = encoder_outputs.accent_logits.argmax(dim=-1).tolist()
        accent_predictions = [accent_map.get(aid, "Unknown") for aid in accent_ids]

        pred_ids = model.generate(feats, attention_mask=mask, encoder_outputs=encoder_outputs)
        raw_predictions = [
            t.strip() for t in processor.batch_decode(pred_ids, skip_special_tokens=True)
        ]

    return [
        {
            "raw_prediction": raw,
            "prediction": processor.tokenizer.normalize(raw),
            "accent_prediction": accent,
        }
        for raw, accent in zip(raw_predictions, accent_predictions, strict=True)
    ]
