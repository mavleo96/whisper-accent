import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from whisper_accent.constants import SAMPLING_RATE
from whisper_accent.inference import load_model, run_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.WARNING)

MODEL_ID = os.environ.get("MODEL_ID", "mavleo96/whisper-accent-small.en")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

_state: dict = {"model": None, "processor": None, "ready": False, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model %s on %s (%s)…", MODEL_ID, DEVICE, DTYPE)
    try:
        model, processor = load_model(MODEL_ID, DEVICE, DTYPE)
        _state["model"] = model
        _state["processor"] = processor
        _state["ready"] = True
        logger.info("Model ready.")
    except Exception as exc:
        _state["error"] = str(exc)
        logger.exception("Model loading failed: %s", exc)
    yield
    _state["model"] = None
    _state["processor"] = None
    _state["ready"] = False


app = FastAPI(title="Whisper Accent API", version="1.0.0", lifespan=lifespan)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: str
    device: str
    error: str | None = None


class TranscribeResponse(BaseModel):
    transcript: str
    accent: str


def _load_audio(audio_bytes: bytes) -> np.ndarray:
    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
    if sr != SAMPLING_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLING_RATE)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if _state["ready"] else "loading",
        model_loaded=_state["ready"],
        model_id=MODEL_ID,
        device=DEVICE,
        error=_state.get("error"),
    )


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: Annotated[UploadFile, File()]):
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    audio_bytes = await audio.read()
    logger.info("Received audio: %d bytes, content_type=%s", len(audio_bytes), audio.content_type)

    try:
        audio_array = _load_audio(audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode audio: {exc}") from exc

    try:
        result = run_inference([audio_array], _state["model"], _state["processor"])[0]
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    logger.info(
        "Transcript: %r | Accent: %s", result["raw_prediction"], result["accent_prediction"]
    )
    return TranscribeResponse(
        transcript=result["raw_prediction"], accent=result["accent_prediction"]
    )
