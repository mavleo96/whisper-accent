import io
import logging
import os

import gradio as gr
import numpy as np
import requests
import scipy.io.wavfile as wavfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
BACKEND_TIMEOUT = 60
MAX_SECONDS = 30


def transcribe(audio):
    if audio is None:
        return "", "—"

    sample_rate, audio_array = audio

    # Downmix to mono, convert to float32 in [-1, 1]
    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=-1)
    audio_float = audio_array.astype(np.float32)
    if np.abs(audio_float).max() > 1.0:
        audio_float /= 32768.0

    # Clip to 30 seconds
    max_samples = MAX_SECONDS * sample_rate
    if len(audio_float) > max_samples:
        logger.info("Clipping audio from %ds to %ds.", len(audio_float) // sample_rate, MAX_SECONDS)
        audio_float = audio_float[:max_samples]

    # Encode as WAV
    audio_int16 = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, audio_int16)
    buf.seek(0)

    try:
        resp = requests.post(
            f"{BACKEND_URL}/transcribe",
            files={"audio": ("audio.wav", buf, "audio/wav")},
            timeout=BACKEND_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["transcript"], data["accent"]
    except requests.exceptions.ConnectionError:
        return "", "Backend unavailable"
    except requests.exceptions.Timeout:
        return "", "Backend timeout"
    except Exception as exc:
        logger.warning("Backend call failed: %s", exc)
        return "", f"Error: {exc}"


with gr.Blocks(title="Whisper Accent") as demo:
    gr.Markdown(
        "## Whisper Accent\n"
        "Record up to 30 seconds of speech, then click **Transcribe**. "
        "Your accent is detected automatically and used to condition the decoder."
    )

    with gr.Row():
        with gr.Column(scale=3):
            audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Microphone")
        with gr.Column(scale=1):
            accent_output = gr.Textbox(label="Detected Accent", value="—", interactive=False)

    transcript_output = gr.Textbox(
        label="Transcript",
        lines=4,
        placeholder="Transcript appears here…",
        interactive=False,
    )

    with gr.Row():
        submit_btn = gr.Button("Transcribe", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")

    submit_btn.click(
        fn=transcribe,
        inputs=[audio_input],
        outputs=[transcript_output, accent_output],
    )
    clear_btn.click(
        fn=lambda: (None, "", "—"),
        inputs=[],
        outputs=[audio_input, transcript_output, accent_output],
    )

    gr.Markdown(
        f"**Backend:** `{BACKEND_URL}` · Audio is clipped to {MAX_SECONDS}s before processing."
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
