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


def _prepare_wav(audio) -> io.BytesIO:
    sample_rate, audio_array = audio

    # Downmix to mono, convert to float32 in [-1, 1]
    original_dtype = audio_array.dtype
    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=-1)
    audio_float = audio_array.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        audio_float /= float(1 << (original_dtype.itemsize * 8 - 1))

    # Clip to 30 seconds
    max_samples = MAX_SECONDS * sample_rate
    if len(audio_float) > max_samples:
        logger.info("Clipping audio from %ds to %ds.", len(audio_float) // sample_rate, MAX_SECONDS)
        audio_float = audio_float[:max_samples]

    # Encode as int16 WAV — standard format, torchaudio handles it on all backends
    audio_int16 = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, audio_int16)
    buf.seek(0)
    return buf


def transcribe(audio):
    if audio is None:
        yield "", "—", "No audio recorded.", gr.skip()
        return

    # Copy before the first yield: Gradio may recycle the widget's numpy buffer
    # when the generator suspends, which would flatten subsequent playback.
    audio = (audio[0], audio[1].copy())

    yield "", "—", "Preparing audio…", gr.skip()

    try:
        buf = _prepare_wav(audio)
    except Exception as exc:
        yield "", "—", f"Audio error: {exc}", audio
        return

    yield "", "—", "Sending to backend…", gr.skip()

    try:
        resp = requests.post(
            f"{BACKEND_URL}/transcribe",
            files={"audio": ("audio.wav", buf, "audio/wav")},  # sample rate encoded in WAV header
            timeout=BACKEND_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        yield data["transcript"], data["accent"], "Done.", audio
    except requests.exceptions.ConnectionError:
        yield "", "—", "Error: backend unavailable.", audio
    except requests.exceptions.Timeout:
        yield "", "—", f"Error: backend timed out after {BACKEND_TIMEOUT}s.", audio
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        yield "", "—", f"Error: {detail}", audio
    except Exception as exc:
        logger.warning("Backend call failed: %s", exc)
        yield "", "—", f"Error: {exc}", audio


with gr.Blocks(title="Whisper Accent") as demo:
    gr.Markdown(
        "## Whisper Accent\n"
        "Record your speech, then click **Transcribe**. "
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

    status_output = gr.Textbox(label="Status", value="", interactive=False)

    with gr.Row():
        submit_btn = gr.Button("Transcribe", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")

    submit_btn.click(
        fn=transcribe,
        inputs=[audio_input],
        outputs=[transcript_output, accent_output, status_output, audio_input],
    )
    clear_btn.click(
        fn=lambda: (None, "", "—", ""),
        inputs=[],
        outputs=[audio_input, transcript_output, accent_output, status_output],
    )

    gr.Markdown(
        f"**Backend:** `{BACKEND_URL}` · Audio is clipped to {MAX_SECONDS}s before processing."
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
