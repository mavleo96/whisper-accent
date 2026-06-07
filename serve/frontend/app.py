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

# Minimum audio length before the first live update, and how often to re-send
# while recording. Shorter = more responsive but more backend load.
LIVE_MIN_SECONDS = 3
LIVE_INTERVAL_SECONDS = 3

# NOTE on "live" transcription — this is not true streaming ASR:
# Whisper is an offline encoder-decoder that attends over a full clip, so it
# can't produce incremental output token-by-token like a streaming acoustic
# model (e.g. a streaming Conformer/RNN-T). What we do instead is re-send the
# *entire* accumulated buffer every LIVE_INTERVAL_SECONDS and overwrite the
# transcript with the fresh full-clip result. This is simple but not ideal:
# - cost and latency both grow with recording length (re-decodes from scratch
#   each time instead of only the new audio),
# - the transcript can visibly "jump"/change as longer context shifts Whisper's
#   earlier-token predictions,
# - LIVE_INTERVAL_SECONDS is a blunt latency/cost knob.
# Improvement scope: chunk with a sliding window + overlap-and-merge instead of
# resending everything, or switch to a model/runtime built for streaming (e.g.
# faster-whisper with VAD-based chunking, whisper_streaming, or a dedicated
# streaming ASR model) and push results over a websocket rather than polling.

# Why streaming=True:
# Gradio's non-streaming path converts the MediaRecorder blob via
# AudioContext.decodeAudioData() before uploading. That Web Audio decoder fails
# silently on WebM/Opus and MP4/AAC blobs on Safari/Chrome, producing all-zero
# audio. streaming=True switches to extendable-media-recorder (WAV PCM chunks
# sent directly to Python) which bypasses AudioContext entirely.


def _prepare_wav(sample_rate: int, audio_array: np.ndarray) -> io.BytesIO:
    original_dtype = audio_array.dtype
    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=-1)

    audio_float = audio_array.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        audio_float /= float(1 << (original_dtype.itemsize * 8 - 1))

    audio_float = audio_float[: MAX_SECONDS * sample_rate]
    audio_int16 = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, audio_int16)
    buf.seek(0)
    return buf


def _send_to_backend(sample_rate: int, audio_array: np.ndarray) -> tuple[str, str]:
    buf = _prepare_wav(sample_rate, audio_array)
    resp = requests.post(
        f"{BACKEND_URL}/transcribe",
        files={"audio": ("audio.wav", buf, "audio/wav")},
        timeout=BACKEND_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["transcript"], data["accent"]


def accumulate_chunk(chunk, state):
    """Append each 0.5s chunk and trigger a live update every LIVE_INTERVAL_SECONDS."""
    if chunk is None:
        return state, gr.skip(), gr.skip(), gr.skip()

    sr, arr = chunk
    if state is None:
        new_audio = arr.copy()
        last_sent = 0
    else:
        _, prev_audio, last_sent = state
        new_audio = np.concatenate([prev_audio, arr])

    new_state = (sr, new_audio, last_sent)
    duration = len(new_audio) / sr
    seconds_since_last = duration - last_sent

    if duration >= LIVE_MIN_SECONDS and seconds_since_last >= LIVE_INTERVAL_SECONDS:
        try:
            transcript, accent = _send_to_backend(sr, new_audio)
            new_state = (sr, new_audio, duration)
            return new_state, transcript, accent, "Live…"
        except Exception:
            pass

    return new_state, gr.skip(), gr.skip(), gr.skip()


def on_stop_recording(state):
    """Final transcription on stop with the full accumulated audio."""
    if state is None:
        yield "", "—", "No audio recorded", None
        return

    sr, full_audio, _ = state
    logger.info("Recording stopped: %.2fs @ %d Hz", len(full_audio) / sr, sr)
    yield "", "—", "Transcribing…", None

    try:
        transcript, accent = _send_to_backend(sr, full_audio)
        yield transcript, accent, "Done", None
    except requests.exceptions.ConnectionError:
        yield "", "—", "Error: backend unavailable", None
    except requests.exceptions.Timeout:
        yield "", "—", f"Error: backend timed out after {BACKEND_TIMEOUT}s", None
    except Exception as exc:
        logger.warning("Transcription failed: %s", exc)
        yield "", "—", f"Error: {exc}", None


with gr.Blocks(title="Whisper Accent") as demo:
    gr.Markdown("## Whisper Accent\nRecord your speech — transcript updates live as you speak.")

    # State: (sample_rate, full_audio_array, last_sent_duration_seconds)
    audio_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=3):
            audio_input = gr.Audio(
                sources=["microphone"],
                streaming=True,
                type="numpy",
                label="Microphone",
            )
        with gr.Column(scale=1):
            accent_output = gr.Textbox(label="Detected Accent", value="—", interactive=False)

    transcript_output = gr.Textbox(
        label="Transcript",
        lines=4,
        placeholder="Transcript appears here…",
        interactive=False,
    )
    status_output = gr.Textbox(label="Status", value="", interactive=False)
    clear_btn = gr.Button("Clear", variant="secondary")

    _live_outputs = [audio_state, transcript_output, accent_output, status_output]
    _stop_outputs = [transcript_output, accent_output, status_output, audio_state]

    audio_input.stream(
        fn=accumulate_chunk, inputs=[audio_input, audio_state], outputs=_live_outputs
    )
    audio_input.stop_recording(fn=on_stop_recording, inputs=[audio_state], outputs=_stop_outputs)
    clear_btn.click(fn=lambda: ("", "—", "", None), inputs=[], outputs=_stop_outputs)

    gr.Markdown(
        f"**Backend:** `{BACKEND_URL}` · Audio clipped to {MAX_SECONDS}s · "
        f"Live update every {LIVE_INTERVAL_SECONDS}s"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
