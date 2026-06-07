# Whisper Accent — Serving API

## Running

```bash
cd serve
docker compose up --build
```

Override the model (defaults to `mavleo96/whisper-accent-medium.en`):

```bash
MODEL_ID=mavleo96/whisper-accent-medium.en docker compose up --build
```

Override the host ports (defaults: backend `8000`, frontend `7860`) — useful if those ports are already taken on the host:

```bash
BACKEND_PORT=8001 FRONTEND_PORT=7861 docker compose up --build
```

These only remap the host side; the services still talk to each other over the internal Docker network on their default ports (`backend:8000`).

The frontend will not start until the backend healthcheck passes (allow ~2 minutes for model download on first run).

The backend image is built on `nvidia/cuda` and requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host. On CPU-only machines, swap the base image in `backend/Dockerfile` to `python:3.12-slim` and remove the `deploy` block from `compose.yaml`.

---

## Microphone Access

Browsers require a **secure context** for microphone access. The frontend launches with Gradio's `share=True`, which prints a public `https://*.gradio.live` URL in the container logs — use that link for microphone access from any machine. Alternatively:

- Access via `http://localhost:7860` (use SSH port-forwarding: `ssh -L 7860:localhost:7860 <host>`)
- Or serve over HTTPS

Plain `http://<remote-ip>:7860` will not work — the browser will silently block the microphone.

---

## Endpoints

### `GET /health`

Returns model load status.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_id": "mavleo96/whisper-accent-medium.en",
  "device": "cpu",
  "error": null
}
```

---

### `POST /transcribe`

Accepts an audio file (WAV, MP3, FLAC, …) and returns a transcript and detected accent.

```bash
curl -X POST http://localhost:8000/transcribe \
     -F "audio=@your_clip.wav"
```

```json
{
  "transcript": "Hello, how are you doing today?",
  "accent": "Indian"
}
```

---

### `GET /docs`

Interactive Swagger UI — send requests directly from the browser.

### `GET /redoc`

Read-only ReDoc UI — cleaner layout, good for sharing.

---

## Known Limitations: "Live" Transcription

The frontend updates the transcript every few seconds while you're still recording, but this is **not true streaming ASR**. Whisper is an offline encoder-decoder model that attends over a full clip — it can't emit incremental tokens like a streaming acoustic model can. Instead, the frontend periodically re-sends the *entire* accumulated recording and overwrites the transcript with a fresh full-clip result (see `LIVE_INTERVAL_SECONDS` / `accumulate_chunk` in `frontend/app.py`).

Tradeoffs of this approach:

- Cost and latency grow with recording length (each update re-decodes from scratch rather than just the new audio).
- The displayed transcript can visibly change/"jump" as added context shifts Whisper's earlier predictions.

**Improvement scope:** chunk with a sliding window and overlap-and-merge instead of resending the whole buffer, or move to a model/runtime designed for streaming (e.g. `faster-whisper` with VAD-based chunking, `whisper_streaming`, or a dedicated streaming ASR architecture) and push updates over a websocket instead of polling.

---

## Services

| Service  | Port | Description                              |
|----------|------|------------------------------------------|
| backend  | 8000 | FastAPI inference server (internal)      |
| frontend | 7860 | Gradio UI (public — also exposed via a `*.gradio.live` share link) |

The backend port is mapped to the host for development. In production, remove the `ports` entry for `backend` in `compose.yaml` to keep it internal only.
