# Whisper Accent — Serving API

## Running

```bash
cd serve
docker-compose up --build
```

Override the model (defaults to `mavleo96/whisper-accent-small.en`):

```bash
MODEL_ID=mavleo96/whisper-accent-medium.en docker-compose up --build
```

The frontend will not start until the backend healthcheck passes (allow ~2 minutes for model download on first run).

The backend image is built on `nvidia/cuda` and requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host. On CPU-only machines, swap the base image in `backend/Dockerfile` to `python:3.12-slim` and remove the `deploy` block from `docker-compose.yml`.

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
  "model_id": "mavleo96/whisper-accent-small.en",
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

Audio longer than 30 seconds is handled by the frontend before submission. If hitting the API directly, clip your audio to 30 seconds beforehand.

---

### `GET /docs`

Interactive Swagger UI — send requests directly from the browser.

### `GET /redoc`

Read-only ReDoc UI — cleaner layout, good for sharing.

---

## Services

| Service  | Port | Description                        |
|----------|------|------------------------------------|
| backend  | 8000 | FastAPI inference server (internal)|
| frontend | 7860 | Gradio UI (public)                 |

The backend port is mapped to the host for development. In production, remove the `ports` entry for `backend` in `docker-compose.yml` to keep it internal only.
