FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf python3.12 /usr/bin/python && ln -sf pip3 /usr/bin/pip

WORKDIR /app

COPY serve/backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install whisper_accent as a package
COPY whisper_accent/ whisper_accent/
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY serve/backend/main.py .

ENV MODEL_ID="mavleo96/whisper-accent-medium.en"

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
