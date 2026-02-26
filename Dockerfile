FROM python:3.11-slim

# ffmpeg for audio decoding/merging
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vad_service.py .
COPY silero_vad.onnx .

# Directories created in case volumes aren't mounted yet
RUN mkdir -p /app/recordings

CMD ["python", "-u", "vad_service.py"]
