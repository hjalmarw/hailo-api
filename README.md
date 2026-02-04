# Hailo API

FastAPI-based object detection and speech-to-text API for Raspberry Pi 5 with Hailo-10H AI accelerator.

## Features

### Object Detection (`POST /api/v1/detect`)
- Real-time object detection using YOLO models on the Hailo-10H NPU
- Supports multiple YOLO variants: YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11
- Hot-swappable models via the `model` parameter
- Tiling support for large/aerial images
- Oriented bounding box (OBB) detection for aerial imagery (DOTA labels)
- Image preprocessing options: CLAHE, histogram equalization
- Zoom-aware confidence adjustment for satellite imagery

### Speech-to-Text (`POST /api/v1/transcribe`)
- Whisper speech-to-text transcription
- Accepts any audio format (WAV, OGG, MP3, etc.) as base64-encoded input
- Supports variants: `tiny`, `tiny.en`, `base`, `base.en`
- Runs on CPU via HuggingFace transformers (~2s for short clips on Pi 5)

### Other Endpoints
- `GET /api/v1/health` — Health check with device and model status
- `GET /api/v1/models` — List all available detection and whisper models
- `GET /api/v1/labels` — Get detection labels (COCO or DOTA)

## Requirements

- Raspberry Pi 5 (or compatible ARM64 system)
- Hailo-10H AI accelerator module
- HailoRT 5.1.1+
- Python 3.11+
- ffmpeg (for audio format conversion)
- YOLO HEF model files in the working directory (not included — download from Hailo Model Zoo)

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

HEF model files must be placed in the working directory. Download from the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo).

## Usage

### Run directly
```bash
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8001
```

### Run as systemd service
```ini
[Unit]
Description=Hailo Detection API
After=network.target

[Service]
Type=simple
User=grazzy
WorkingDirectory=/path/to/hailo-api
ExecStart=/path/to/hailo-api/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001
Restart=always

[Install]
WantedBy=multi-user.target
```

### Example: Detect objects in an image
```bash
IMAGE=$(base64 -w0 photo.jpg)
curl -X POST http://localhost:8001/api/v1/detect \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE\", \"min_confidence\": 0.4}"
```

### Example: Transcribe audio
```bash
AUDIO=$(base64 -w0 recording.ogg)
curl -X POST http://localhost:8001/api/v1/transcribe \
  -H "Content-Type: application/json" \
  -d "{\"audio\": \"$AUDIO\"}"
```

## Architecture

- **`main.py`** — FastAPI application with all endpoints
- **`detector.py`** — Hailo-10H object detection with model caching, tiling, OBB support
- **`transcriber.py`** — Whisper speech-to-text via HuggingFace transformers (CPU)
- **`coco_labels.py`** — COCO-80 label definitions

## Limitations

- **Whisper runs on CPU, not NPU.** The pre-compiled Whisper decoder HEF files from the Hailo S3 bucket are built for HailoRT 5.2.0 and produce incorrect output on HailoRT 5.1.1. Transcription uses CPU-based inference as a workaround (~2s per short clip). Once HailoRT 5.2.0 is available, NPU-accelerated whisper can be revisited.
- **Detection models not included.** YOLO HEF files must be downloaded separately from the Hailo Model Zoo.
- **Single-device concurrency.** The Hailo-10H is shared between detection models using `group_id="SHARED"` with round-robin scheduling. Concurrent heavy workloads may queue.
- **ARM64 only.** Designed for Raspberry Pi 5 with Hailo-10H. The `hailo_platform` Python bindings are platform-specific.

## License

MIT
