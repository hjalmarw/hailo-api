# Piper TTS on Raspberry Pi - Complete Tutorial

Fast, offline text-to-speech using neural voice synthesis. Generates natural-sounding speech in ~50ms on Pi 5.

## Requirements

- Raspberry Pi 4/5 (64-bit OS)
- Python 3.9+
- Audio output (speaker/headphones)
- ~500MB disk space (piper + voice model)

## Installation

### 1. System Dependencies

```bash
sudo apt update
sudo apt install -y python3-venv libasound2-dev
```

### 2. Create Virtual Environment

```bash
cd /home/grazzy/projects
mkdir piper-tts && cd piper-tts
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Piper

```bash
pip install piper-tts
```

### 4. Download Voice Model

Piper needs two files per voice: `.onnx` (model) and `.onnx.json` (config).

```bash
mkdir -p models

# English - Amy (medium quality, good balance of speed/quality)
wget -O models/en_US-amy-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx"

wget -O models/en_US-amy-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json"
```

**Other popular voices:**

| Voice | Language | Size | Download |
|-------|----------|------|----------|
| amy-medium | en_US | 63MB | Above |
| lessac-medium | en_US | 63MB | `en_US-lessac-medium` |
| jenny-medium | en_GB | 63MB | `en_GB-jenny_dioco-medium` |
| thorsten-medium | de_DE | 63MB | `de_DE-thorsten-medium` |

Browse all voices: https://huggingface.co/rhasspy/piper-voices

## Basic Usage

### Command Line

```bash
# Generate WAV file
echo "Hello from Raspberry Pi!" | piper -m models/en_US-amy-medium.onnx --output_file hello.wav

# Play it
aplay hello.wav

# Stream directly to speakers (faster)
echo "This plays immediately" | piper -m models/en_US-amy-medium.onnx --output-raw | \
  aplay -r 22050 -f S16_LE -t raw -
```

### Python API

```python
#!/usr/bin/env python3
"""Basic Piper TTS usage."""

import wave
import subprocess
from pathlib import Path

MODEL_PATH = Path("models/en_US-amy-medium.onnx")

def text_to_speech(text: str, output_path: str = "output.wav"):
    """Convert text to speech and save as WAV file."""
    process = subprocess.Popen(
        ["piper", "-m", str(MODEL_PATH), "--output_file", output_path],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    process.communicate(input=text.encode())
    return output_path

def text_to_speech_bytes(text: str) -> bytes:
    """Convert text to speech and return raw audio bytes."""
    process = subprocess.Popen(
        ["piper", "-m", str(MODEL_PATH), "--output-raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    audio_data, _ = process.communicate(input=text.encode())
    return audio_data

def play_text(text: str):
    """Speak text directly through speakers."""
    piper = subprocess.Popen(
        ["piper", "-m", str(MODEL_PATH), "--output-raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    aplay = subprocess.Popen(
        ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
        stdin=piper.stdout,
        stderr=subprocess.PIPE
    )
    piper.stdin.write(text.encode())
    piper.stdin.close()
    aplay.wait()

if __name__ == "__main__":
    # Test it
    play_text("Piper text to speech is working!")
```

## FastAPI Integration

Create a TTS API that integrates with your existing Hailo detection system.

### tts_server.py

```python
#!/usr/bin/env python3
"""Piper TTS REST API server."""

import io
import time
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Piper TTS API", version="1.0.0")

# Configuration
MODELS_DIR = Path("/home/grazzy/projects/piper-tts/models")
DEFAULT_MODEL = "en_US-amy-medium"

# Available voices
VOICES = {
    "amy": MODELS_DIR / "en_US-amy-medium.onnx",
    "lessac": MODELS_DIR / "en_US-lessac-medium.onnx",
}

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "amy"

class TTSResponse(BaseModel):
    success: bool
    processing_ms: float
    voice: str
    text_length: int

@app.get("/api/v1/voices")
def list_voices():
    """List available voices."""
    available = {name: path.exists() for name, path in VOICES.items()}
    return {"voices": available, "default": DEFAULT_MODEL}

@app.post("/api/v1/tts")
def synthesize(request: TTSRequest):
    """Synthesize speech and return WAV audio."""
    # Validate voice
    voice_path = VOICES.get(request.voice)
    if not voice_path or not voice_path.exists():
        raise HTTPException(404, f"Voice '{request.voice}' not found")

    if not request.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    if len(request.text) > 5000:
        raise HTTPException(400, "Text too long (max 5000 chars)")

    start = time.perf_counter()

    # Run piper
    process = subprocess.Popen(
        ["piper", "-m", str(voice_path), "--output-raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    audio_data, stderr = process.communicate(input=request.text.encode())

    if process.returncode != 0:
        raise HTTPException(500, f"TTS failed: {stderr.decode()}")

    processing_ms = (time.perf_counter() - start) * 1000

    # Convert raw PCM to WAV
    wav_buffer = io.BytesIO()
    import wave
    with wave.open(wav_buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(22050)
        wav.writeframes(audio_data)

    wav_buffer.seek(0)

    return StreamingResponse(
        wav_buffer,
        media_type="audio/wav",
        headers={
            "X-Processing-Ms": str(round(processing_ms, 1)),
            "X-Voice": request.voice,
            "Content-Disposition": "attachment; filename=speech.wav"
        }
    )

@app.post("/api/v1/tts/stream")
def synthesize_stream(request: TTSRequest):
    """Synthesize speech and stream raw PCM audio."""
    voice_path = VOICES.get(request.voice)
    if not voice_path or not voice_path.exists():
        raise HTTPException(404, f"Voice '{request.voice}' not found")

    def generate():
        process = subprocess.Popen(
            ["piper", "-m", str(voice_path), "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        process.stdin.write(request.text.encode())
        process.stdin.close()

        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="audio/raw",
        headers={
            "X-Sample-Rate": "22050",
            "X-Sample-Format": "S16_LE",
            "X-Channels": "1"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
```

### Run the Server

```bash
# Install additional dependency
pip install fastapi uvicorn

# Run
python tts_server.py
```

### Test the API

```bash
# List voices
curl http://localhost:8002/api/v1/voices

# Generate speech (saves to file)
curl -X POST http://localhost:8002/api/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from the TTS API!", "voice": "amy"}' \
  --output speech.wav

# Play it
aplay speech.wav
```

## Systemd Service

Create `/etc/systemd/system/piper-tts.service`:

```ini
[Unit]
Description=Piper TTS API
After=network.target

[Service]
Type=simple
User=grazzy
WorkingDirectory=/home/grazzy/projects/piper-tts
ExecStart=/home/grazzy/projects/piper-tts/venv/bin/python tts_server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable piper-tts
sudo systemctl start piper-tts
```

## Performance Tuning

### Voice Quality Levels

| Quality | File Size | Speed | Use Case |
|---------|-----------|-------|----------|
| low | ~15MB | Fastest | Real-time responses |
| medium | ~63MB | Fast | Good balance (recommended) |
| high | ~100MB+ | Slower | Best quality |

### Adjust Speech Rate

Edit the `.onnx.json` config file:

```json
{
  "inference": {
    "noise_scale": 0.667,
    "length_scale": 1.0,
    "noise_w": 0.8
  }
}
```

- `length_scale`: 0.8 = faster, 1.2 = slower
- `noise_scale`: Higher = more variation

### Preload Model (Reduces First-Request Latency)

```python
# Keep piper process running with --json-input
import subprocess
import json

class PiperPool:
    def __init__(self, model_path: str):
        self.process = subprocess.Popen(
            ["piper", "-m", model_path, "--json-input", "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    def synthesize(self, text: str) -> bytes:
        request = json.dumps({"text": text}) + "\n"
        self.process.stdin.write(request.encode())
        self.process.stdin.flush()
        # Read audio output...
```

## Integration with Hailo Detection

Example: Announce detections via TTS

```python
import requests

def announce_detection(detections: list):
    """Speak detection results."""
    if not detections:
        return

    # Build announcement
    counts = {}
    for d in detections:
        label = d['label']
        counts[label] = counts.get(label, 0) + 1

    parts = [f"{count} {label}{'s' if count > 1 else ''}"
             for label, count in counts.items()]
    text = f"Detected: {', '.join(parts)}"

    # Send to TTS API
    response = requests.post(
        "http://localhost:8002/api/v1/tts",
        json={"text": text, "voice": "amy"}
    )

    if response.ok:
        # Play the audio
        import subprocess
        process = subprocess.Popen(
            ["aplay", "-"],
            stdin=subprocess.PIPE
        )
        process.communicate(input=response.content)
```

## Troubleshooting

### No audio output

```bash
# Check audio devices
aplay -l

# Test speakers
speaker-test -t wav -c 2

# Set default output (if using HDMI)
sudo raspi-config  # Advanced Options > Audio
```

### Model download fails

```bash
# Use curl instead of wget
curl -L -o models/en_US-amy-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx"
```

### Slow first synthesis

The first request loads the model into memory (~1-2 seconds). Subsequent requests are fast (~50ms). Use the preload pattern above to avoid this.

## Resources

- [Piper GitHub](https://github.com/rhasspy/piper)
- [Voice Models](https://huggingface.co/rhasspy/piper-voices)
- [Voice Samples](https://rhasspy.github.io/piper-samples/)
- [PyPI Package](https://pypi.org/project/piper-tts/)
