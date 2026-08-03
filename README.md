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

### Face Recognition (`/api/v1/faces/*`)
- Identify known people in images or video, on the NPU
- Two-stage pipeline: SCRFD-10G face detection with 5 landmarks → similarity-transform
  alignment to a canonical 112×112 crop → ArcFace/MobileFaceNet 512-d embedding
- Enrolled people are stored as embeddings and matched by cosine similarity
- Quality gates reject faces that are too small or too motion-blurred to judge, and say
  which gate they failed rather than silently dropping them
- Video mode aggregates sightings per person — who appeared, how often, and when

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/faces/enroll` | Add a person from a photo containing exactly one face |
| `POST /api/v1/faces/identify` | Find and name every face in an image |
| `POST /api/v1/faces/identify_video` | Identify people across a video |
| `GET /api/v1/faces/persons` | List enrolled people |
| `DELETE /api/v1/faces/persons/{name}` | Remove a person and all their embeddings |

#### Example: enroll and identify
```bash
# Enroll — call repeatedly with different photos of the same person.
# Varied lighting and angle measurably improve recognition on camera footage.
IMAGE=$(base64 -w0 alice.jpg)
curl -X POST http://localhost:8001/api/v1/faces/enroll \
  -H "Content-Type: application/json" \
  -d "{\"name\": \"Alice\", \"image\": \"$IMAGE\"}"

# Identify
IMAGE=$(base64 -w0 doorbell.jpg)
curl -X POST http://localhost:8001/api/v1/faces/identify \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE\"}"
```

#### Video: which frames get analysed

Long footage is mostly nothing happening. Analysing every Nth frame wastes the NPU on
an empty driveway and can still miss the one clear frame of someone's face. The default
`adaptive` strategy addresses both:

1. **Motion gate** — frames whose downscaled grey level barely differs from the previous
   one are dropped before any inference. This is where most of the saving comes from.
2. **Detect on survivors** — SCRFD runs only on frames that passed the gate.
3. **Track and score** — detections are linked into tracks (box overlap, falling back to
   centre proximity so a head turn doesn't split one person into three). Each detection
   is scored on face size, sharpness, detection confidence and **frontality**, estimated
   from where the nose sits between the eyes.
4. **Embed only the winners** — ArcFace runs on the best few crops per track, and the
   results vote. One unlucky frame cannot decide who someone is.

Measured on 90s of synthetic driveway footage (faces present in 12% of frames):

| | uniform | adaptive |
|---|---|---|
| Frames examined | 450 | 450 |
| Skipped by motion gate | 0 | 392 |
| Reached face detection | 450 | 58 |
| **ArcFace runs** | **30** | **3** |
| **Wall clock** | **17.9s** | **4.5s** |
| People found | identical | identical |

Cost tracks *how much happens* in the video rather than how long it is. On the dense
demo montage — constant motion, nothing for the gate to skip — adaptive still wins on
quality, cutting 39 fragmented tracks to 24 correct ones.

Set `strategy: "uniform"` for a plain fixed-stride scan with per-frame counts.

```bash
curl -X POST http://localhost:8001/api/v1/faces/identify_video \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/mnt/rai-models/footage/front_door.mp4",
       "sample_every": 3, "include_crops": true}'
```

Each track reports the appearance and its best frame:
```json
{"track_id": 0, "name": "Anna", "votes": "3/3", "detections": 30,
 "first_seen_s": 20.0, "last_seen_s": 25.8, "best_frame_s": 23.2,
 "frontality": 0.987, "face_pixels": 18042, "best_crop_jpeg_b64": "..."}
```

##### Video parameters

| Parameter | Default | Effect |
|---|---|---|
| `strategy` | `adaptive` | `adaptive` selects frames by motion and face quality; `uniform` uses a fixed stride |
| `sample_every` | 5 | Examine every Nth frame. With `adaptive` this can be small — the motion gate does the real filtering. |
| `motion_threshold` | 2.0 | Mean pixel change needed to analyse a frame. Raise for noisy sensors, set `0` to disable. |
| `embeddings_per_track` | 3 | Best frames per person that get embedded and voted on |
| `track_iou` | 0.3 | Box overlap treated as the same person |
| `track_max_gap_s` | 2.0 | Absence before a person counts as a new appearance |
| `min_track_quality` | 0.02 | Appearances below this are reported but not embedded — a one-frame profile glance can't be identified anyway |
| `include_crops` | false | Return the best face crop per track as base64 JPEG |

##### Reading the diagnostics

`frames_skipped_no_motion`, `faces_rejected_small`, `faces_rejected_blurry` and
`unidentifiable_tracks` exist so an empty result is never ambiguous. If nobody was
found, these say whether the scene was empty, the faces were too far away, or the
footage was too blurry.

> **`min_face_pixels` is the parameter most likely to need tuning.** The 12000 default
> (~110×110) suits doorbell-distance faces. A camera covering a driveway will produce
> faces well below it — in testing, a face at 10598px was silently correct to reject by
> the default and identified at 0.947 once the gate was lowered. **Check
> `faces_rejected_small` before concluding nobody was there.**

#### Tuning

| Parameter | Default | Effect |
|---|---|---|
| `match_threshold` | 0.5 | Cosine similarity needed to claim a match. Measured separation on validation data was ~0.89 (same-person ≥0.93, different-person ≤0.04), so 0.5 sits in a wide empty band. Raise toward 0.7 to be stricter. |
| `min_face_pixels` | 12000 | ≈110×110. Faces smaller than this are reported as `face_too_small`. |
| `blur_tolerance` | 150 | Variance of Laplacian. Motion-blurred faces are reported as `too_blurry`. |

#### Notes

- **Enrollment refuses images containing more than one face.** Silently picking one is
  how the wrong person's embedding ends up attached to a name.
- **`video_path` is restricted to an allowlist** (`ALLOWED_VIDEO_DIRS` in `main.py`).
  The API has no authentication, so an unrestricted path would be an arbitrary file
  read for anyone on the LAN.
- **The gallery is biometric data.** It lives in `face_gallery/` and is gitignored.
  Treat it accordingly.
- Recognition is ~65ms per image end to end (detect + align + embed + match) on a
  720p frame; video runs ~88ms per sampled frame including decode.

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
- **`face.py`** — SCRFD face detection, ArcFace embedding, and the enrolled-person gallery
- **`coco_labels.py`** — COCO-80 label definitions

## Limitations

- **Whisper runs on CPU, not NPU.** The pre-compiled Whisper decoder HEF files from the Hailo S3 bucket are built for HailoRT 5.2.0 and produce incorrect output on HailoRT 5.1.1. Transcription uses CPU-based inference as a workaround (~2s per short clip). Once HailoRT 5.2.0 is available, NPU-accelerated whisper can be revisited.
- **Detection models not included.** YOLO HEF files must be downloaded separately from the Hailo Model Zoo.
- **Single-device concurrency.** The Hailo-10H is shared between detection models using `group_id="SHARED"` with round-robin scheduling. Concurrent heavy workloads may queue.
- **ARM64 only.** Designed for Raspberry Pi 5 with Hailo-10H. The `hailo_platform` Python bindings are platform-specific.

## License

MIT
