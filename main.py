"""Hailo Detection API - FastAPI application."""

import base64
import io
import tempfile
import time
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from detector import (
    get_detector, get_current_model, list_available_models, close_all_detectors,
    HailoDetector, get_obb_detector, is_obb_model, is_hailo_obb_model,
    get_hailo_obb_detector, DOTA_LABELS
)
from coco_labels import COCO_LABELS
from transcriber import (
    get_transcriber, close_transcriber, list_whisper_models, audio_bytes_to_numpy
)
import face as face_module
from face import (
    analyze_image, analyze_video, enroll_from_image, get_gallery, close_face_models,
    DEFAULT_MIN_FACE_PIXELS, DEFAULT_BLUR_TOLERANCE,
)


class PreprocessingType(str, Enum):
    """Available preprocessing options for satellite/aerial imagery."""
    none = "none"
    clahe = "clahe"  # Contrast Limited Adaptive Histogram Equalization
    histogram_eq = "histogram_eq"  # Standard histogram equalization


def apply_preprocessing(image: Image.Image, preprocessing: PreprocessingType) -> Image.Image:
    """Apply preprocessing to enhance image for detection.

    Args:
        image: PIL Image in RGB mode
        preprocessing: Type of preprocessing to apply

    Returns:
        Preprocessed PIL Image
    """
    if preprocessing == PreprocessingType.none:
        return image

    # Convert PIL to numpy array
    img_array = np.array(image)

    if preprocessing == PreprocessingType.clahe:
        # Apply CLAHE to L channel in LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    elif preprocessing == PreprocessingType.histogram_eq:
        # Apply histogram equalization to V channel in HSV
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return Image.fromarray(img_array)


# Zoom-aware confidence thresholds for satellite/aerial imagery
# Based on GSD (Ground Sample Distance) analysis - trained on ~0.3m GSD (zoom 20)
ZOOM_CONFIDENCE_MAP = {
    17: 0.70,  # ~2.4m GSD - cars ~2px, very high threshold
    18: 0.55,  # ~1.2m GSD - cars ~6px, high threshold
    19: 0.40,  # ~0.6m GSD - cars ~25px, moderate threshold
    20: 0.25,  # ~0.3m GSD - cars ~100px, optimal (default)
    21: 0.30,  # ~0.15m GSD - cars ~400px, slightly higher
}

def get_zoom_adjusted_confidence(zoom: Optional[int], user_confidence: float) -> tuple[float, Optional[str]]:
    """Get zoom-adjusted confidence threshold and optional warning."""
    if zoom is None:
        return user_confidence, None

    base_confidence = ZOOM_CONFIDENCE_MAP.get(zoom, 0.25)
    effective_confidence = max(base_confidence, user_confidence)

    warning = None
    if zoom < 19:
        warning = f"Zoom {zoom} has reduced accuracy (GSD too large). Recommend zoom 19-20 for best results."

    return effective_confidence, warning


# Request/Response models
class DetectionRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded JPG or PNG image")
    min_confidence: float = Field(default=0.4, ge=0.0, le=1.0, description="Minimum confidence threshold")
    labels: Optional[list[str]] = Field(default=None, description="Filter to specific labels")
    # Model selection
    model: Optional[str] = Field(default=None, description="Model name (e.g., 'yolov8n', 'yolov8x'). Default: yolov8n")
    # Tiling options for large/aerial images
    tiling: bool = Field(default=False, description="Enable tiling for large images")
    tile_size: int = Field(default=640, ge=320, le=1280, description="Tile size in pixels")
    tile_overlap: float = Field(default=0.25, ge=0.0, le=0.5, description="Overlap between tiles (0.25 = 25%)")
    iou_threshold: float = Field(default=0.5, ge=0.1, le=0.9, description="IoU threshold for NMS merging")
    # Preprocessing for satellite/aerial imagery
    preprocessing: PreprocessingType = Field(default=PreprocessingType.none, description="Image preprocessing: none, clahe, histogram_eq")
    # Zoom level for satellite/aerial imagery (affects confidence threshold)
    zoom: Optional[int] = Field(default=None, ge=1, le=22, description="Map zoom level (17-21). Auto-adjusts confidence for GSD.")


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: BoundingBox


class ImageSize(BaseModel):
    width: int
    height: int


class DetectionResponse(BaseModel):
    detections: list[Detection]
    image_size: ImageSize
    model: str
    processing_ms: int
    tiles: Optional[int] = Field(default=None, description="Number of tiles processed (if tiling enabled)")
    preprocessing: Optional[str] = Field(default=None, description="Preprocessing applied (if any)")
    warning: Optional[str] = Field(default=None, description="Warning message (e.g., low zoom level)")
    effective_confidence: Optional[float] = Field(default=None, description="Actual confidence used (may be adjusted for zoom)")


class TranscribeRequest(BaseModel):
    audio: str = Field(..., description="Base64-encoded audio (WAV, MP3, or any ffmpeg-supported format)")
    language: str = Field(default="en", description="Language code")
    variant: str = Field(default="tiny.en", description="Whisper variant: tiny, tiny.en")


class TranscribeResponse(BaseModel):
    text: str
    language: str
    model: str
    processing_ms: int
    audio_duration_s: float


class FaceEnrollRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Person's name")
    image: str = Field(..., description="Base64-encoded image containing exactly one face")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Face detection threshold")


class FaceEnrollResponse(BaseModel):
    name: str
    detection_confidence: float
    sharpness: float
    face_pixels: int
    total_embeddings: int


class FaceIdentifyRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded JPG or PNG image")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Face detection threshold")
    match_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Cosine similarity needed to claim a match")
    min_face_pixels: float = Field(default=DEFAULT_MIN_FACE_PIXELS, ge=0, description="Reject faces smaller than this area in pixels")
    blur_tolerance: float = Field(default=DEFAULT_BLUR_TOLERANCE, ge=0, description="Reject faces blurrier than this (variance of Laplacian)")
    identify: bool = Field(default=True, description="Set false to detect faces without matching against the gallery")
    min_margin: float = Field(default=0.0, ge=0.0, le=1.0, description="Abstain unless the best match leads the runner-up by this much. On real camera footage 0.2 lifted precision from 76.6% to 94.3%, at the cost of answering less often. 0 disables.")


class FaceResult(BaseModel):
    bbox: list[float] = Field(..., description="[x_min, y_min, x_max, y_max] in original image pixels")
    detection_confidence: float
    landmarks: list[list[float]] = Field(..., description="5 points: eyes, nose, mouth corners")
    name: Optional[str] = Field(default=None, description="Matched person, or null if unknown or abstained")
    similarity: Optional[float] = Field(default=None, description="Cosine similarity to the matched person")
    margin: Optional[float] = Field(default=None, description="Lead over the runner-up. Low means the evidence did not favour one person.")
    sharpness: Optional[float] = None
    skipped: Optional[str] = Field(default=None, description="Why this face was not identified")
    abstained: Optional[bool] = Field(default=None, description="True when a match existed but the margin was too thin to claim")
    best_guess: Optional[str] = Field(default=None, description="Who it would have been, had the margin gate not fired")
    best_guess_similarity: Optional[float] = None
    best_guess_margin: Optional[float] = None


class FaceIdentifyResponse(BaseModel):
    faces: list[FaceResult]
    detected: int
    recognized: int
    abstained: int = Field(default=0, description="Faces with a match too thin to claim")
    too_small: int
    too_blurry: int
    image_size: ImageSize
    processing_ms: int


class VideoStrategy(str, Enum):
    """How frames are chosen for analysis."""
    adaptive = "adaptive"  # motion-gate, track, embed only the best frames per person
    uniform = "uniform"    # fixed stride, embed every face found


class FaceVideoRequest(BaseModel):
    video: Optional[str] = Field(default=None, description="Base64-encoded video file")
    video_path: Optional[str] = Field(default=None, description="Path to a video file, restricted to allowed directories")
    strategy: VideoStrategy = Field(default=VideoStrategy.adaptive, description="'adaptive' picks frames by motion and face quality; 'uniform' uses a fixed stride")
    sample_every: int = Field(default=5, ge=1, le=300, description="Examine every Nth frame. With 'adaptive' this can be small — the motion gate does the real filtering.")
    max_frames: int = Field(default=2000, ge=1, le=100000, description="Cap on frames examined")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    match_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_face_pixels: float = Field(default=DEFAULT_MIN_FACE_PIXELS, ge=0, description="Reject faces below this area. The default (~110x110) is often too strict for cameras viewing at distance — check faces_rejected_small before concluding nobody was there.")
    blur_tolerance: float = Field(default=DEFAULT_BLUR_TOLERANCE, ge=0)
    # Adaptive-only controls
    motion_threshold: float = Field(default=2.0, ge=0.0, le=255.0, description="Mean pixel change needed to analyse a frame. 0 disables the motion gate.")
    embeddings_per_track: int = Field(default=3, ge=1, le=10, description="How many of the best frames per person get embedded and voted on")
    track_iou: float = Field(default=0.3, ge=0.0, le=1.0, description="Box overlap needed to treat a detection as the same person")
    track_max_gap_s: float = Field(default=2.0, ge=0.0, le=60.0, description="Seconds of absence before a person is treated as a new appearance")
    min_margin: float = Field(default=0.0, ge=0.0, le=1.0, description="Abstain unless the best match leads the runner-up by this much. 0.2 measured 94.3% precision on real footage vs 76.6% ungated.")
    min_track_quality: float = Field(default=0.02, ge=0.0, le=1.0, description="Skip embedding appearances below this quality — a one-frame profile glance cannot be identified anyway. Set 0 to embed everything.")
    include_crops: bool = Field(default=False, description="Return the best face crop per track as base64 JPEG")


class PersonSighting(BaseModel):
    name: str
    sightings: int
    best_similarity: float
    first_seen_s: float
    last_seen_s: float


class TrackResult(BaseModel):
    track_id: int
    name: Optional[str] = Field(default=None, description="Matched person, or null if nobody matched")
    similarity: Optional[float] = None
    votes: str = Field(..., description="How many of the embedded frames agreed, e.g. '3/3'")
    detections: int = Field(..., description="Frames this person was detected in")
    first_seen_s: float
    last_seen_s: float
    best_frame_s: float = Field(..., description="Timestamp of the highest-quality frame")
    best_quality: float
    sharpness: float
    frontality: float = Field(..., description="1.0 = face-on, 0.0 = profile")
    face_pixels: int
    best_crop_jpeg_b64: Optional[str] = Field(default=None, description="Best face crop, if include_crops was set")


class FaceVideoResponse(BaseModel):
    strategy: str
    people: list[PersonSighting]
    tracks: list[TrackResult] = Field(..., description="One entry per distinct appearance, including unmatched ones")
    unknown_tracks: int = Field(..., description="Appearances by people who matched nobody enrolled")
    unknown_face_sightings: int
    unidentifiable_tracks: list[dict] = Field(default_factory=list, description="Appearances too low-quality to embed (e.g. a single profile glance). Reported rather than dropped, so a person passing through is never silently absent.")
    frames_sampled: int = Field(..., description="Frames examined after the stride")
    frames_skipped_no_motion: int = Field(..., description="Frames the motion gate discarded before any inference")
    frames_analysed: int = Field(..., description="Frames that actually reached face detection")
    frames_with_faces: int
    faces_rejected_small: int
    faces_rejected_blurry: int
    embeddings_computed: int = Field(..., description="ArcFace runs — the expensive operation")
    video_frames: int
    fps: float
    duration_s: Optional[float]
    processing_ms: int
    truncated: bool = Field(..., description="True if max_frames stopped the scan before the end")


class PersonEntry(BaseModel):
    name: str
    embeddings: int


class PersonsResponse(BaseModel):
    persons: list[PersonEntry]
    count: int


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    uptime_seconds: int
    whisper: Optional[str] = Field(default=None, description="Whisper model status")
    faces: Optional[str] = Field(default=None, description="Face gallery status")


class LabelsResponse(BaseModel):
    labels: list[str]
    count: int


class ErrorResponse(BaseModel):
    error: str




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print("Initializing Hailo detector...")
    try:
        detector = get_detector()  # Load default model
        print(f"Detector initialized: {detector.model_name}")
        print(f"Input size: {detector.input_size}")
        print(f"Available models: {list_available_models()}")
    except Exception as e:
        print(f"Warning: Failed to initialize detector: {e}")
    yield
    # Cleanup
    close_all_detectors()
    close_transcriber()
    close_face_models()
    print("All detectors, transcribers and face models closed")


app = FastAPI(
    title="Hailo Detection API",
    description="Object detection API using Hailo AI accelerator",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/api/v1/detect",
          response_model=DetectionResponse,
          responses={
              400: {"model": ErrorResponse},
              503: {"model": ErrorResponse}
          })
async def detect(request: DetectionRequest):
    """Run object detection on a base64-encoded image."""
    # Check if OBB model requested
    use_hailo_obb = is_hailo_obb_model(request.model) if request.model else False
    use_obb = is_obb_model(request.model) if request.model else False

    # Get detector for requested model
    try:
        if use_hailo_obb:
            # Hailo-accelerated OBB (~50ms)
            detector = get_hailo_obb_detector(request.model)
        elif use_obb:
            # CPU-based OBB (~1000ms)
            detector = get_obb_detector()
        else:
            # Standard Hailo COCO detector
            detector = get_detector(request.model)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
    except RuntimeError as e:
        return JSONResponse(
            status_code=503,
            content={"error": f"Failed to load model: {e}"}
        )

    if not detector.is_ready:
        return JSONResponse(
            status_code=503,
            content={"error": "Detector not ready"}
        )

    # Decode image
    try:
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid image data: {str(e)}"}
        )

    # Get original image size
    width, height = image.size

    # Apply preprocessing if requested
    if request.preprocessing != PreprocessingType.none:
        image = apply_preprocessing(image, request.preprocessing)

    # Apply zoom-aware confidence adjustment for satellite/aerial imagery
    effective_confidence, zoom_warning = get_zoom_adjusted_confidence(
        request.zoom, request.min_confidence
    )

    # Run detection (with or without tiling)
    try:
        tiles = None
        if request.tiling:
            detections, processing_ms, tiles = detector.detect_tiled(
                image,
                min_confidence=effective_confidence,
                labels=request.labels,
                tile_size=request.tile_size,
                overlap=request.tile_overlap,
                iou_threshold=request.iou_threshold
            )
        else:
            detections, processing_ms = detector.detect(
                image,
                min_confidence=effective_confidence,
                labels=request.labels
            )
    except RuntimeError as e:
        error_msg = str(e)
        # Device busy - return 503
        if "busy" in error_msg.lower() or "in use" in error_msg.lower():
            return JSONResponse(
                status_code=503,
                content={"error": "Hailo device busy - try again shortly"}
            )
        return JSONResponse(
            status_code=500,
            content={"error": f"Detection failed: {error_msg}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Detection failed: {str(e)}"}
        )

    return DetectionResponse(
        detections=[Detection(**d) for d in detections],
        image_size=ImageSize(width=width, height=height),
        model=f"{detector.model_name}-hailo10",
        processing_ms=processing_ms,
        tiles=tiles,
        preprocessing=request.preprocessing.value if request.preprocessing != PreprocessingType.none else None,
        warning=zoom_warning,
        effective_confidence=effective_confidence if request.zoom else None
    )


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    """Get API health status."""
    try:
        detector = get_detector()
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"error": "Hailo device not ready"}
        )

    if not detector.is_ready:
        return JSONResponse(
            status_code=503,
            content={"error": "Hailo device not ready"}
        )

    whisper_models = list_whisper_models()
    whisper_status = f"available ({', '.join(whisper_models)})" if whisper_models else "no models"

    # Report the gallery without touching the NPU — health must stay cheap and must not
    # fail just because the face models have not been loaded on demand yet.
    try:
        persons = get_gallery().persons()
        face_status = (f"{len(persons)} enrolled "
                       f"({sum(p['embeddings'] for p in persons)} embeddings)"
                       if persons else "gallery empty")
    except Exception as e:
        face_status = f"unavailable: {e}"

    return HealthResponse(
        status="ok",
        model=f"{detector.model_name}-hailo10",
        device="hailo10",
        uptime_seconds=detector.uptime_seconds,
        whisper=whisper_status,
        faces=face_status,
    )


class ModelsResponse(BaseModel):
    models: list[str]
    current: Optional[str]
    count: int


@app.get("/api/v1/models", response_model=ModelsResponse)
async def models():
    """Get available detection models."""
    from pathlib import Path

    available = list_available_models()

    # Add OBB models (CPU-based, slower but works for aerial/satellite)
    obb_models = ["yolo11n-obb", "yolo11s-obb", "yolo11m-obb"]

    # Check for Hailo-accelerated OBB models
    obb_hef_dir = Path("/usr/local/hailo/resources/models/hailo10h")
    hailo_obb_models = []
    if obb_hef_dir.exists():
        for hef in obb_hef_dir.glob("*obb*.hef"):
            model_name = hef.stem.replace("_", "-") + "-hailo"
            hailo_obb_models.append(model_name)

    # Whisper speech-to-text models
    whisper_models = [f"whisper-{v}" for v in list_whisper_models()]

    # Face pipeline models, exposed so callers can see the full capability surface
    face_models = ["scrfd_10g-face-detect", "arcface_mobilefacenet-face-embed"]

    all_models = available + obb_models + hailo_obb_models + whisper_models + face_models
    return ModelsResponse(
        models=all_models,
        current=get_current_model(),
        count=len(all_models)
    )


@app.get("/api/v1/labels", response_model=LabelsResponse)
async def labels(model_type: str = "coco"):
    """Get available detection labels.

    Args:
        model_type: 'coco' for standard YOLO models, 'dota' for OBB aerial models
    """
    if model_type.lower() == "dota":
        return LabelsResponse(
            labels=DOTA_LABELS,
            count=len(DOTA_LABELS)
        )
    return LabelsResponse(
        labels=COCO_LABELS,
        count=len(COCO_LABELS)
    )


@app.post("/api/v1/transcribe",
          response_model=TranscribeResponse,
          responses={
              400: {"model": ErrorResponse},
              503: {"model": ErrorResponse}
          })
async def transcribe(request: TranscribeRequest):
    """Transcribe audio to text using Whisper on Hailo NPU."""
    # Validate variant
    available = list_whisper_models()
    if request.variant not in available:
        return JSONResponse(
            status_code=400,
            content={"error": f"Variant '{request.variant}' not available. Available: {available}"}
        )

    # Decode audio
    try:
        audio_bytes = base64.b64decode(request.audio)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid base64 audio data: {e}"}
        )

    # Convert to numpy
    try:
        audio_np = audio_bytes_to_numpy(audio_bytes)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to decode audio: {e}"}
        )

    # Get transcriber
    try:
        transcriber = get_transcriber(request.variant)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"error": f"Failed to load whisper model: {e}"}
        )

    # Transcribe
    try:
        text, processing_ms, audio_duration_s = transcriber.transcribe(audio_np, language=request.language)
    except RuntimeError as e:
        error_msg = str(e)
        if "busy" in error_msg.lower():
            return JSONResponse(
                status_code=503,
                content={"error": "Hailo device busy - try again shortly"}
            )
        return JSONResponse(
            status_code=500,
            content={"error": f"Transcription failed: {error_msg}"}
        )

    return TranscribeResponse(
        text=text,
        language=request.language,
        model=f"whisper-{request.variant}",
        processing_ms=processing_ms,
        audio_duration_s=audio_duration_s,
    )


# Directories that /faces/identify_video may read from. The API has no authentication,
# so an unrestricted path parameter would hand any host on the LAN an arbitrary file
# read. Extend this list deliberately rather than removing the check.
ALLOWED_VIDEO_DIRS = [
    Path("/home/grazzy/media"),
    Path("/home/grazzy/projects/hailo-api/videos"),
    Path("/usr/local/hailo/resources/videos"),
    Path("/mnt"),
]


def resolve_video_path(raw_path: str) -> Path:
    """Resolve a caller-supplied video path, refusing anything outside the allowlist."""
    candidate = Path(raw_path).resolve()
    for allowed in ALLOWED_VIDEO_DIRS:
        try:
            if candidate.is_relative_to(allowed.resolve()):
                break
        except (OSError, ValueError):
            continue
    else:
        raise ValueError(
            f"Path not permitted. Allowed roots: {', '.join(str(d) for d in ALLOWED_VIDEO_DIRS)}"
        )

    if not candidate.is_file():
        raise ValueError(f"No such file: {candidate}")
    return candidate


@app.post("/api/v1/faces/enroll",
          response_model=FaceEnrollResponse,
          responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def faces_enroll(request: FaceEnrollRequest):
    """Add a person to the recognition gallery from a photo containing one face.

    Call repeatedly with different photos of the same name to improve robustness —
    varied lighting and angle materially improve recognition on camera footage.
    """
    try:
        image = Image.open(io.BytesIO(base64.b64decode(request.image)))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image data: {e}"})

    try:
        result = enroll_from_image(request.name, image, min_confidence=request.min_confidence)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except RuntimeError as e:
        if "busy" in str(e).lower():
            return JSONResponse(status_code=503, content={"error": "Hailo device busy - try again shortly"})
        return JSONResponse(status_code=503, content={"error": str(e)})

    total = sum(p["embeddings"] for p in get_gallery().persons() if p["name"] == request.name)
    return FaceEnrollResponse(**result, total_embeddings=total)


@app.post("/api/v1/faces/identify",
          response_model=FaceIdentifyResponse,
          responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def faces_identify(request: FaceIdentifyRequest):
    """Detect every face in an image and match each against the enrolled gallery.

    Faces failing a quality gate are still returned, carrying a `skipped` reason, so a
    caller can tell "nobody was there" apart from "the face was too small to judge".
    """
    try:
        image = Image.open(io.BytesIO(base64.b64decode(request.image)))
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image data: {e}"})

    width, height = image.size
    start = time.perf_counter()

    try:
        faces, stats = analyze_image(
            image,
            min_confidence=request.min_confidence,
            match_threshold=request.match_threshold,
            min_face_pixels=request.min_face_pixels,
            blur_tolerance=request.blur_tolerance,
            identify=request.identify,
            min_margin=request.min_margin,
        )
    except RuntimeError as e:
        if "busy" in str(e).lower():
            return JSONResponse(status_code=503, content={"error": "Hailo device busy - try again shortly"})
        return JSONResponse(status_code=500, content={"error": f"Face analysis failed: {e}"})

    return FaceIdentifyResponse(
        faces=[FaceResult(**f) for f in faces],
        detected=stats["detected"],
        recognized=stats["recognized"],
        abstained=stats.get("abstained", 0),
        too_small=stats["too_small"],
        too_blurry=stats["too_blurry"],
        image_size=ImageSize(width=width, height=height),
        processing_ms=int((time.perf_counter() - start) * 1000),
    )


@app.post("/api/v1/faces/identify_video",
          response_model=FaceVideoResponse,
          responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def faces_identify_video(request: FaceVideoRequest):
    """Identify people across a video, aggregating sightings per person.

    Supply either `video` (base64) or `video_path` (a file under an allowed root).
    Results are per-person rather than per-frame: how often someone appeared, their
    best match confidence, and the window they were visible in.
    """
    if not request.video and not request.video_path:
        return JSONResponse(status_code=400, content={"error": "Provide either 'video' or 'video_path'"})
    if request.video and request.video_path:
        return JSONResponse(status_code=400, content={"error": "Provide only one of 'video' or 'video_path'"})

    temp_path = None
    try:
        if request.video_path:
            try:
                path = resolve_video_path(request.video_path)
            except ValueError as e:
                return JSONResponse(status_code=400, content={"error": str(e)})
        else:
            try:
                data = base64.b64decode(request.video)
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid base64 video: {e}"})
            handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            handle.write(data)
            handle.close()
            temp_path = Path(handle.name)
            path = temp_path

        try:
            result = analyze_video(
                str(path),
                sample_every=request.sample_every,
                max_frames=request.max_frames,
                min_confidence=request.min_confidence,
                match_threshold=request.match_threshold,
                min_face_pixels=request.min_face_pixels,
                blur_tolerance=request.blur_tolerance,
                strategy=request.strategy.value,
                motion_threshold=request.motion_threshold,
                embeddings_per_track=request.embeddings_per_track,
                track_iou=request.track_iou,
                track_max_gap_s=request.track_max_gap_s,
                min_track_quality=request.min_track_quality,
                min_margin=request.min_margin,
                include_crops=request.include_crops,
            )
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except RuntimeError as e:
            if "busy" in str(e).lower():
                return JSONResponse(status_code=503, content={"error": "Hailo device busy - try again shortly"})
            return JSONResponse(status_code=500, content={"error": f"Video analysis failed: {e}"})
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return FaceVideoResponse(
        strategy=result["strategy"],
        people=[PersonSighting(**p) for p in result["people"]],
        tracks=[TrackResult(**t) for t in result["tracks"]],
        unknown_tracks=result["unknown_tracks"],
        unknown_face_sightings=result["unknown_face_sightings"],
        unidentifiable_tracks=result["unidentifiable_tracks"],
        frames_sampled=result["frames_sampled"],
        frames_skipped_no_motion=result["frames_skipped_no_motion"],
        frames_analysed=result["frames_analysed"],
        frames_with_faces=result["frames_with_faces"],
        faces_rejected_small=result["faces_rejected_small"],
        faces_rejected_blurry=result["faces_rejected_blurry"],
        embeddings_computed=result["embeddings_computed"],
        video_frames=result["video_frames"],
        fps=result["fps"],
        duration_s=result["duration_s"],
        processing_ms=result["processing_ms"],
        truncated=result["truncated"],
    )


@app.get("/api/v1/faces/persons", response_model=PersonsResponse)
async def faces_persons():
    """List everyone enrolled in the gallery and how many embeddings each has."""
    persons = get_gallery().persons()
    return PersonsResponse(
        persons=[PersonEntry(**p) for p in persons],
        count=len(persons),
    )


@app.delete("/api/v1/faces/persons/{name}",
            responses={404: {"model": ErrorResponse}})
async def faces_delete_person(name: str):
    """Remove a person and all their embeddings from the gallery."""
    removed = get_gallery().delete(name)
    if removed == 0:
        return JSONResponse(status_code=404, content={"error": f"No person named '{name}' in the gallery"})
    return {"name": name, "removed_embeddings": removed}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
