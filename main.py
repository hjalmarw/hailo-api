"""Hailo Detection API - FastAPI application."""

import base64
import io
from contextlib import asynccontextmanager
from enum import Enum
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


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    uptime_seconds: int
    whisper: Optional[str] = Field(default=None, description="Whisper model status")


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
    print("All detectors and transcribers closed")


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

    return HealthResponse(
        status="ok",
        model=f"{detector.model_name}-hailo10",
        device="hailo10",
        uptime_seconds=detector.uptime_seconds,
        whisper=whisper_status,
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

    all_models = available + obb_models + hailo_obb_models + whisper_models
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
