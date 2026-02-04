"""Hailo-based YOLOv8 object detector with persistent model loading and tiling support."""

import time
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List, Tuple
import threading

from hailo_platform import HEF, VDevice, FormatType, HailoSchedulingAlgorithm

from coco_labels import COCO_LABELS

# Default model path for Hailo10H
DEFAULT_MODEL_PATH = "/usr/local/hailo/resources/models/hailo10h/yolov8n.hef"


def compute_iou(box1: dict, box2: dict) -> float:
    """Compute IoU between two bounding boxes (normalized coordinates)."""
    x1 = max(box1["x_min"], box2["x_min"])
    y1 = max(box1["y_min"], box2["y_min"])
    x2 = min(box1["x_max"], box2["x_max"])
    y2 = min(box1["y_max"], box2["y_max"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1["x_max"] - box1["x_min"]) * (box1["y_max"] - box1["y_min"])
    area2 = (box2["x_max"] - box2["x_min"]) * (box2["y_max"] - box2["y_min"])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def nms_detections(detections: List[dict], iou_threshold: float = 0.5) -> List[dict]:
    """Apply Non-Maximum Suppression to merge overlapping detections."""
    if not detections:
        return []

    # Sort by confidence (highest first)
    sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    kept = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        kept.append(best)

        # Remove detections with high IoU overlap (same class only)
        sorted_dets = [
            det for det in sorted_dets
            if det["label"] != best["label"] or
               compute_iou(det["bbox"], best["bbox"]) < iou_threshold
        ]

    return kept


def parse_nms_output(raw_output: list) -> list:
    """Parse HAILO NMS BY CLASS output format.

    The output is a list of 80 numpy arrays (one per class).
    Each array has shape (num_detections, 5) where 5 = [y_min, x_min, y_max, x_max, score].
    Coordinates are normalized 0-1.

    Returns:
        List of tuples: (class_id, y_min, x_min, y_max, x_max, score)
    """
    detections = []

    for class_id, class_detections in enumerate(raw_output):
        for det in class_detections:
            y_min, x_min, y_max, x_max, score = det
            detections.append((class_id, y_min, x_min, y_max, x_max, score))

    return detections


class HailoDetector:
    """YOLOv8 object detector using Hailo accelerator.

    Uses persistent model loading with SHARED group_id to allow other
    applications (like temp_monitor using Device API) to coexist.
    The VDevice with ROUND_ROBIN scheduler allows the Device API to
    still access the hardware for control operations.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.model_name = Path(self.model_path).stem
        self.labels = COCO_LABELS
        self.start_time = time.time()

        # Lock to serialize inference requests
        self._inference_lock = threading.Lock()

        # Persistent device and model references
        self._device = None
        self._infer_model = None
        self._config_ctx = None
        self._configured_model = None
        self._hef = None
        self._input_shape = None

    def initialize(self) -> bool:
        """Initialize detector with persistent device and model."""
        try:
            # Load HEF for metadata
            self._hef = HEF(self.model_path)
            self._input_shape = self._hef.get_input_vstream_infos()[0].shape

            # Create VDevice with SHARED group to allow Device API coexistence
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            params.group_id = "SHARED"
            self._device = VDevice(params)

            # Create and configure inference model
            self._infer_model = self._device.create_infer_model(self.model_path)
            self._infer_model.set_batch_size(1)

            # Set output format to float32
            for output in self._infer_model.outputs:
                output.set_format_type(FormatType.FLOAT32)

            # Configure model (keep context open)
            self._config_ctx = self._infer_model.configure()
            self._configured_model = self._config_ctx.__enter__()

            return True
        except Exception as e:
            print(f"Failed to initialize Hailo device: {e}")
            self.close()
            return False

    @property
    def is_ready(self) -> bool:
        """Check if detector is ready."""
        return self._configured_model is not None

    @property
    def input_size(self) -> tuple:
        """Get model input size (height, width)."""
        if self._input_shape:
            return (self._input_shape[0], self._input_shape[1])
        return (640, 640)

    @property
    def uptime_seconds(self) -> int:
        """Get uptime in seconds."""
        return int(time.time() - self.start_time)

    def preprocess(self, image: Image.Image) -> tuple:
        """Preprocess image for inference."""
        original_size = image.size  # (width, height)
        target_h, target_w = self.input_size

        if image.mode != 'RGB':
            image = image.convert('RGB')

        orig_w, orig_h = original_size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = image.resize((new_w, new_h), Image.BILINEAR)

        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        padded = Image.new('RGB', (target_w, target_h), (114, 114, 114))
        padded.paste(resized, (pad_w, pad_h))

        img_array = np.array(padded, dtype=np.uint8)

        return img_array, original_size, scale, (pad_w, pad_h)

    def postprocess(self, raw_output: list, original_size: tuple, scale: float,
                    pad: tuple, min_confidence: float = 0.4,
                    label_filter: Optional[list] = None) -> list:
        """Postprocess model outputs to get detections."""
        detections = []
        target_h, target_w = self.input_size
        pad_w, pad_h = pad
        orig_w, orig_h = original_size

        filter_indices = None
        if label_filter:
            filter_indices = set()
            for label in label_filter:
                label_lower = label.lower()
                for idx, coco_label in enumerate(self.labels):
                    if coco_label.lower() == label_lower:
                        filter_indices.add(idx)

        raw_detections = parse_nms_output(raw_output)

        for class_id, y_min, x_min, y_max, x_max, score in raw_detections:
            if score < min_confidence:
                continue

            if filter_indices and class_id not in filter_indices:
                continue

            x_min_px = x_min * target_w
            y_min_px = y_min * target_h
            x_max_px = x_max * target_w
            y_max_px = y_max * target_h

            x_min_px -= pad_w
            y_min_px -= pad_h
            x_max_px -= pad_w
            y_max_px -= pad_h

            x_min_orig = x_min_px / scale
            y_min_orig = y_min_px / scale
            x_max_orig = x_max_px / scale
            y_max_orig = y_max_px / scale

            x_min_norm = max(0.0, min(1.0, x_min_orig / orig_w))
            y_min_norm = max(0.0, min(1.0, y_min_orig / orig_h))
            x_max_norm = max(0.0, min(1.0, x_max_orig / orig_w))
            y_max_norm = max(0.0, min(1.0, y_max_orig / orig_h))

            if x_max_norm <= x_min_norm or y_max_norm <= y_min_norm:
                continue

            label = self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}"

            detections.append({
                "label": label,
                "confidence": round(float(score), 2),
                "bbox": {
                    "x_min": round(float(x_min_norm), 2),
                    "y_min": round(float(y_min_norm), 2),
                    "x_max": round(float(x_max_norm), 2),
                    "y_max": round(float(y_max_norm), 2)
                }
            })

        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return detections

    def _run_inference(self, input_data: np.ndarray) -> list:
        """Run inference on preprocessed input data."""
        with self._inference_lock:
            try:
                output_buffers = {}
                for output_info in self._hef.get_output_vstream_infos():
                    output_buffers[output_info.name] = np.empty(
                        self._infer_model.output(output_info.name).shape,
                        dtype=np.float32
                    )

                bindings = self._configured_model.create_bindings(output_buffers=output_buffers)
                bindings.input().set_buffer(input_data)
                self._configured_model.run([bindings], timeout=10000)

                return bindings.output().get_buffer()

            except Exception as e:
                if "74" in str(e) or "OUT_OF_PHYSICAL_DEVICES" in str(e):
                    raise RuntimeError("Hailo device busy")
                raise RuntimeError(f"Inference failed: {e}")

    def detect(self, image: Image.Image, min_confidence: float = 0.4,
               labels: Optional[list] = None) -> tuple:
        """Run detection on an image."""
        if not self.is_ready:
            raise RuntimeError("Detector not initialized")

        start_time = time.perf_counter()

        # Preprocess
        input_data, original_size, scale, pad = self.preprocess(image)

        # Run inference
        raw_output = self._run_inference(input_data)

        # Postprocess
        detections = self.postprocess(raw_output, original_size, scale, pad,
                                       min_confidence, labels)

        processing_ms = int((time.perf_counter() - start_time) * 1000)
        return detections, processing_ms

    def detect_tiled(self, image: Image.Image, min_confidence: float = 0.4,
                     labels: Optional[list] = None, tile_size: int = 640,
                     overlap: float = 0.25, iou_threshold: float = 0.5) -> tuple:
        """Run detection with tiling for large/aerial images.

        Args:
            image: Input image (any size)
            min_confidence: Minimum detection confidence
            labels: Filter to specific labels
            tile_size: Size of each tile (default 640 for YOLO)
            overlap: Overlap between tiles as fraction (0.25 = 25%)
            iou_threshold: IoU threshold for NMS merging

        Returns:
            (detections, processing_ms, tile_count)
        """
        if not self.is_ready:
            raise RuntimeError("Detector not initialized")

        start_time = time.perf_counter()

        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_w, img_h = image.size
        target_h, target_w = self.input_size

        # If image is small enough, use regular detection
        if img_w <= tile_size and img_h <= tile_size:
            dets, proc_ms = self.detect(image, min_confidence, labels)
            return dets, proc_ms, 1

        # Calculate tile positions with overlap
        stride = int(tile_size * (1 - overlap))
        all_detections = []
        tile_count = 0

        for y in range(0, img_h, stride):
            for x in range(0, img_w, stride):
                # Calculate tile boundaries
                x_end = min(x + tile_size, img_w)
                y_end = min(y + tile_size, img_h)
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)

                # Extract tile
                tile = image.crop((x_start, y_start, x_end, y_end))
                tile_w, tile_h = tile.size

                # Preprocess and run inference
                input_data, _, scale, pad = self.preprocess(tile)
                raw_output = self._run_inference(input_data)

                # Get detections for this tile
                tile_dets = self.postprocess(
                    raw_output, (tile_w, tile_h), scale, pad,
                    min_confidence, labels
                )

                # Transform coordinates back to original image space
                for det in tile_dets:
                    bbox = det["bbox"]
                    # Convert from tile-relative to image-relative
                    det["bbox"] = {
                        "x_min": round((x_start + bbox["x_min"] * tile_w) / img_w, 4),
                        "y_min": round((y_start + bbox["y_min"] * tile_h) / img_h, 4),
                        "x_max": round((x_start + bbox["x_max"] * tile_w) / img_w, 4),
                        "y_max": round((y_start + bbox["y_max"] * tile_h) / img_h, 4),
                    }
                    all_detections.append(det)

                tile_count += 1

        # Apply NMS to merge overlapping detections from tile boundaries
        merged_detections = nms_detections(all_detections, iou_threshold)

        processing_ms = int((time.perf_counter() - start_time) * 1000)
        return merged_detections, processing_ms, tile_count

    def close(self):
        """Clean up resources."""
        if self._config_ctx:
            try:
                self._config_ctx.__exit__(None, None, None)
            except:
                pass
        self._configured_model = None
        self._infer_model = None
        self._device = None
        self._hef = None


# Available models directory
MODELS_DIR = "/usr/local/hailo/resources/models/hailo10h"

# Cache of loaded detectors (model_name -> HailoDetector)
_detector_cache: dict[str, HailoDetector] = {}
_current_model: Optional[str] = None
_cache_lock = threading.Lock()

# Maximum number of models to keep loaded (to manage memory)
MAX_CACHED_MODELS = 2


def list_available_models() -> list[str]:
    """List available model names."""
    models_path = Path(MODELS_DIR)
    if not models_path.exists():
        return []
    return sorted([f.stem for f in models_path.glob("*.hef")])


def get_detector(model_name: Optional[str] = None) -> HailoDetector:
    """Get or create a detector for the specified model.

    Args:
        model_name: Model name (without .hef extension). If None, uses default (yolov8n).

    Returns:
        Initialized HailoDetector instance.
    """
    global _detector_cache, _current_model

    # Default to yolov8n if not specified
    if model_name is None:
        model_name = "yolov8n"

    model_path = Path(MODELS_DIR) / f"{model_name}.hef"
    if not model_path.exists():
        available = list_available_models()
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")

    with _cache_lock:
        # Return cached detector if available
        if model_name in _detector_cache:
            _current_model = model_name
            return _detector_cache[model_name]

        # Need to load new model - first check cache size
        if len(_detector_cache) >= MAX_CACHED_MODELS:
            # Remove oldest cached model (not the one we're about to use)
            for old_name in list(_detector_cache.keys()):
                if old_name != _current_model:
                    old_detector = _detector_cache.pop(old_name)
                    old_detector.close()
                    print(f"Unloaded model: {old_name}")
                    break

        # Load new model
        print(f"Loading model: {model_name}")
        detector = HailoDetector(str(model_path))
        if not detector.initialize():
            raise RuntimeError(f"Failed to initialize model: {model_name}")

        _detector_cache[model_name] = detector
        _current_model = model_name
        print(f"Model loaded: {model_name}")
        return detector


def get_current_model() -> Optional[str]:
    """Get the name of the currently active model."""
    return _current_model


def close_all_detectors():
    """Close all cached detectors."""
    global _detector_cache
    with _cache_lock:
        for name, detector in _detector_cache.items():
            detector.close()
            print(f"Closed detector: {name}")
        _detector_cache.clear()


# =============================================================================
# OBB (Oriented Bounding Box) Detector using Ultralytics YOLO11-OBB
# For aerial/satellite imagery with rotated bounding boxes
# =============================================================================

# DOTA dataset class labels
DOTA_LABELS = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool"
]

# Cache for OBB detector (separate from Hailo detectors)
_obb_detector = None
_obb_lock = threading.Lock()


class OBBDetector:
    """YOLO11-OBB detector for aerial/satellite imagery using Ultralytics.

    Uses CPU inference (slower but compatible). Returns oriented bounding boxes
    with rotation angle for vehicles, ships, planes, etc.
    """

    def __init__(self, model_name: str = "yolo11n-obb"):
        self.model_name = model_name
        self.model = None
        self.labels = DOTA_LABELS
        self.start_time = time.time()
        self._inference_lock = threading.Lock()

    def initialize(self) -> bool:
        """Initialize the OBB detector."""
        try:
            from ultralytics import YOLO
            print(f"Loading OBB model: {self.model_name}")
            self.model = YOLO(f"{self.model_name}.pt")
            print(f"OBB model loaded: {self.model_name}")
            return True
        except Exception as e:
            print(f"Failed to load OBB model: {e}")
            return False

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def input_size(self) -> tuple:
        return (1024, 1024)  # OBB models typically use larger input

    @property
    def uptime_seconds(self) -> int:
        return int(time.time() - self.start_time)

    def detect(self, image: Image.Image, min_confidence: float = 0.25,
               labels: Optional[List[str]] = None) -> tuple:
        """Run OBB detection on an image.

        Args:
            image: PIL Image
            min_confidence: Minimum confidence threshold
            labels: Optional filter for specific labels

        Returns:
            (detections, processing_ms)
        """
        if not self.is_ready:
            raise RuntimeError("OBB Detector not initialized")

        start_time = time.perf_counter()

        with self._inference_lock:
            # Run inference
            results = self.model(image, conf=min_confidence, verbose=False)

        # Parse results
        detections = []
        orig_w, orig_h = image.size

        if results[0].obb is not None:
            for i, (cls, conf, xyxyxyxy) in enumerate(zip(
                results[0].obb.cls,
                results[0].obb.conf,
                results[0].obb.xyxyxyxy
            )):
                class_id = int(cls)
                label = self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}"

                # Filter by label if specified
                if labels:
                    if not any(l.lower() == label.lower() for l in labels):
                        continue

                # Get bounding box from rotated corners (xyxyxyxy = 4 corner points)
                # Convert to axis-aligned bbox for compatibility
                corners = xyxyxyxy.cpu().numpy()
                x_coords = corners[:, 0]
                y_coords = corners[:, 1]

                x_min = float(x_coords.min()) / orig_w
                x_max = float(x_coords.max()) / orig_w
                y_min = float(y_coords.min()) / orig_h
                y_max = float(y_coords.max()) / orig_h

                # Clamp to [0, 1]
                x_min = max(0.0, min(1.0, x_min))
                x_max = max(0.0, min(1.0, x_max))
                y_min = max(0.0, min(1.0, y_min))
                y_max = max(0.0, min(1.0, y_max))

                if x_max > x_min and y_max > y_min:
                    detections.append({
                        "label": label,
                        "confidence": round(float(conf), 2),
                        "bbox": {
                            "x_min": round(x_min, 4),
                            "y_min": round(y_min, 4),
                            "x_max": round(x_max, 4),
                            "y_max": round(y_max, 4)
                        }
                    })

        # Sort by confidence
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        processing_ms = int((time.perf_counter() - start_time) * 1000)
        return detections, processing_ms

    def detect_tiled(self, image: Image.Image, min_confidence: float = 0.25,
                     labels: Optional[List[str]] = None, tile_size: int = 1024,
                     overlap: float = 0.25, iou_threshold: float = 0.5) -> tuple:
        """Run OBB detection with tiling (same interface as HailoDetector)."""
        if not self.is_ready:
            raise RuntimeError("OBB Detector not initialized")

        start_time = time.perf_counter()

        img_w, img_h = image.size

        # If image is small enough, use regular detection
        if img_w <= tile_size and img_h <= tile_size:
            dets, _ = self.detect(image, min_confidence, labels)
            processing_ms = int((time.perf_counter() - start_time) * 1000)
            return dets, processing_ms, 1

        # Tile the image
        stride = int(tile_size * (1 - overlap))
        all_detections = []
        tile_count = 0

        for y in range(0, img_h, stride):
            for x in range(0, img_w, stride):
                x_end = min(x + tile_size, img_w)
                y_end = min(y + tile_size, img_h)
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)

                tile = image.crop((x_start, y_start, x_end, y_end))
                tile_w, tile_h = tile.size

                tile_dets, _ = self.detect(tile, min_confidence, labels)

                # Transform to original image coordinates
                for det in tile_dets:
                    bbox = det["bbox"]
                    det["bbox"] = {
                        "x_min": round((x_start + bbox["x_min"] * tile_w) / img_w, 4),
                        "y_min": round((y_start + bbox["y_min"] * tile_h) / img_h, 4),
                        "x_max": round((x_start + bbox["x_max"] * tile_w) / img_w, 4),
                        "y_max": round((y_start + bbox["y_max"] * tile_h) / img_h, 4),
                    }
                    all_detections.append(det)

                tile_count += 1

        # Apply NMS
        merged = nms_detections(all_detections, iou_threshold)

        processing_ms = int((time.perf_counter() - start_time) * 1000)
        return merged, processing_ms, tile_count

    def close(self):
        """Clean up resources."""
        self.model = None


def get_obb_detector() -> OBBDetector:
    """Get or create the OBB detector instance."""
    global _obb_detector

    with _obb_lock:
        if _obb_detector is None:
            _obb_detector = OBBDetector()
            if not _obb_detector.initialize():
                raise RuntimeError("Failed to initialize OBB detector")
        return _obb_detector


def is_obb_model(model_name: str) -> bool:
    """Check if model name is an OBB model."""
    return model_name and "obb" in model_name.lower()


def is_hailo_obb_model(model_name: str) -> bool:
    """Check if model name is a Hailo-accelerated OBB model.

    Returns True if:
    - Model name contains 'obb' AND 'hailo', OR
    - A matching OBB HEF exists in the models directory
    """
    if not model_name:
        return False

    name_lower = model_name.lower()

    # Explicit hailo OBB request
    if "obb" in name_lower and "hailo" in name_lower:
        return True

    # Check if OBB HEF exists for this model
    if "obb" in name_lower:
        # Normalize name to match HEF filename
        hef_name = model_name.replace("_", "-").replace("-hailo", "")
        hef_path = Path(OBB_MODELS_DIR) / f"{hef_name}.hef"
        if hef_path.exists():
            return True

    return False


# =============================================================================
# Hailo-Accelerated OBB Detector
# Hardware-accelerated OBB inference using Hailo-10H NPU
# =============================================================================

# OBB HEF model path
OBB_MODELS_DIR = "/usr/local/hailo/resources/models/hailo10h"

# Cache for Hailo OBB detector
_hailo_obb_detector = None
_hailo_obb_lock = threading.Lock()


def decode_obb_angle(raw_angle: np.ndarray) -> np.ndarray:
    """Decode OBB rotation angle from raw network output.

    Formula: angle = (sigmoid(raw) - 0.25) × π
    This maps the output to [-π/4, 3π/4] range.
    """
    sigmoid = 1.0 / (1.0 + np.exp(-raw_angle))
    return (sigmoid - 0.25) * np.pi


def xywhr_to_xyxyxyxy(x: float, y: float, w: float, h: float, r: float) -> np.ndarray:
    """Convert center (x, y, w, h, rotation) to 4 corner points.

    Args:
        x, y: Center coordinates
        w, h: Width and height
        r: Rotation angle in radians

    Returns:
        Array of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    cos_r = np.cos(r)
    sin_r = np.sin(r)

    # Half dimensions
    hw, hh = w / 2, h / 2

    # Corner offsets before rotation
    corners = np.array([
        [-hw, -hh],
        [hw, -hh],
        [hw, hh],
        [-hw, hh]
    ])

    # Rotation matrix
    rot_matrix = np.array([
        [cos_r, -sin_r],
        [sin_r, cos_r]
    ])

    # Rotate and translate
    rotated = corners @ rot_matrix.T
    rotated[:, 0] += x
    rotated[:, 1] += y

    return rotated


def compute_obb_iou(box1_corners: np.ndarray, box2_corners: np.ndarray) -> float:
    """Compute IoU between two oriented bounding boxes using Shapely.

    Falls back to axis-aligned IoU if Shapely not available.
    """
    try:
        from shapely.geometry import Polygon
        poly1 = Polygon(box1_corners)
        poly2 = Polygon(box2_corners)
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return intersection / union if union > 0 else 0.0
    except ImportError:
        # Fallback to axis-aligned IoU
        x1_min, y1_min = box1_corners.min(axis=0)
        x1_max, y1_max = box1_corners.max(axis=0)
        x2_min, y2_min = box2_corners.min(axis=0)
        x2_max, y2_max = box2_corners.max(axis=0)

        xi1, yi1 = max(x1_min, x2_min), max(y1_min, y2_min)
        xi2, yi2 = min(x1_max, x2_max), min(y1_max, y2_max)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0


class HailoOBBDetector:
    """YOLO11-OBB detector using Hailo-10H NPU acceleration.

    Uses Hailo hardware for fast inference (~50ms) on aerial/satellite imagery.
    Outputs oriented bounding boxes with rotation angles.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model_name = Path(model_path).stem if model_path else "yolo11s_obb_hailo10h"
        self.labels = DOTA_LABELS
        self.start_time = time.time()

        self._inference_lock = threading.Lock()
        self._device = None
        self._infer_model = None
        self._config_ctx = None
        self._configured_model = None
        self._hef = None
        self._input_shape = None

    def initialize(self) -> bool:
        """Initialize Hailo OBB detector."""
        if not self.model_path or not Path(self.model_path).exists():
            print(f"OBB HEF not found: {self.model_path}")
            return False

        try:
            # Load HEF
            self._hef = HEF(self.model_path)
            self._input_shape = self._hef.get_input_vstream_infos()[0].shape

            # Create VDevice with SHARED group
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            params.group_id = "SHARED"
            self._device = VDevice(params)

            # Create and configure inference model
            self._infer_model = self._device.create_infer_model(self.model_path)
            self._infer_model.set_batch_size(1)

            # Set output format to float32
            for output in self._infer_model.outputs:
                output.set_format_type(FormatType.FLOAT32)

            # Configure model
            self._config_ctx = self._infer_model.configure()
            self._configured_model = self._config_ctx.__enter__()

            print(f"Hailo OBB detector initialized: {self.model_name}")
            print(f"Input shape: {self._input_shape}")
            print(f"Outputs: {len(self._hef.get_output_vstream_infos())}")

            return True

        except Exception as e:
            print(f"Failed to initialize Hailo OBB detector: {e}")
            self.close()
            return False

    @property
    def is_ready(self) -> bool:
        return self._configured_model is not None

    @property
    def input_size(self) -> tuple:
        if self._input_shape:
            return (self._input_shape[0], self._input_shape[1])
        return (640, 640)

    @property
    def uptime_seconds(self) -> int:
        return int(time.time() - self.start_time)

    def preprocess(self, image: Image.Image) -> tuple:
        """Preprocess image for OBB inference."""
        original_size = image.size
        target_h, target_w = self.input_size

        if image.mode != 'RGB':
            image = image.convert('RGB')

        orig_w, orig_h = original_size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = image.resize((new_w, new_h), Image.BILINEAR)

        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        padded = Image.new('RGB', (target_w, target_h), (114, 114, 114))
        padded.paste(resized, (pad_w, pad_h))

        img_array = np.array(padded, dtype=np.uint8)

        return img_array, original_size, scale, (pad_w, pad_h)

    def postprocess_obb(self, raw_outputs: dict, original_size: tuple,
                        scale: float, pad: tuple, min_confidence: float = 0.25,
                        label_filter: Optional[List[str]] = None) -> List[dict]:
        """Postprocess OBB model outputs from 9 raw tensors (vectorized).

        Implements DFL decoding, anchor generation, and rotated bbox extraction.
        Based on hailo-apps oriented_object_detection C++ implementation.
        """
        detections = []
        pad_w, pad_h = pad
        orig_w, orig_h = original_size

        # Constants
        reg_max = 16
        cls_num = len(self.labels)  # 15 DOTA classes
        strides = [8, 16, 32]
        dfl_weights = np.arange(reg_max, dtype=np.float32)

        # Build filter indices
        filter_indices = None
        if label_filter:
            filter_indices = set()
            for label in label_filter:
                label_lower = label.lower()
                for idx, dota_label in enumerate(self.labels):
                    if dota_label.lower() == label_lower:
                        filter_indices.add(idx)

        # Sort outputs by layer name (alphabetical order)
        sorted_names = sorted(raw_outputs.keys())
        tensors = [raw_outputs[name] for name in sorted_names]

        # Process each scale (vectorized)
        for scale_idx in range(3):
            bbox_idx = scale_idx * 3
            angle_idx = scale_idx * 3 + 1
            class_idx = scale_idx * 3 + 2

            bbox_tensor = tensors[bbox_idx]    # HxWx64
            angle_tensor = tensors[angle_idx]  # HxWx1
            class_tensor = tensors[class_idx]  # HxWx15

            H, W = bbox_tensor.shape[:2]
            N = H * W
            stride = strides[scale_idx]

            # Reshape tensors
            bbox_flat = bbox_tensor.reshape(N, 64).astype(np.float32)
            angle_flat = angle_tensor.reshape(N).astype(np.float32)
            class_flat = class_tensor.reshape(N, cls_num).astype(np.float32)

            # 1. Vectorized angle decode
            angle_sigmoid = 1.0 / (1.0 + np.exp(-angle_flat))
            angle_rad = (angle_sigmoid - 0.25) * np.pi
            cos_r = np.cos(angle_rad)
            sin_r = np.sin(angle_rad)

            # 2. Vectorized DFL decode (softmax + weighted sum)
            bbox_reshaped = bbox_flat.reshape(N, 4, reg_max)  # (N, 4, 16)
            bbox_max = bbox_reshaped.max(axis=2, keepdims=True)
            bbox_exp = np.exp(bbox_reshaped - bbox_max)
            bbox_softmax = bbox_exp / bbox_exp.sum(axis=2, keepdims=True)
            dfl_decoded = (bbox_softmax * dfl_weights).sum(axis=2)  # (N, 4)

            # 3. Vectorized box coordinate conversion
            dfl_div_left = (dfl_decoded[:, 2] - dfl_decoded[:, 0]) / 2.0
            dfl_div_right = (dfl_decoded[:, 3] - dfl_decoded[:, 1]) / 2.0

            offset_x = dfl_div_left * cos_r - dfl_div_right * sin_r
            offset_y = dfl_div_left * sin_r + dfl_div_right * cos_r

            # Anchor positions
            anchor_x = (np.arange(N) % W) + 0.5
            anchor_y = (np.arange(N) // W) + 0.5

            cx = (anchor_x + offset_x) * stride
            cy = (anchor_y + offset_y) * stride
            w = (dfl_decoded[:, 0] + dfl_decoded[:, 2]) * stride
            h = (dfl_decoded[:, 1] + dfl_decoded[:, 3]) * stride

            # 4. Vectorized class scores
            class_scores = 1.0 / (1.0 + np.exp(-class_flat))
            class_ids = np.argmax(class_scores, axis=1)
            confidences = class_scores[np.arange(N), class_ids]

            # 5. Filter by confidence (vectorized)
            mask = confidences >= min_confidence
            mask &= (w >= 2) & (h >= 2)

            if filter_indices:
                label_mask = np.array([cid in filter_indices for cid in class_ids])
                mask &= label_mask

            # Get valid indices
            valid_indices = np.where(mask)[0]

            # Process valid detections
            for idx in valid_indices:
                # Convert to corner points
                corners = xywhr_to_xyxyxyxy(cx[idx], cy[idx], w[idx], h[idx], angle_rad[idx])

                # Remove padding and scale to original image
                corners[:, 0] = (corners[:, 0] - pad_w) / scale
                corners[:, 1] = (corners[:, 1] - pad_h) / scale

                # Normalize to [0, 1]
                corners[:, 0] /= orig_w
                corners[:, 1] /= orig_h

                # Get axis-aligned bbox from corners
                x_min = float(np.clip(corners[:, 0].min(), 0, 1))
                x_max = float(np.clip(corners[:, 0].max(), 0, 1))
                y_min = float(np.clip(corners[:, 1].min(), 0, 1))
                y_max = float(np.clip(corners[:, 1].max(), 0, 1))

                if x_max <= x_min or y_max <= y_min:
                    continue

                class_id = int(class_ids[idx])
                confidence = float(confidences[idx])

                label = self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}"

                detections.append({
                    "label": label,
                    "confidence": round(confidence, 2),
                    "bbox": {
                        "x_min": round(x_min, 4),
                        "y_min": round(y_min, 4),
                        "x_max": round(x_max, 4),
                        "y_max": round(y_max, 4)
                    }
                })

        # Sort by confidence and apply NMS
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        # Simple NMS - remove overlapping detections of same class
        kept = []
        for det in detections:
            should_keep = True
            for kept_det in kept:
                if det["label"] == kept_det["label"]:
                    iou = compute_iou(det["bbox"], kept_det["bbox"])
                    if iou > 0.5:
                        should_keep = False
                        break
            if should_keep:
                kept.append(det)

        return kept

    def _run_inference(self, input_data: np.ndarray) -> dict:
        """Run inference on Hailo NPU."""
        with self._inference_lock:
            try:
                output_buffers = {}
                for output_info in self._hef.get_output_vstream_infos():
                    output_buffers[output_info.name] = np.empty(
                        self._infer_model.output(output_info.name).shape,
                        dtype=np.float32
                    )

                bindings = self._configured_model.create_bindings(output_buffers=output_buffers)
                bindings.input().set_buffer(input_data)
                self._configured_model.run([bindings], timeout=10000)

                # Return all output buffers
                return {name: buf.copy() for name, buf in output_buffers.items()}

            except Exception as e:
                if "74" in str(e) or "OUT_OF_PHYSICAL_DEVICES" in str(e):
                    raise RuntimeError("Hailo device busy")
                raise RuntimeError(f"OBB inference failed: {e}")

    def detect(self, image: Image.Image, min_confidence: float = 0.25,
               labels: Optional[List[str]] = None) -> tuple:
        """Run OBB detection on an image."""
        if not self.is_ready:
            raise RuntimeError("Hailo OBB Detector not initialized")

        start_time = time.perf_counter()

        # Preprocess
        input_data, original_size, scale, pad = self.preprocess(image)

        # Run inference
        raw_outputs = self._run_inference(input_data)

        # Postprocess
        detections = self.postprocess_obb(
            raw_outputs, original_size, scale, pad,
            min_confidence, labels
        )

        processing_ms = int((time.perf_counter() - start_time) * 1000)
        return detections, processing_ms

    def detect_tiled(self, image: Image.Image, min_confidence: float = 0.25,
                     labels: Optional[List[str]] = None, tile_size: int = 640,
                     overlap: float = 0.25, iou_threshold: float = 0.5) -> tuple:
        """Run OBB detection with tiling."""
        if not self.is_ready:
            raise RuntimeError("Hailo OBB Detector not initialized")

        start_time = time.perf_counter()

        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_w, img_h = image.size

        # If small enough, use regular detection
        if img_w <= tile_size and img_h <= tile_size:
            dets, _ = self.detect(image, min_confidence, labels)
            processing_ms = int((time.perf_counter() - start_time) * 1000)
            return dets, processing_ms, 1

        # Tile the image
        stride = int(tile_size * (1 - overlap))
        all_detections = []
        tile_count = 0

        for y in range(0, img_h, stride):
            for x in range(0, img_w, stride):
                x_end = min(x + tile_size, img_w)
                y_end = min(y + tile_size, img_h)
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)

                tile = image.crop((x_start, y_start, x_end, y_end))
                tile_w, tile_h = tile.size

                tile_dets, _ = self.detect(tile, min_confidence, labels)

                # Transform to original coordinates
                for det in tile_dets:
                    bbox = det["bbox"]
                    det["bbox"] = {
                        "x_min": round((x_start + bbox["x_min"] * tile_w) / img_w, 4),
                        "y_min": round((y_start + bbox["y_min"] * tile_h) / img_h, 4),
                        "x_max": round((x_start + bbox["x_max"] * tile_w) / img_w, 4),
                        "y_max": round((y_start + bbox["y_max"] * tile_h) / img_h, 4),
                    }
                    all_detections.append(det)

                tile_count += 1

        # Apply NMS
        merged = nms_detections(all_detections, iou_threshold)

        processing_ms = int((time.perf_counter() - start_time) * 1000)
        return merged, processing_ms, tile_count

    def close(self):
        """Clean up resources."""
        if self._config_ctx:
            try:
                self._config_ctx.__exit__(None, None, None)
            except:
                pass
        self._configured_model = None
        self._infer_model = None
        self._device = None
        self._hef = None


def get_hailo_obb_detector(model_name: str = "yolo11n-obb") -> HailoOBBDetector:
    """Get or create a Hailo OBB detector instance."""
    global _hailo_obb_detector

    # Normalize model name to match HEF filename
    hef_name = model_name.replace("_", "-").replace("-hailo", "")
    model_path = Path(OBB_MODELS_DIR) / f"{hef_name}.hef"

    # Try alternate naming if not found
    if not model_path.exists():
        alt_name = model_name.replace("-", "_")
        model_path = Path(OBB_MODELS_DIR) / f"{alt_name}.hef"

    with _hailo_obb_lock:
        if _hailo_obb_detector is None or _hailo_obb_detector.model_path != str(model_path):
            if _hailo_obb_detector is not None:
                _hailo_obb_detector.close()

            if not model_path.exists():
                # List available OBB HEFs
                available = list(Path(OBB_MODELS_DIR).glob("*obb*.hef"))
                avail_str = ", ".join([p.stem for p in available]) if available else "none"
                raise ValueError(f"Hailo OBB model not found: {model_path}. "
                                f"Available: {avail_str}. "
                                f"Or use CPU model 'yolo11s-obb' instead.")

            _hailo_obb_detector = HailoOBBDetector(str(model_path))
            if not _hailo_obb_detector.initialize():
                raise RuntimeError(f"Failed to initialize Hailo OBB detector: {model_name}")

        return _hailo_obb_detector
