"""Face detection and recognition on the Hailo-10H NPU.

Two-stage pipeline:
  1. SCRFD-10G detects faces and 5 facial landmarks (640x640 input)
  2. Faces are aligned to a canonical 112x112 crop via similarity transform
  3. ArcFace/MobileFaceNet produces a 512-d embedding per face
  4. Embeddings are matched against an enrolled gallery by cosine similarity

The gallery is brute-force numpy rather than a vector DB: at the scale this runs at
(hundreds of people, a handful of embeddings each) a 512-d dot product over the whole
gallery is sub-millisecond and exact, so an index would only add a dependency and an
approximation.

Both models follow the same persistent-load pattern as HailoDetector: SHARED group_id
with ROUND_ROBIN scheduling so they coexist with the YOLO detectors on one device.
"""

import json
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from hailo_platform import HEF, VDevice, FormatType, HailoSchedulingAlgorithm

MODELS_DIR = Path("/usr/local/hailo/resources/models/hailo10h")
SCRFD_PATH = MODELS_DIR / "scrfd_10g.hef"
ARCFACE_PATH = MODELS_DIR / "arcface_mobilefacenet.hef"

GALLERY_DIR = Path(__file__).parent / "face_gallery"

# SCRFD-10G emits three feature levels, two anchors per cell.
SCRFD_STRIDES = (8, 16, 32)
SCRFD_ANCHORS_PER_CELL = 2

# Canonical 5-point landmark template ArcFace was trained against, in 112x112 space.
# Aligning to this is what makes embeddings comparable across pose and scale.
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)

# Quality gates, mirroring the tolerances Hailo ships in face_recon_algo_params.json.
DEFAULT_MIN_FACE_PIXELS = 12000     # ~110x110; below this recognition degrades sharply
DEFAULT_BLUR_TOLERANCE = 150.0      # variance of Laplacian


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4) -> List[int]:
    """Standard greedy NMS. Returns indices of kept boxes, highest score first."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        order = rest[iou <= iou_threshold]

    return keep


def align_face(image_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Warp a face to the canonical 112x112 ArcFace crop using its 5 landmarks.

    Uses a partial affine (similarity: rotation + uniform scale + translation) rather
    than a full affine, which would let shear distort facial geometry and corrupt the
    embedding. Falls back to a plain resize of the landmark bounding region if the
    transform cannot be estimated.
    """
    src = landmarks.astype(np.float32).reshape(5, 2)
    matrix, _ = cv2.estimateAffinePartial2D(
        src, ARCFACE_TEMPLATE, method=cv2.LMEDS
    )

    if matrix is None:
        x_min, y_min = np.floor(src.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(src.max(axis=0)).astype(int)
        h, w = image_rgb.shape[:2]
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, max(x_max, x_min + 1)), min(h, max(y_max, y_min + 1))
        crop = image_rgb[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)

    return cv2.warpAffine(
        image_rgb, matrix, (112, 112), borderValue=0.0, flags=cv2.INTER_LINEAR
    )


def blur_score(face_bgr_or_rgb: np.ndarray) -> float:
    """Variance of the Laplacian. Higher is sharper; motion blur drives this toward 0."""
    gray = cv2.cvtColor(face_bgr_or_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class _HailoModel:
    """Shared persistent-load plumbing for the two face models."""

    def __init__(self, model_path: Path, name: str):
        self.model_path = str(model_path)
        self.model_name = name
        self.start_time = time.time()

        self._inference_lock = threading.Lock()
        self._device = None
        self._infer_model = None
        self._config_ctx = None
        self._configured_model = None
        self._hef = None
        self._input_shape = None

    def initialize(self) -> bool:
        if not Path(self.model_path).exists():
            print(f"HEF not found: {self.model_path}")
            return False
        try:
            self._hef = HEF(self.model_path)
            self._input_shape = self._hef.get_input_vstream_infos()[0].shape

            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            params.group_id = "SHARED"
            self._device = VDevice(params)

            self._infer_model = self._device.create_infer_model(self.model_path)
            self._infer_model.set_batch_size(1)
            for output in self._infer_model.outputs:
                output.set_format_type(FormatType.FLOAT32)

            self._config_ctx = self._infer_model.configure()
            self._configured_model = self._config_ctx.__enter__()

            print(f"{self.model_name} initialized, input {self._input_shape}")
            return True
        except Exception as e:
            print(f"Failed to initialize {self.model_name}: {e}")
            self.close()
            return False

    @property
    def is_ready(self) -> bool:
        return self._configured_model is not None

    @property
    def input_size(self) -> Tuple[int, int]:
        if self._input_shape:
            return (self._input_shape[0], self._input_shape[1])
        return (640, 640)

    @property
    def uptime_seconds(self) -> int:
        return int(time.time() - self.start_time)

    def _run_inference(self, input_data: np.ndarray) -> dict:
        with self._inference_lock:
            try:
                output_buffers = {}
                for output_info in self._hef.get_output_vstream_infos():
                    output_buffers[output_info.name] = np.empty(
                        self._infer_model.output(output_info.name).shape,
                        dtype=np.float32
                    )

                bindings = self._configured_model.create_bindings(
                    output_buffers=output_buffers
                )
                bindings.input().set_buffer(input_data)
                self._configured_model.run([bindings], timeout=10000)

                return {name: buf.copy() for name, buf in output_buffers.items()}
            except Exception as e:
                if "74" in str(e) or "OUT_OF_PHYSICAL_DEVICES" in str(e):
                    raise RuntimeError("Hailo device busy")
                raise RuntimeError(f"{self.model_name} inference failed: {e}")

    def close(self):
        try:
            if self._config_ctx is not None and self._configured_model is not None:
                self._config_ctx.__exit__(None, None, None)
        except Exception:
            pass
        self._config_ctx = None
        self._configured_model = None
        self._infer_model = None
        self._device = None
        self._hef = None


class FaceDetector(_HailoModel):
    """SCRFD-10G face detector producing boxes plus 5 landmarks."""

    def __init__(self):
        super().__init__(SCRFD_PATH, "scrfd_10g")

    def _preprocess(self, image: Image.Image) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox to 640x640, preserving aspect ratio."""
        target_h, target_w = self.input_size
        orig_w, orig_h = image.size

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        resized = image.resize((new_w, new_h), Image.BILINEAR)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        padded.paste(resized, (pad_w, pad_h))

        return np.array(padded, dtype=np.uint8), scale, (pad_w, pad_h)

    def _group_outputs(self, raw: dict) -> List[dict]:
        """Bucket the 9 output tensors into three stride levels.

        Outputs are keyed by opaque conv names, so they're identified by channel count
        (2 = score, 8 = bbox distances, 20 = landmark offsets) and grid size rather
        than by hardcoded layer names, which differ between HEF builds.
        """
        levels = {}
        for name, buf in raw.items():
            arr = np.squeeze(buf)
            if arr.ndim != 3:
                continue
            grid_h, _, channels = arr.shape
            level = levels.setdefault(grid_h, {})
            if channels == SCRFD_ANCHORS_PER_CELL:
                level["score"] = arr
            elif channels == SCRFD_ANCHORS_PER_CELL * 4:
                level["bbox"] = arr
            elif channels == SCRFD_ANCHORS_PER_CELL * 10:
                level["kps"] = arr

        ordered = []
        for grid_h in sorted(levels.keys(), reverse=True):  # 80, 40, 20
            level = levels[grid_h]
            if {"score", "bbox", "kps"} <= level.keys():
                level["grid"] = grid_h
                ordered.append(level)
        return ordered

    def _decode(self, raw: dict, min_confidence: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode SCRFD outputs into boxes, scores and landmarks in 640x640 space."""
        levels = self._group_outputs(raw)
        input_h, _ = self.input_size

        all_boxes, all_scores, all_kps = [], [], []

        for level in levels:
            grid = level["grid"]
            stride = input_h // grid
            if stride not in SCRFD_STRIDES:
                continue

            scores = level["score"].reshape(-1)
            # Hailo may or may not fold the final sigmoid into the HEF; detect and apply.
            if scores.min() < 0.0 or scores.max() > 1.0:
                scores = _sigmoid(scores)

            keep = scores >= min_confidence
            if not np.any(keep):
                continue

            # Anchor centres: one per (row, col), repeated per anchor.
            ys, xs = np.mgrid[0:grid, 0:grid]
            centres = np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(np.float32) * stride
            centres = np.repeat(centres, SCRFD_ANCHORS_PER_CELL, axis=0)

            bbox = level["bbox"].reshape(-1, 4) * stride
            kps = level["kps"].reshape(-1, 5, 2) * stride

            cx, cy = centres[:, 0], centres[:, 1]
            boxes = np.stack([
                cx - bbox[:, 0], cy - bbox[:, 1],
                cx + bbox[:, 2], cy + bbox[:, 3],
            ], axis=-1)

            points = kps + centres[:, None, :]

            all_boxes.append(boxes[keep])
            all_scores.append(scores[keep])
            all_kps.append(points[keep])

        if not all_boxes:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0, 5, 2))

        return (
            np.concatenate(all_boxes, axis=0),
            np.concatenate(all_scores, axis=0),
            np.concatenate(all_kps, axis=0),
        )

    def detect(self, image: Image.Image, min_confidence: float = 0.5,
               iou_threshold: float = 0.4) -> List[dict]:
        """Detect faces. Returns boxes and landmarks in the ORIGINAL image's pixels."""
        if not self.is_ready:
            raise RuntimeError("Face detector not initialized")

        if image.mode != "RGB":
            image = image.convert("RGB")

        input_data, scale, (pad_w, pad_h) = self._preprocess(image)
        raw = self._run_inference(input_data)
        boxes, scores, kps = self._decode(raw, min_confidence)

        if len(boxes) == 0:
            return []

        keep = _nms(boxes, scores, iou_threshold)
        orig_w, orig_h = image.size

        faces = []
        for i in keep:
            box = (boxes[i] - np.array([pad_w, pad_h, pad_w, pad_h])) / scale
            points = (kps[i] - np.array([pad_w, pad_h])) / scale

            x_min = float(np.clip(box[0], 0, orig_w))
            y_min = float(np.clip(box[1], 0, orig_h))
            x_max = float(np.clip(box[2], 0, orig_w))
            y_max = float(np.clip(box[3], 0, orig_h))
            if x_max <= x_min or y_max <= y_min:
                continue

            faces.append({
                "bbox": [x_min, y_min, x_max, y_max],
                "confidence": float(scores[i]),
                "landmarks": points.tolist(),
                "area_px": (x_max - x_min) * (y_max - y_min),
            })

        return faces


class FaceEmbedder(_HailoModel):
    """ArcFace/MobileFaceNet producing L2-normalised 512-d embeddings."""

    def __init__(self):
        super().__init__(ARCFACE_PATH, "arcface_mobilefacenet")

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        """Embed one aligned 112x112 RGB face. Returns a unit-norm 512-d vector."""
        if not self.is_ready:
            raise RuntimeError("Face embedder not initialized")

        if aligned_face.shape[:2] != (112, 112):
            aligned_face = cv2.resize(aligned_face, (112, 112),
                                      interpolation=cv2.INTER_LINEAR)

        raw = self._run_inference(np.ascontiguousarray(aligned_face, dtype=np.uint8))
        vector = np.squeeze(next(iter(raw.values()))).astype(np.float32).reshape(-1)

        norm = np.linalg.norm(vector)
        # Cosine similarity is only meaningful on unit vectors; a degenerate all-zero
        # output would otherwise divide by zero and poison the gallery.
        return vector / norm if norm > 0 else vector


class FaceGallery:
    """Persistent set of enrolled people and their face embeddings.

    Stored as an .npz of stacked vectors plus a JSON sidecar of names, so the gallery
    survives restarts and can be inspected or hand-edited without special tooling.
    """

    def __init__(self, directory: Path = GALLERY_DIR):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._vectors_path = self.dir / "embeddings.npz"
        self._meta_path = self.dir / "persons.json"

        self._lock = threading.Lock()
        self._names: List[str] = []
        self._vectors = np.empty((0, 512), dtype=np.float32)
        self._meta: List[dict] = []
        self._load()

    def _load(self):
        try:
            if self._vectors_path.exists() and self._meta_path.exists():
                data = np.load(self._vectors_path)
                self._vectors = data["vectors"].astype(np.float32)
                payload = json.loads(self._meta_path.read_text())
                self._names = payload["names"]
                self._meta = payload.get("meta", [{} for _ in self._names])
                if len(self._names) != len(self._vectors):
                    print("Face gallery out of sync; starting empty")
                    self._reset()
        except Exception as e:
            print(f"Failed to load face gallery: {e}")
            self._reset()

    def _reset(self):
        self._names = []
        self._meta = []
        self._vectors = np.empty((0, 512), dtype=np.float32)

    def _save(self):
        np.savez_compressed(self._vectors_path, vectors=self._vectors)
        self._meta_path.write_text(json.dumps(
            {"names": self._names, "meta": self._meta}, indent=2
        ))

    def add(self, name: str, vector: np.ndarray, source: Optional[str] = None):
        with self._lock:
            self._names.append(name)
            self._meta.append({"source": source, "added_at": time.time()})
            self._vectors = np.vstack([self._vectors, vector.reshape(1, -1)])
            self._save()

    def identify(self, vector: np.ndarray, threshold: float = 0.5) -> Optional[dict]:
        """Return the closest enrolled person above `threshold`, else None.

        Both sides are unit-norm, so the dot product is cosine similarity directly.
        """
        with self._lock:
            if len(self._vectors) == 0:
                return None
            similarities = self._vectors @ vector.reshape(-1)
            best = int(np.argmax(similarities))
            score = float(similarities[best])

        if score < threshold:
            return None
        return {"name": self._names[best], "similarity": score}

    def persons(self) -> List[dict]:
        with self._lock:
            counts = {}
            for name in self._names:
                counts[name] = counts.get(name, 0) + 1
            return [{"name": n, "embeddings": c} for n, c in sorted(counts.items())]

    def delete(self, name: str) -> int:
        """Remove every embedding for a person. Returns how many were removed."""
        with self._lock:
            keep = [i for i, n in enumerate(self._names) if n != name]
            removed = len(self._names) - len(keep)
            if removed:
                self._names = [self._names[i] for i in keep]
                self._meta = [self._meta[i] for i in keep]
                self._vectors = (self._vectors[keep] if keep
                                 else np.empty((0, 512), dtype=np.float32))
                self._save()
            return removed


# Singletons. The NPU serialises inference anyway, so one instance of each is correct.
_detector: Optional[FaceDetector] = None
_embedder: Optional[FaceEmbedder] = None
_gallery: Optional[FaceGallery] = None
_init_lock = threading.Lock()


def get_face_detector() -> FaceDetector:
    global _detector
    with _init_lock:
        if _detector is None:
            detector = FaceDetector()
            if not detector.initialize():
                raise RuntimeError("Failed to initialize SCRFD face detector")
            _detector = detector
    return _detector


def get_face_embedder() -> FaceEmbedder:
    global _embedder
    with _init_lock:
        if _embedder is None:
            embedder = FaceEmbedder()
            if not embedder.initialize():
                raise RuntimeError("Failed to initialize ArcFace embedder")
            _embedder = embedder
    return _embedder


def get_gallery() -> FaceGallery:
    global _gallery
    with _init_lock:
        if _gallery is None:
            _gallery = FaceGallery()
    return _gallery


def close_face_models():
    global _detector, _embedder
    with _init_lock:
        if _detector is not None:
            _detector.close()
            _detector = None
        if _embedder is not None:
            _embedder.close()
            _embedder = None


def analyze_image(image: Image.Image, min_confidence: float = 0.5,
                  match_threshold: float = 0.5,
                  min_face_pixels: float = DEFAULT_MIN_FACE_PIXELS,
                  blur_tolerance: float = DEFAULT_BLUR_TOLERANCE,
                  identify: bool = True) -> Tuple[List[dict], dict]:
    """Detect, align, embed and optionally identify every face in one image.

    Returns (faces, stats). Faces failing a quality gate are still returned with their
    box and a `skipped` reason, so callers can see *why* someone went unrecognised
    rather than silently getting fewer results.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    detector = get_face_detector()
    detections = detector.detect(image, min_confidence=min_confidence)

    image_rgb = np.array(image)
    embedder = get_face_embedder() if identify else None
    gallery = get_gallery() if identify else None

    results = []
    stats = {"detected": len(detections), "recognized": 0, "too_small": 0, "too_blurry": 0}

    for face in detections:
        entry = {
            "bbox": face["bbox"],
            "detection_confidence": face["confidence"],
            "landmarks": face["landmarks"],
        }

        if face["area_px"] < min_face_pixels:
            entry["skipped"] = "face_too_small"
            stats["too_small"] += 1
            results.append(entry)
            continue

        aligned = align_face(image_rgb, np.array(face["landmarks"], dtype=np.float32))

        sharpness = blur_score(aligned)
        entry["sharpness"] = round(sharpness, 1)
        if sharpness < blur_tolerance:
            entry["skipped"] = "too_blurry"
            stats["too_blurry"] += 1
            results.append(entry)
            continue

        if not identify:
            results.append(entry)
            continue

        vector = embedder.embed(aligned)
        match = gallery.identify(vector, threshold=match_threshold)
        if match:
            entry["name"] = match["name"]
            entry["similarity"] = round(match["similarity"], 4)
            stats["recognized"] += 1
        else:
            entry["name"] = None
        results.append(entry)

    return results, stats


def enroll_from_image(name: str, image: Image.Image, min_confidence: float = 0.5,
                      source: Optional[str] = None) -> dict:
    """Enroll the single largest face in an image under `name`.

    Deliberately refuses images containing more than one face: silently picking one
    is how a gallery ends up with the wrong person's embedding attached to a name.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    detector = get_face_detector()
    faces = detector.detect(image, min_confidence=min_confidence)

    if not faces:
        raise ValueError("No face found in the image")
    if len(faces) > 1:
        raise ValueError(
            f"Found {len(faces)} faces; enrollment needs exactly one. "
            "Crop the image to the intended person."
        )

    face = faces[0]
    image_rgb = np.array(image)
    aligned = align_face(image_rgb, np.array(face["landmarks"], dtype=np.float32))
    sharpness = blur_score(aligned)

    vector = get_face_embedder().embed(aligned)
    get_gallery().add(name, vector, source=source)

    return {
        "name": name,
        "detection_confidence": face["confidence"],
        "sharpness": round(sharpness, 1),
        "face_pixels": int(face["area_px"]),
    }


def analyze_video(video_path: str, sample_every: int = 15, max_frames: int = 600,
                  min_confidence: float = 0.5, match_threshold: float = 0.5,
                  min_face_pixels: float = DEFAULT_MIN_FACE_PIXELS,
                  blur_tolerance: float = DEFAULT_BLUR_TOLERANCE) -> dict:
    """Identify people across a video by sampling every Nth frame.

    Rather than returning per-frame noise, sightings are aggregated per person: the
    best similarity seen, how many frames they appeared in, and the timestamp window.
    That is what "who was in this footage" actually means.

    Unknown faces that pass the quality gates are counted separately so an intruder
    doesn't vanish just because they aren't enrolled.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    people: dict = {}
    unknown_sightings = 0
    frames_processed = 0
    frames_with_faces = 0
    frame_index = 0

    start = time.perf_counter()
    try:
        while frames_processed < max_frames:
            ok = capture.grab()
            if not ok:
                break

            if frame_index % sample_every != 0:
                frame_index += 1
                continue

            ok, frame_bgr = capture.retrieve()
            frame_index += 1
            if not ok:
                break

            frames_processed += 1
            timestamp = (frame_index - 1) / fps if fps else 0.0

            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            faces, _ = analyze_image(
                image,
                min_confidence=min_confidence,
                match_threshold=match_threshold,
                min_face_pixels=min_face_pixels,
                blur_tolerance=blur_tolerance,
                identify=True,
            )

            if faces:
                frames_with_faces += 1

            for face in faces:
                if face.get("skipped"):
                    continue
                name = face.get("name")
                if name is None:
                    unknown_sightings += 1
                    continue

                record = people.setdefault(name, {
                    "name": name,
                    "sightings": 0,
                    "best_similarity": 0.0,
                    "first_seen_s": timestamp,
                    "last_seen_s": timestamp,
                })
                record["sightings"] += 1
                record["best_similarity"] = max(
                    record["best_similarity"], face.get("similarity", 0.0)
                )
                record["last_seen_s"] = timestamp
    finally:
        capture.release()

    for record in people.values():
        record["first_seen_s"] = round(record["first_seen_s"], 2)
        record["last_seen_s"] = round(record["last_seen_s"], 2)
        record["best_similarity"] = round(record["best_similarity"], 4)

    return {
        "people": sorted(people.values(), key=lambda p: -p["sightings"]),
        "unknown_face_sightings": unknown_sightings,
        "frames_sampled": frames_processed,
        "frames_with_faces": frames_with_faces,
        "video_frames": total_frames,
        "fps": round(fps, 2),
        "duration_s": round(total_frames / fps, 2) if fps and total_frames else None,
        "processing_ms": int((time.perf_counter() - start) * 1000),
        "truncated": frames_processed >= max_frames,
    }
