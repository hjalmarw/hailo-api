"""Whole-person appearance embeddings on the NPU, via CLIP.

Faces are the exception in real camera footage, not the rule — people mostly pass
through at an angle, looking down, too far away. This gives a second channel that works
whenever a *person* is visible, with no face required.

CLIP ResNet-50x4's image encoder is already compiled for the Hailo-10H (288x288 -> 640-d,
~19ms). It is not a purpose-built person re-identification model — Hailo's repvgg ReID
HEFs exist only for Hailo-8 and would need recompiling on x86 — but CLIP embeddings are
strong general appearance descriptors, and being already on the right silicon beats
being theoretically better.

MEASURED RESULT: this does NOT work as a re-identification channel.

Leave-one-clip-out over six real clips and two people gave 30.3% top-1 - worse than the
50% coin-flip baseline, against 76.6% for the face channel on the same protocol. Margins
between candidates were 0.013-0.048, i.e. essentially nothing. CLIP is trained to align
images with text, so it encodes "a person outdoors in a garden" rather than *which*
person; it captures category, not instance. Abstention could not rescue it either -
gating on margin >= 0.05 still only reached 50% precision, exactly chance.

Keep this module for what CLIP is actually good at - open-vocabulary description of a
crop ("person in a dark jacket carrying a bag"), which is useful for alert text and
search even though it cannot assign identity. For real person re-identification, the
right model is repvgg_a0_person_reid or osnet, and those ship compiled for Hailo-8 only;
Hailo-10H would need a recompile from ONNX with the Dataflow Compiler on x86.

What this channel is and isn't:

  - It encodes *appearance*: clothing, build, hair, what someone is carrying.
  - So it is identity-stable within a session and across cameras on the same day, and
    unreliable once someone changes clothes. Treat it as "same person as the one the
    back camera saw ten minutes ago", never as a durable identity.
  - Face remains the identity anchor. Appearance links; faces name.
"""

import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from face import _HailoModel, MODELS_DIR

CLIP_PATH = MODELS_DIR / "clip_resnet_50x4_image_encoder.hef"

# Person boxes are cropped with a margin: detectors clip tightly to the body, and the
# surrounding context (bag straps, a hood, stance) carries real appearance signal.
CROP_MARGIN = 0.08


class ClipEncoder(_HailoModel):
    """CLIP ResNet-50x4 image encoder: 288x288 RGB in, unit-norm 640-d embedding out."""

    def __init__(self):
        super().__init__(CLIP_PATH, "clip_resnet_50x4_image_encoder")

    def embed(self, image_rgb: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            raise RuntimeError("CLIP encoder not initialized")

        if image_rgb.shape[:2] != (288, 288):
            image_rgb = cv2.resize(image_rgb, (288, 288), interpolation=cv2.INTER_AREA)

        raw = self._run_inference(np.ascontiguousarray(image_rgb, dtype=np.uint8))
        vector = np.squeeze(next(iter(raw.values()))).astype(np.float32).reshape(-1)

        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector


def crop_person(image_rgb: np.ndarray, bbox: List[float],
                margin: float = CROP_MARGIN) -> np.ndarray:
    """Crop a person box with margin, letterboxed to square.

    Letterboxing rather than stretching matters here: a person box is tall and narrow,
    and squashing it to 288x288 would distort build and proportions — exactly the cues
    this channel depends on.
    """
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w, box_h = x2 - x1, y2 - y1

    x1 = int(max(0, x1 - box_w * margin))
    y1 = int(max(0, y1 - box_h * margin))
    x2 = int(min(width, x2 + box_w * margin))
    y2 = int(min(height, y2 + box_h * margin))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((288, 288, 3), dtype=np.uint8)

    crop = image_rgb[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]

    scale = 288.0 / max(crop_h, crop_w)
    new_w, new_h = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((288, 288, 3), dtype=np.uint8)
    off_y, off_x = (288 - new_h) // 2, (288 - new_w) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas


_encoder: Optional[ClipEncoder] = None
_encoder_lock = threading.Lock()


def get_clip_encoder() -> ClipEncoder:
    global _encoder
    with _encoder_lock:
        if _encoder is None:
            encoder = ClipEncoder()
            if not encoder.initialize():
                raise RuntimeError("Failed to initialize CLIP encoder")
            _encoder = encoder
    return _encoder


def close_clip_encoder():
    global _encoder
    with _encoder_lock:
        if _encoder is not None:
            _encoder.close()
            _encoder = None


def embed_people(image_rgb: np.ndarray, boxes: List[List[float]]) -> List[np.ndarray]:
    """Appearance embedding for each person box in one frame."""
    encoder = get_clip_encoder()
    return [encoder.embed(crop_person(image_rgb, box)) for box in boxes]
