"""Find the faces SCRFD is missing.

grazzy's correction: every event he supplied contains people. #255698 yielding 283
object false-positives and no real face is a DETECTOR failure, not an absence.

Prime suspect: SCRFD ingests a 640x640 letterbox. On 3840x2160 footage that is a 6x
downscale, so a 60px face in the original arrives as ~10px and disappears. Everything
measured today rests on detection, so if this is the bottleneck it matters more than any
embedder comparison.

Three strategies on identical frames:

  A whole-frame   SCRFD on the letterboxed full frame              (what ships)
  B person-first  YOLO finds people -> crop each -> SCRFD on the crop at native
                  resolution. The crop is far smaller than the frame, so the face
                  survives the resize to 640.
  C tiled         SCRFD over overlapping tiles of the full frame, no downscale

B is the interesting one: person detection is robust at small scale in a way face
detection is not, so it can act as a zoom oracle for the face detector.
"""
import os

import cv2
import numpy as np
from PIL import Image

import face
from detector import get_detector

CLIPS = {
    "joanna_255698": "/home/grazzy/media/crusty/joanna_255698.mp4",
    "ring_joanna_265110": "/home/grazzy/media/crusty/ring_joanna_265110.mp4",
    "yard_hjalmar": "/home/grazzy/media/crusty/hjalmar_271514.mp4",
}
STRIDE = 15
MIN_CONF = 0.25

face_detector = face.get_face_detector()
person_detector = get_detector("yolov8m")


def whole_frame(rgb):
    return face_detector.detect(Image.fromarray(rgb), min_confidence=MIN_CONF)


def person_first(rgb):
    """YOLO -> person crop -> SCRFD on the crop, boxes mapped back to frame coords."""
    height, width = rgb.shape[:2]
    people, _ = person_detector.detect(Image.fromarray(rgb), min_confidence=0.35,
                                       labels=["person"])
    found = []
    for person in people:
        box = person["bbox"]
        x1 = int(max(0, box["x_min"] * width))
        y1 = int(max(0, box["y_min"] * height))
        x2 = int(min(width, box["x_max"] * width))
        y2 = int(min(height, box["y_max"] * height))
        if x2 - x1 < 32 or y2 - y1 < 32:
            continue
        crop = rgb[y1:y2, x1:x2]
        for d in face_detector.detect(Image.fromarray(crop), min_confidence=MIN_CONF):
            b = d["bbox"]
            found.append({
                "bbox": [b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1],
                "confidence": d["confidence"],
                "landmarks": [[p[0] + x1, p[1] + y1] for p in d["landmarks"]],
                "area_px": d["area_px"],
            })
    return found


def tiled(rgb, tile=960, overlap=0.2):
    height, width = rgb.shape[:2]
    step = int(tile * (1 - overlap))
    found = []
    for y in range(0, max(1, height - 1), step):
        for x in range(0, max(1, width - 1), step):
            patch = rgb[y:min(y + tile, height), x:min(x + tile, width)]
            if patch.shape[0] < 64 or patch.shape[1] < 64:
                continue
            for d in face_detector.detect(Image.fromarray(patch),
                                          min_confidence=MIN_CONF):
                b = d["bbox"]
                found.append({
                    "bbox": [b[0] + x, b[1] + y, b[2] + x, b[3] + y],
                    "confidence": d["confidence"],
                    "landmarks": [[p[0] + x, p[1] + y] for p in d["landmarks"]],
                    "area_px": d["area_px"],
                })
    return found


STRATEGIES = {"A whole-frame": whole_frame,
              "B person-first": person_first,
              "C tiled": tiled}

for label, path in CLIPS.items():
    if not os.path.exists(path):
        print(f"{label}: missing")
        continue
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    index = 0
    while len(frames) < 12:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        if index % STRIDE:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    print(f"\n=== {label}  {width}x{height}  {len(frames)} frames sampled ===")
    for name, fn in STRATEGIES.items():
        good = 0
        qualities = []
        for rgb in frames:
            for d in fn(rgb):
                if d["area_px"] < 1500:
                    continue
                aligned = face.align_face(
                    rgb, np.array(d["landmarks"], dtype=np.float32))
                sharp = face.blur_score(aligned)
                q = face.quality_score(d["area_px"], sharp, d["confidence"],
                                       face.frontality(d["landmarks"]))
                qualities.append(q)
                if q >= 0.05:
                    good += 1
        if qualities:
            print(f"  {name:16} {len(qualities):4} dets, {good:3} above q0.05, "
                  f"best q={max(qualities):.3f}")
        else:
            print(f"  {name:16}    0 dets")
