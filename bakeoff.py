"""Head-to-head bake-off of recognition pipelines on the real footage.

Same leave-one-clip-out protocol throughout, so the numbers are comparable. Every route
below runs entirely on the NPU with models already on the box.

  A  baseline            SCRFD -> align -> ArcFace                      (what ships)
  B  + flip TTA          embed the crop and its mirror, average         (free)
  C  + super-resolution  real_esrgan_x2 the face region first           (~x2 detail)
  D  B + C

Route C is the interesting one. Faces on the distant camera are ~67px against ArcFace's
112px design point, and super-resolution is the only lever on the box that addresses
resolution directly. It may also hurt: ESRGAN hallucinates plausible detail, and
plausible-but-wrong detail is exactly what corrupts an identity embedding. Worth
measuring rather than assuming either way.
"""
import time

import cv2
import numpy as np
from PIL import Image

import face
from hailo_platform import HEF, VDevice, FormatType, HailoSchedulingAlgorithm

CLIPS = {
    "clip3": "man", "clip8": "man", "clip9": "man",
    "clip4": "woman", "clip5": "woman", "clip10": "woman",
}
MIN_CONF, MIN_PX, BLUR, STRIDE = 0.3, 1500, 30.0, 5

detector = face.get_face_detector()
embedder = face.get_face_embedder()


class Upscaler(face._HailoModel):
    """real_esrgan_x2: 512x512 -> 1024x1024."""

    def __init__(self):
        super().__init__(face.MODELS_DIR / "real_esrgan_x2.hef", "real_esrgan_x2")

    def upscale(self, image_rgb: np.ndarray) -> np.ndarray:
        raw = self._run_inference(np.ascontiguousarray(image_rgb, dtype=np.uint8))
        out = np.squeeze(next(iter(raw.values())))
        return np.clip(out, 0, 255).astype(np.uint8)


upscaler = Upscaler()
if not upscaler.initialize():
    raise SystemExit("could not init real_esrgan_x2")


def embed_plain(rgb, det, flip=False):
    aligned = face.align_face(rgb, np.array(det["landmarks"], dtype=np.float32))
    v = embedder.embed(aligned)
    if flip:
        v2 = embedder.embed(np.ascontiguousarray(aligned[:, ::-1]))
        v = v + v2
        n = np.linalg.norm(v)
        v = v / n if n else v
    return v


def embed_upscaled(rgb, det, flip=False):
    """Crop the face region, super-resolve it, then align in the upscaled frame."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = det["bbox"]
    bw, bh = x2 - x1, y2 - y1
    # generous context: alignment needs room around the landmarks
    cx1 = int(max(0, x1 - bw * 0.35)); cy1 = int(max(0, y1 - bh * 0.35))
    cx2 = int(min(w, x2 + bw * 0.35)); cy2 = int(min(h, y2 + bh * 0.35))
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return embed_plain(rgb, det, flip)

    crop = rgb[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    s = 512.0 / max(ch, cw)
    nw, nh = max(1, int(cw * s)), max(1, int(ch * s))
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    oy, ox = (512 - nh) // 2, (512 - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(crop, (nw, nh),
                                                interpolation=cv2.INTER_CUBIC)

    big = upscaler.upscale(canvas)            # 1024x1024, 2x the canvas
    if big.shape[0] != 1024:
        big = cv2.resize(big, (1024, 1024), interpolation=cv2.INTER_LINEAR)

    # map original-frame landmarks into the upscaled canvas
    lm = np.array(det["landmarks"], dtype=np.float32).reshape(5, 2)
    lm = (lm - np.array([cx1, cy1], dtype=np.float32)) * s
    lm = (lm + np.array([ox, oy], dtype=np.float32)) * 2.0

    aligned = face.align_face(big, lm)
    v = embedder.embed(aligned)
    if flip:
        v2 = embedder.embed(np.ascontiguousarray(aligned[:, ::-1]))
        v = v + v2
        n = np.linalg.norm(v)
        v = v / n if n else v
    return v


ROUTES = {
    "A baseline":       lambda rgb, d: embed_plain(rgb, d, flip=False),
    "B flip-TTA":       lambda rgb, d: embed_plain(rgb, d, flip=True),
    "C super-res":      lambda rgb, d: embed_upscaled(rgb, d, flip=False),
    "D super-res+flip": lambda rgb, d: embed_upscaled(rgb, d, flip=True),
}

# Collect detections once; every route re-embeds the SAME faces so the comparison is
# clean - any difference is the embedding pipeline, not which faces were found.
print("detecting faces once, shared across all routes...")
detections = {}
for clip in CLIPS:
    cap = cv2.VideoCapture(f"/home/grazzy/media/{clip}.mp4")
    i, found = 0, []
    while True:
        if not cap.grab():
            break
        if i % STRIDE:
            i += 1
            continue
        ok, frame = cap.retrieve()
        i += 1
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for d in detector.detect(Image.fromarray(rgb), min_confidence=MIN_CONF):
            if d["area_px"] < MIN_PX:
                continue
            a = face.align_face(rgb, np.array(d["landmarks"], dtype=np.float32))
            if face.blur_score(a) < BLUR:
                continue
            found.append((rgb, d))
    cap.release()
    detections[clip] = found
    print(f"  {clip:7} {CLIPS[clip]:5} {len(found):3} faces")
print()

people = sorted(set(CLIPS.values()))


def centroid(vs):
    c = np.mean(vs, axis=0)
    n = np.linalg.norm(c)
    return c / n if n else c


print(f"{'route':18} {'top-1':>8} {'mean margin':>12} {'ms/face':>9}")
results = {}
for label, fn in ROUTES.items():
    t0 = time.perf_counter()
    vectors = {c: [fn(rgb, d) for rgb, d in detections[c]] for c in CLIPS}
    n_faces = sum(len(v) for v in vectors.values())
    per_face = (time.perf_counter() - t0) / max(n_faces, 1) * 1000

    hits = total = 0
    margins = []
    for probe in CLIPS:
        if not vectors[probe]:
            continue
        gallery = {}
        for p in people:
            vs = [v for c in CLIPS if CLIPS[c] == p and c != probe for v in vectors[c]]
            if vs:
                gallery[p] = centroid(np.stack(vs))
        if len(gallery) < 2:
            continue
        names = list(gallery)
        matrix = np.stack([gallery[n] for n in names])
        truth = CLIPS[probe]
        for v in vectors[probe]:
            scores = matrix @ v
            order = np.argsort(scores)[::-1]
            margins.append(float(scores[order[0]] - scores[order[1]]))
            total += 1
            if names[int(order[0])] == truth:
                hits += 1
    acc = hits / total if total else 0.0
    results[label] = (acc, np.mean(margins) if margins else 0.0, per_face)
    print(f"{label:18} {acc:7.1%} {np.mean(margins):12.4f} {per_face:9.0f}")

print()
best = max(results, key=lambda k: results[k][0])
print(f"BEST: {best}  ({results[best][0]:.1%}, baseline was {results['A baseline'][0]:.1%})")
print(f"delta vs baseline: {results[best][0] - results['A baseline'][0]:+.1%}")
