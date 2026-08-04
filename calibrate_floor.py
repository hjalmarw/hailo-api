"""Can landmark geometry reject the flower pot?

quality_score cannot: the pot is large, sharp, and SCRFD is confident about it, so it
scores 0.14-0.28 — real-face range.

But quality_score never looks at whether the five landmarks form a plausible FACE. A real
face has strong structure: eyes roughly level and separated, nose between and below them,
mouth corners below the nose, and the whole arrangement roughly symmetric. Whatever SCRFD
emits for a plant pot has no reason to obey that.

This measures several geometric invariants on known-pot detections (tiled crops from
yard_johanna, verified visually as pots) against known-face detections (whole-frame crops
from clips with confirmed faces), to see whether any of them separate cleanly.

If they do, it is a free non-face filter: no extra model, no NPU time, just arithmetic on
landmarks we already have.
"""
import cv2
import numpy as np
from PIL import Image

import face

# yard_johanna tiled = the pot false positives; ring/clip3 whole-frame = real faces
POTS = ("/home/grazzy/media/crusty/johanna_271566.mp4", True)
FACES = [("/home/grazzy/media/crusty/ring_johanna_271570.mp4", False),
         ("/home/grazzy/media/crusty/ring_joanna_265110.mp4", False),
         ("/home/grazzy/media/crusty/ring_joanna2_264994.mp4", False),
         ("/home/grazzy/media/crusty/hjalmar_271514.mp4", False),
         ("/home/grazzy/media/clip3.mp4", False),
         ("/home/grazzy/media/clip5.mp4", False),
         ("/home/grazzy/media/clip8.mp4", False),
         ("/home/grazzy/media/clip9.mp4", False)]

MIN_CONF, MIN_PX, BLUR, QFLOOR = 0.3, 1500, 30.0, 0.05
STRIDE = 4

detector = face.get_face_detector()


def geometry(landmarks):
    """Shape descriptors of the 5-point landmark set.

    left eye, right eye, nose, left mouth, right mouth
    """
    p = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    le, re, nose, lm, rm = p

    eye_d = float(np.linalg.norm(re - le))
    if eye_d < 1e-3:
        return None

    eye_mid = (le + re) / 2.0
    mouth_mid = (lm + rm) / 2.0
    mouth_w = float(np.linalg.norm(rm - lm))

    # vertical axis of the face, eyes -> mouth
    axis = mouth_mid - eye_mid
    axis_len = float(np.linalg.norm(axis))
    if axis_len < 1e-3:
        return None
    axis_u = axis / axis_len

    # how far along eyes->mouth the nose sits (a real face is ~0.4-0.7)
    nose_along = float(np.dot(nose - eye_mid, axis_u)) / axis_len
    # lateral offset of nose from the vertical midline, in eye-widths
    perp_u = np.array([-axis_u[1], axis_u[0]], dtype=np.float32)
    nose_off = abs(float(np.dot(nose - eye_mid, perp_u))) / eye_d
    # eyes should be roughly perpendicular to the eyes->mouth axis
    eye_axis = (re - le) / eye_d
    orthogonality = abs(float(np.dot(eye_axis, axis_u)))
    # proportions
    mouth_eye_ratio = mouth_w / eye_d
    aspect = axis_len / eye_d

    return {"nose_along": nose_along, "nose_off": nose_off,
            "orthogonality": orthogonality, "mouth_eye": mouth_eye_ratio,
            "aspect": aspect}


def harvest(path, tiled, limit=400):
    cap = cv2.VideoCapture(path)
    index, out = 0, []
    while len(out) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        if index % STRIDE:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        dets = (detector.detect_tiled(img, MIN_CONF) if tiled
                else detector.detect(img, MIN_CONF))
        for d in dets:
            if d["area_px"] < MIN_PX:
                continue
            a = face.align_face(rgb, np.array(d["landmarks"], dtype=np.float32))
            s = face.blur_score(a)
            if s < BLUR:
                continue
            q = face.quality_score(d["area_px"], s, d["confidence"],
                                   face.frontality(d["landmarks"]))
            if q < QFLOOR:
                continue
            g = geometry(d["landmarks"])
            if g:
                g["q"] = q
                out.append(g)
    cap.release()
    return out


print("harvesting known pots (tiled yard_johanna) ...")
pots = harvest(*POTS)
print(f"  {len(pots)} pot detections")

print("harvesting known faces (whole-frame, verified clips) ...")
faces = []
for path, tiled in FACES:
    faces.extend(harvest(path, tiled, limit=150))
print(f"  {len(faces)} face detections")

if not pots or not faces:
    raise SystemExit("need both populations")

print()
print(f"{'metric':16} {'FACES med [p10-p90]':>30} {'POTS med [p10-p90]':>30}  separable?")
for key in ("nose_along", "nose_off", "orthogonality", "mouth_eye", "aspect", "q"):
    f = np.array([e[key] for e in faces])
    p = np.array([e[key] for e in pots])
    f_lo, f_hi = np.percentile(f, 10), np.percentile(f, 90)
    p_lo, p_hi = np.percentile(p, 10), np.percentile(p, 90)
    overlap = not (f_hi < p_lo or p_hi < f_lo)
    print(f"{key:16} {np.median(f):8.3f} [{f_lo:6.3f}-{f_hi:6.3f}] "
          f"{np.median(p):11.3f} [{p_lo:6.3f}-{p_hi:6.3f}]   "
          f"{'overlap' if overlap else 'CLEAN SPLIT'}")

# ---- the number we actually need: the quality floor ----
fq = np.array([e["q"] for e in faces])
pq = np.array([e["q"] for e in pots])
print()
print(f"quality floor sweep   (faces n={len(fq)}, pots n={len(pq)})")
print(f"{'floor':>7} {'faces kept':>12} {'pots kept':>11}  {'pot share of survivors':>24}")
for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.35, 0.40):
    kf, kp = (fq >= t).sum(), (pq >= t).sum()
    share = kp / (kf + kp) if (kf + kp) else 0.0
    print(f"{t:7.2f} {kf/len(fq):11.1%} {kp/len(pq):11.1%}  {share:23.1%}")

# Best single-threshold rule per metric
print("\nbest single-threshold rule per metric (maximising pot rejection at >=95% face keep):")
for key in ("nose_along", "nose_off", "orthogonality", "mouth_eye", "aspect"):
    f = np.array([e[key] for e in faces])
    p = np.array([e[key] for e in pots])
    best = None
    for direction in ("<=", ">="):
        for t in np.percentile(np.concatenate([f, p]), np.arange(1, 100)):
            keep_f = (f <= t).mean() if direction == "<=" else (f >= t).mean()
            keep_p = (p <= t).mean() if direction == "<=" else (p >= t).mean()
            if keep_f >= 0.95 and (best is None or keep_p < best[2]):
                best = (direction, t, keep_p)
    if best:
        print(f"  {key:16} keep faces if {best[0]} {best[1]:7.3f}  "
              f"-> pots surviving: {best[2]:5.1%}")
    else:
        print(f"  {key:16} no threshold keeps 95% of faces")
