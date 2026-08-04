"""Are the 771 faces tiling found in yard_johanna real, or false positives?

Tiling raised that clip from 19 usable faces to 771, and accuracy collapsed:
mobilefacenet 73.4% -> 51.8%, adaface_ir18 94.8% -> 70.3%, with the fold noise floor
roughly tripling.

Two explanations, and they demand opposite actions:

  a) tiling surfaces genuinely harder faces -> the task got harder, the models are fine,
     and the quality floor needs raising
  b) tiling manufactures false positives -> the extra detections are junk polluting the
     gallery, and auto-tiling should be reverted

The earlier detector test sampled 12 frames and looked at BEST quality, which flatters
(a). This looks at the whole distribution and renders the crops, which is the check I
should have run before shipping auto-tiling.
"""
import cv2
import numpy as np
from PIL import Image

import face

CLIP = "/home/grazzy/media/crusty/johanna_271566.mp4"
MIN_CONF, MIN_PX, BLUR, QFLOOR = 0.3, 1500, 30.0, 0.05
STRIDE = 20
TILE, COLS, PAD = 112, 12, 30

detector = face.get_face_detector()

whole, tiled = [], []
cap = cv2.VideoCapture(CLIP)
index = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    index += 1
    if index % STRIDE:
        continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    for label, dets in (("whole", detector.detect(img, MIN_CONF)),
                        ("tiled", detector.detect_tiled(img, MIN_CONF))):
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
            (whole if label == "whole" else tiled).append(
                {"crop": a, "q": q, "t": round(index / 25.0, 1),
                 "px": int(d["area_px"]), "conf": d["confidence"]})
cap.release()

for label, pool in (("whole-frame", whole), ("tiled", tiled)):
    if not pool:
        print(f"{label:12} 0 faces")
        continue
    qs = np.array([e["q"] for e in pool])
    pxs = np.array([e["px"] for e in pool])
    print(f"{label:12} n={len(pool):4}  q: med={np.median(qs):.3f} p90={np.percentile(qs,90):.3f} "
          f"max={qs.max():.3f}  px: med={int(np.median(pxs))}")

# Render a RANDOM sample of tiled crops, not the best ones — the best always look fine.
if tiled:
    rng = np.random.default_rng(0)
    picked = [tiled[i] for i in rng.choice(len(tiled), size=min(36, len(tiled)),
                                           replace=False)]
    rows = (len(picked) + COLS - 1) // COLS
    sheet = np.full((rows * (TILE + PAD), COLS * TILE, 3), 24, dtype=np.uint8)
    for i, e in enumerate(picked):
        r, c = divmod(i, COLS)
        y, x = r * (TILE + PAD), c * TILE
        sheet[y:y + TILE, x:x + TILE] = cv2.cvtColor(e["crop"], cv2.COLOR_RGB2BGR)
        cv2.putText(sheet, f"q{e['q']:.2f} {e['t']}s", (x + 2, y + TILE + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"px{e['px']}", (x + 2, y + TILE + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150, 200, 150), 1, cv2.LINE_AA)
    cv2.imwrite("/home/grazzy/media/tiled_random_sample.png", sheet)
    print(f"\nwrote tiled_random_sample.png ({len(picked)} RANDOM crops, not cherry-picked)")
