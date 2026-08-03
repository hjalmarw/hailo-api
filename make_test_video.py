"""Synthesise a video shaped like real security footage.

The bundled Hailo demo is a dense montage of faces with constant motion — the opposite
of a driveway camera, which is mostly an empty static scene with occasional brief
appearances. Benchmarking frame selection on the montage measures the wrong thing.

Produces 90s at 15fps (1350 frames): a static background, with two short appearances
where an enrolled person's face is composited in and drifts across the frame.
"""
import cv2
import numpy as np
from pathlib import Path

FPS = 15
DURATION_S = 90
W, H = 1280, 720
OUT = Path("/home/grazzy/media/synthetic_driveway.mp4")
FACES = Path("/home/grazzy/hailo-apps/local_resources/faces")

# Appearances: (start_s, end_s, person, source image)
APPEARANCES = [
    (20.0, 26.0, "Anna", FACES / "Anna" / "2.jpeg"),
    (58.0, 63.0, "Alice", FACES / "Alice" / "3.jpeg"),
]

OUT.parent.mkdir(parents=True, exist_ok=True)

# A static, slightly textured background — flat grey would make the motion gate look
# better than it deserves, since real sensors always carry a little noise.
rng = np.random.default_rng(7)
background = np.full((H, W, 3), 60, dtype=np.uint8)
background[H // 2:, :] = 82                                   # "ground"
background += rng.integers(0, 6, (H, W, 3), dtype=np.uint8)   # sensor texture

# Pre-extract each person's face region so the composite is a real face, not a rectangle.
crops = {}
for _, _, name, path in APPEARANCES:
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"missing source image: {path}")
    crops[name] = image

writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
if not writer.isOpened():
    raise SystemExit("could not open VideoWriter")

total = DURATION_S * FPS
for index in range(total):
    t = index / FPS
    frame = background.copy()
    # Faint per-frame noise so even "empty" frames are not bit-identical.
    frame = cv2.add(frame, rng.integers(0, 3, (H, W, 3), dtype=np.uint8))

    for start, end, name, _ in APPEARANCES:
        if not (start <= t < end):
            continue
        crop = crops[name]
        scale = 0.55
        small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ch, cw = small.shape[:2]

        # Walk left-to-right across the scene over the appearance window.
        progress = (t - start) / (end - start)
        x = int(80 + progress * (W - cw - 160))
        y = int(H * 0.18 + 30 * np.sin(progress * 6))
        x, y = max(0, min(x, W - cw)), max(0, min(y, H - ch))
        frame[y:y + ch, x:x + cw] = small

    writer.write(frame)

writer.release()
print(f"wrote {OUT} — {total} frames, {DURATION_S}s @ {FPS}fps")
print(f"appearances: {[(s, e, n) for s, e, n, _ in APPEARANCES]}")
print(f"frames containing a face: "
      f"{sum(int((e - s) * FPS) for s, e, _, _ in APPEARANCES)} of {total} "
      f"({100 * sum((e - s) for s, e, _, _ in APPEARANCES) / DURATION_S:.0f}%)")
