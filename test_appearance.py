"""Does CLIP appearance separate these two people, where faces struggled?

Faces gave 76.6% top-1 and were unusable on some clips because no face was presented.
Appearance should be available far more often. The question is whether it discriminates.

Protocol mirrors the face evaluation: leave one clip out, build appearance centroids
from the remaining clips, classify every person crop in the held-out clip.

  man   : clip3 (cam1 4K), clip8 (cam2), clip9 (cam3)
  woman : clip4 (cam1 4K), clip5 (cam2), clip10 (cam2)

Important caveat this test cannot escape: clips of the same person were likely filmed
the same day in the same clothes, which flatters an appearance model. Cross-DAY footage
is what would really test it. Reported accuracy is therefore an upper bound.
"""
import cv2
import numpy as np
from PIL import Image

import appearance
from detector import get_detector

CLIPS = {
    "clip3": "man", "clip8": "man", "clip9": "man",
    "clip4": "woman", "clip5": "woman", "clip10": "woman",
}
STRIDE = 10
MIN_BOX_PX = 4000

person_detector = get_detector("yolov8m")
encoder = appearance.get_clip_encoder()


def harvest(clip):
    capture = cv2.VideoCapture(f"/home/grazzy/media/{clip}.mp4")
    index, out = 0, []
    while True:
        if not capture.grab():
            break
        if index % STRIDE != 0:
            index += 1
            continue
        ok, frame = capture.retrieve()
        index += 1
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        detections, _ = person_detector.detect(Image.fromarray(rgb),
                                               min_confidence=0.4, labels=["person"])
        for det in detections:
            box = det["bbox"]
            pixels = [box["x_min"] * width, box["y_min"] * height,
                      box["x_max"] * width, box["y_max"] * height]
            area = (pixels[2] - pixels[0]) * (pixels[3] - pixels[1])
            if area < MIN_BOX_PX:
                continue
            out.append({
                "v": encoder.embed(appearance.crop_person(rgb, pixels)),
                "area": int(area),
                "conf": det["confidence"],
            })
    capture.release()
    return out


print("harvesting person crops (yolov8m -> CLIP)...")
pools = {}
for clip in CLIPS:
    pools[clip] = harvest(clip)
    areas = [e["area"] for e in pools[clip]]
    print(f"  {clip:7} {CLIPS[clip]:5} {len(pools[clip]):3} person crops"
          + (f"  median_area={int(np.median(areas))}" if areas else ""))
print()

people = sorted(set(CLIPS.values()))


def centroid(vectors):
    centre = np.mean(vectors, axis=0)
    norm = np.linalg.norm(centre)
    return centre / norm if norm else centre


print("=== leave-one-clip-out, appearance only ===")
total_hits = total_n = 0
for probe in CLIPS:
    if not pools[probe]:
        continue
    gallery = {}
    for person in people:
        vectors = [e["v"] for c in CLIPS if CLIPS[c] == person and c != probe
                   for e in pools[c]]
        if vectors:
            gallery[person] = centroid(np.stack(vectors))
    if len(gallery) < 2:
        continue

    names = list(gallery)
    matrix = np.stack([gallery[n] for n in names])
    truth = CLIPS[probe]
    hits = margins = 0
    margin_values = []
    for entry in pools[probe]:
        scores = matrix @ entry["v"]
        order = np.argsort(scores)[::-1]
        margin_values.append(float(scores[order[0]] - scores[order[1]]))
        if names[int(order[0])] == truth:
            hits += 1
    total_hits += hits
    total_n += len(pools[probe])
    print(f"  {probe:7} truth={truth:5} {hits:3}/{len(pools[probe]):3} = "
          f"{hits/len(pools[probe]):6.1%}  mean_margin={np.mean(margin_values):.3f}")

print()
print(f"APPEARANCE top-1 accuracy: {total_hits}/{total_n} = {total_hits/total_n:.1%}")
print("(face channel, same protocol, was 76.6%)")

print()
print("=== abstention: precision vs coverage on the appearance channel ===")
for floor in (0.0, 0.02, 0.05, 0.10, 0.15):
    correct = total = 0
    for probe in CLIPS:
        if not pools[probe]:
            continue
        gallery = {}
        for person in people:
            vectors = [e["v"] for c in CLIPS if CLIPS[c] == person and c != probe
                       for e in pools[c]]
            if vectors:
                gallery[person] = centroid(np.stack(vectors))
        if len(gallery) < 2:
            continue
        names = list(gallery)
        matrix = np.stack([gallery[n] for n in names])
        truth = CLIPS[probe]
        for entry in pools[probe]:
            scores = matrix @ entry["v"]
            order = np.argsort(scores)[::-1]
            if float(scores[order[0]] - scores[order[1]]) < floor:
                continue
            total += 1
            if names[int(order[0])] == truth:
                correct += 1
    if total:
        print(f"  margin>={floor:.2f}  precision={correct/total:6.1%}  "
              f"coverage={total/total_n:6.1%}  ({correct}/{total})")

print()
print("=== availability: how often is a PERSON visible vs a usable FACE? ===")
for clip in CLIPS:
    print(f"  {clip:7} person crops={len(pools[clip]):3}")
