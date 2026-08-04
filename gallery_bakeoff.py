"""Round 2: how should a person's gallery be built from many noisy embeddings?

Round 1 varied the embedding pipeline. This varies what happens *after* — given N
embeddings of a person, most of them mediocre, what represents them best?

  plain-centroid    mean of everything, renormalised            (what ships)
  quality-weighted  mean weighted by the quality score          - lets good frames lead
  top-k-quality     mean of only the best K frames              - curated enrolment
  median            per-dimension median                        - robust to outliers
  max-over-all      nearest single embedding                    (the old default)
  two-centroids     k=2 clustering, best of either              - handles multi-modality
                    (a person with a cap and without is genuinely two appearances,
                     and one mean smears them into something matching neither)

Same leave-one-clip-out protocol as everywhere else, so numbers compare directly.
"""
import cv2
import numpy as np
from PIL import Image

import face

CLIPS = {
    "clip3": "man", "clip8": "man", "clip9": "man",
    "clip4": "woman", "clip5": "woman", "clip10": "woman",
}
MIN_CONF, MIN_PX, BLUR, STRIDE = 0.3, 1500, 30.0, 5

detector = face.get_face_detector()
embedder = face.get_face_embedder()

print("harvesting embeddings + quality...")
pools = {}
for clip in CLIPS:
    cap = cv2.VideoCapture(f"/home/grazzy/media/{clip}.mp4")
    i, out = 0, []
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
            sharp = face.blur_score(a)
            if sharp < BLUR:
                continue
            out.append({
                "v": embedder.embed(a),
                "q": face.quality_score(d["area_px"], sharp, d["confidence"],
                                        face.frontality(d["landmarks"])),
            })
    cap.release()
    pools[clip] = out
    print(f"  {clip:7} {CLIPS[clip]:5} {len(out):3}")
print()

people = sorted(set(CLIPS.values()))


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def build(entries, mode):
    """Return a list of representative vectors for one person."""
    vectors = np.stack([e["v"] for e in entries])
    weights = np.array([e["q"] for e in entries], dtype=np.float32)

    if mode == "plain-centroid":
        return [_unit(vectors.mean(axis=0))]
    if mode == "quality-weighted":
        if weights.sum() <= 0:
            return [_unit(vectors.mean(axis=0))]
        return [_unit((vectors * weights[:, None]).sum(axis=0) / weights.sum())]
    if mode == "top-k-quality":
        k = max(1, len(entries) // 3)
        idx = np.argsort(weights)[::-1][:k]
        return [_unit(vectors[idx].mean(axis=0))]
    if mode == "median":
        return [_unit(np.median(vectors, axis=0))]
    if mode == "max-over-all":
        return list(vectors)
    if mode == "two-centroids":
        if len(vectors) < 4:
            return [_unit(vectors.mean(axis=0))]
        # tiny deterministic k-means, k=2, cosine space
        c = np.stack([vectors[0], vectors[int(np.argmin(vectors @ vectors[0]))]])
        for _ in range(12):
            assign = np.argmax(vectors @ c.T, axis=1)
            new = []
            for j in (0, 1):
                members = vectors[assign == j]
                new.append(_unit(members.mean(axis=0)) if len(members) else c[j])
            new = np.stack(new)
            if np.allclose(new, c, atol=1e-5):
                break
            c = new
        return list(c)
    raise ValueError(mode)


MODES = ["plain-centroid", "quality-weighted", "top-k-quality", "median",
         "max-over-all", "two-centroids"]

print(f"{'gallery strategy':20} {'top-1':>8} {'mean margin':>12}")
results = {}
for mode in MODES:
    hits = total = 0
    margins = []
    for probe in CLIPS:
        if not pools[probe]:
            continue
        gallery = {}
        for person in people:
            entries = [e for c in CLIPS if CLIPS[c] == person and c != probe
                       for e in pools[c]]
            if entries:
                gallery[person] = build(entries, mode)
        if len(gallery) < 2:
            continue

        names = list(gallery)
        for entry in pools[probe]:
            # a person scores as their best representative
            scores = [max(float(r @ entry["v"]) for r in gallery[n]) for n in names]
            order = np.argsort(scores)[::-1]
            margins.append(scores[order[0]] - scores[order[1]])
            total += 1
            if names[int(order[0])] == CLIPS[probe]:
                hits += 1
    acc = hits / total if total else 0.0
    results[mode] = acc
    print(f"{mode:20} {acc:7.1%} {np.mean(margins):12.4f}")

print()
best = max(results, key=results.get)
print(f"BEST: {best} ({results[best]:.1%}) vs shipped plain-centroid "
      f"({results['plain-centroid']:.1%}) -> {results[best] - results['plain-centroid']:+.1%}")
