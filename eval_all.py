"""Evaluate against every labelled clip, not just the garden subset.

I had been re-running the same 8 clips while four downloaded ring-doorbell clips sat
unused. This uses all of them.

Single-person clips (clean labels, usable for leave-one-clip-out):

  hjalmar  clip3, clip8, clip9, yard_271514                       4 clips, 3 cameras
  johanna  clip4, clip5, clip10, yard_271566, ring_271570         5 clips
  joanna   ring_265110, ring_264994, 255698                       3 clips

Chance drops from 50% to 33%, and joanna finally has enough clips to be scored as a
probe rather than sitting in the gallery as a distractor.

255698 is included deliberately. It is the clip that produced only flower-pot detections;
at the calibrated 0.30 floor those should be rejected, so it is also a live test of
whether the floor holds on the exact footage that exposed the problem.

Mixed clips (ring_271714 johanna+ylva, room3_271844 three people) are excluded here:
without per-face labels they cannot be scored. Handled separately by clustering.
"""
import importlib
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

import face

CLIPS = {
    "clip3":         ("hjalmar", "/home/grazzy/media/clip3.mp4"),
    "clip8":         ("hjalmar", "/home/grazzy/media/clip8.mp4"),
    "clip9":         ("hjalmar", "/home/grazzy/media/clip9.mp4"),
    "yard_hjalmar":  ("hjalmar", "/home/grazzy/media/crusty/hjalmar_271514.mp4"),
    "clip4":         ("johanna", "/home/grazzy/media/clip4.mp4"),
    "clip5":         ("johanna", "/home/grazzy/media/clip5.mp4"),
    "clip10":        ("johanna", "/home/grazzy/media/clip10.mp4"),
    "yard_johanna":  ("johanna", "/home/grazzy/media/crusty/johanna_271566.mp4"),
    "ring_johanna":  ("johanna", "/home/grazzy/media/crusty/ring_johanna_271570.mp4"),
    "ring_joanna":   ("joanna",  "/home/grazzy/media/crusty/ring_joanna_265110.mp4"),
    "ring_joanna2":  ("joanna",  "/home/grazzy/media/crusty/ring_joanna2_264994.mp4"),
    "yard_joanna":   ("joanna",  "/home/grazzy/media/crusty/joanna_255698.mp4"),
}
MIN_CONF, MIN_PX, BLUR = 0.3, 1500, 30.0
QUALITY_FLOOR = 0.30

detector = face.get_face_detector()

print(f"harvesting every frame, quality floor {QUALITY_FLOOR} (whole-frame, no tiling)")
pools = {}
for name, (who, path) in CLIPS.items():
    if not os.path.exists(path):
        print(f"  {name:14} MISSING")
        pools[name] = []
        continue
    cap = cv2.VideoCapture(path)
    kept, rejected = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for d in detector.detect(Image.fromarray(rgb), min_confidence=MIN_CONF):
            if d["area_px"] < MIN_PX:
                continue
            a = face.align_face(rgb, np.array(d["landmarks"], dtype=np.float32))
            s = face.blur_score(a)
            if s < BLUR:
                continue
            q = face.quality_score(d["area_px"], s, d["confidence"],
                                   face.frontality(d["landmarks"]))
            if q < QUALITY_FLOOR:
                rejected += 1
                continue
            kept.append({"crop": a, "q": q})
    cap.release()
    pools[name] = kept
    print(f"  {name:14} {who:8} {len(kept):4} kept, {rejected:5} below floor")

people = sorted({w for w, _ in CLIPS.values()})
n_total = sum(len(v) for v in pools.values())
print(f"\n{n_total} usable faces, {len(people)} identities {people}, "
      f"chance {1/len(people):.1%}\n")


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def score(vectors, label):
    per_fold, scored, rows = [], [], []
    for probe in CLIPS:
        if not vectors[probe]:
            continue
        truth = CLIPS[probe][0]
        gallery = {}
        for person in people:
            items = [(v, e["q"]) for c in CLIPS if CLIPS[c][0] == person and c != probe
                     for v, e in zip(vectors[c], pools[c])]
            if not items:
                continue
            vs = np.stack([v for v, _ in items])
            ws = np.array([max(q, 1e-6) for _, q in items], dtype=np.float32)
            gallery[person] = _unit((vs * ws[:, None]).sum(axis=0) / ws.sum())
        if truth not in gallery or len(gallery) < 2:
            rows.append((probe, truth, None, len(vectors[probe])))
            continue
        names = list(gallery)
        matrix = np.stack([gallery[n] for n in names])
        hits = 0
        for v in vectors[probe]:
            s = matrix @ v
            o = np.argsort(s)[::-1]
            ok = names[int(o[0])] == truth
            scored.append((ok, float(s[o[0]] - s[o[1]])))
            hits += int(ok)
        acc = hits / len(vectors[probe])
        per_fold.append(acc)
        rows.append((probe, truth, acc, len(vectors[probe])))

    mean = float(np.mean(per_fold)) if per_fold else 0.0
    sem = (float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
           if len(per_fold) > 1 else 0.0)
    cov = 0.0
    for floor in np.arange(0.0, 0.9, 0.02):
        kept = [c for c, m in scored if m >= floor]
        if kept and all(kept):
            cov = len(kept) / len(scored)
            break

    print(f"--- {label} ---")
    for probe, truth, acc, n in rows:
        shown = f"{acc:6.1%}" if acc is not None else "   -- "
        print(f"   {probe:14} {truth:8} {shown}  n={n:4}")
    print(f"   MEAN {mean:.1%} +/-{sem:.1%}   coverage@100%precision {cov:.1%}\n")
    return mean, sem, cov


def load_adaface(repo, arch):
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo)
    if local not in sys.path:
        sys.path.insert(0, local)
    m = importlib.import_module("models.iresnet.model")
    net = getattr(m, arch)([112, 112])
    raw = torch.load(os.path.join(local, "pretrained_model", "model.pt"),
                     map_location="cpu", weights_only=False)
    st = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    st = {k[len("net."):]: v for k, v in st.items() if k.startswith("net.")}
    net.load_state_dict(st, strict=False)
    net.eval()
    return net


def embed(model, crops):
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), 8):
            arr = np.stack(crops[i:i + 8]).astype(np.float32)
            arr = (arr / 255.0 - 0.5) / 0.5
            t = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))
            y = model(t)
            if isinstance(y, (tuple, list)):
                y = y[0]
            out.extend(y.cpu().numpy())
    return [_unit(v.astype(np.float32).reshape(-1)) for v in out]


results = {}
npu = face.get_face_embedder()
results["mobilefacenet (NPU)"] = score(
    {c: [npu.embed(e["crop"]) for e in pools[c]] for c in CLIPS}, "mobilefacenet (NPU)")

for label, repo, arch in [
        ("adaface_ir18", "minchul/cvlface_adaface_ir18_webface4m", "IR_18"),
        ("adaface_ir50", "minchul/cvlface_adaface_ir50_webface4m", "IR_50")]:
    model = load_adaface(repo, arch)
    t0 = time.perf_counter()
    vecs = {c: embed(model, [e["crop"] for e in pools[c]]) for c in CLIPS}
    ms = (time.perf_counter() - t0) / max(n_total, 1) * 1000
    results[label] = score(vecs, f"{label} ({ms:.0f} ms/face)")
    del model

print("=" * 74)
print(f"{'model':24} {'top-1':>8} {'noise':>8} {'cov@100%':>10}")
for k, (a, s, c) in sorted(results.items(), key=lambda kv: -kv[1][0]):
    print(f"{k:24} {a:8.1%} {s:8.1%} {c:10.1%}")
print(f"\nchance with {len(people)} identities: {1/len(people):.1%}")

with open("/home/grazzy/media/eval_all.json", "w") as fh:
    json.dump({k: {"acc": v[0], "sem": v[1], "cov100": v[2]}
               for k, v in results.items()}
              | {"people": people, "faces": {k: len(v) for k, v in pools.items()}},
              fh, indent=2)
