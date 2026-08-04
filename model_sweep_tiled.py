"""Sweep every cvlface model worth trying, so we compile the RIGHT one to the NPU.

IR-50 gave 88.9%. It was simply the first thing that loaded — not a considered choice.
Bigger and better-trained variants exist and cost minutes to test on the Pi's CPU, which
is nothing against compiling the wrong model to a HEF.

Fixes two flaws in the previous evaluation:

  * a QUALITY FLOOR. joanna_255698 yielded 283 detections that were SCRFD
    false-positiving on a static object, max quality 0.030, and they polluted the
    gallery. Nothing was filtering on the quality score that already existed.
  * EVERY frame is examined, not every 5th, per grazzy's instruction — dense sampling
    finds the good moments that a stride can skip straight past.
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

# joanna_255698 is deliberately excluded: it contains no real face at any quality.
CLIPS = {
    "clip3":        ("hjalmar", "/home/grazzy/media/clip3.mp4"),
    "clip8":        ("hjalmar", "/home/grazzy/media/clip8.mp4"),
    "clip9":        ("hjalmar", "/home/grazzy/media/clip9.mp4"),
    "yard_hjalmar": ("hjalmar", "/home/grazzy/media/crusty/hjalmar_271514.mp4"),
    "clip4":        ("johanna", "/home/grazzy/media/clip4.mp4"),
    "clip5":        ("johanna", "/home/grazzy/media/clip5.mp4"),
    "clip10":       ("johanna", "/home/grazzy/media/clip10.mp4"),
    "yard_johanna": ("johanna", "/home/grazzy/media/crusty/johanna_271566.mp4"),
}

MIN_CONF, MIN_PX, BLUR = 0.3, 1500, 30.0
QUALITY_FLOOR = 0.05      # below this it is object noise, measured not guessed
STRIDE = 3                # tiling is ~9x the detector work per frame

detector = face.get_face_detector()

print(f"harvesting EVERY frame, TILED detection, quality floor {QUALITY_FLOOR} ...")
pools = {}
for name, (who, path) in CLIPS.items():
    if not os.path.exists(path):
        pools[name] = []
        continue
    cap = cv2.VideoCapture(path)
    index, kept, rejected = 0, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        if index % STRIDE:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for d in detector.detect_auto(Image.fromarray(rgb), min_confidence=MIN_CONF):
            if d["area_px"] < MIN_PX:
                continue
            aligned = face.align_face(rgb, np.array(d["landmarks"], dtype=np.float32))
            sharp = face.blur_score(aligned)
            if sharp < BLUR:
                continue
            quality = face.quality_score(d["area_px"], sharp, d["confidence"],
                                         face.frontality(d["landmarks"]))
            if quality < QUALITY_FLOOR:
                rejected += 1
                continue
            kept.append({"crop": aligned, "q": quality})
    cap.release()
    pools[name] = kept
    print(f"  {name:14} {who:8} {len(kept):4} kept, {rejected:5} below floor "
          f"({index} frames)")

n_total = sum(len(v) for v in pools.values())
people = sorted({who for who, _ in CLIPS.values()})
print(f"\n{n_total} usable faces, {len(people)} identities, chance {1/len(people):.1%}\n")


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def score(vectors):
    per_fold, scored = [], []
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
            continue
        names = list(gallery)
        matrix = np.stack([gallery[n] for n in names])
        hits = 0
        for v in vectors[probe]:
            s = matrix @ v
            o = np.argsort(s)[::-1]
            correct = names[int(o[0])] == truth
            scored.append((correct, float(s[o[0]] - s[o[1]])))
            hits += int(correct)
        per_fold.append(hits / len(vectors[probe]))
    mean = float(np.mean(per_fold)) if per_fold else 0.0
    sem = (float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
           if len(per_fold) > 1 else 0.0)
    # coverage at 100% precision — the number a secondary identity system lives on
    best_cov = 0.0
    for floor in np.arange(0.0, 0.8, 0.02):
        kept = [c for c, m in scored if m >= floor]
        if kept and all(kept):
            best_cov = len(kept) / len(scored)
            break
    return mean, sem, best_cov


def load_cvlface(repo, arch):
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo)
    if local not in sys.path:
        sys.path.insert(0, local)
    module = importlib.import_module(f"models.{arch[0]}.model")
    net = getattr(module, arch[1])([112, 112])
    raw = torch.load(os.path.join(local, "pretrained_model", "model.pt"),
                     map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    state = {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")}
    net.load_state_dict(state, strict=False)
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


CANDIDATES = [
    ("adaface_ir18",   "minchul/cvlface_adaface_ir18_webface4m",  ("iresnet", "IR_18")),
    ("adaface_ir50",   "minchul/cvlface_adaface_ir50_webface4m",  ("iresnet", "IR_50")),
]

results = {}

npu = face.get_face_embedder()
t0 = time.perf_counter()
npu_vectors = {c: [npu.embed(e["crop"]) for e in pools[c]] for c in CLIPS}
npu_ms = (time.perf_counter() - t0) / max(n_total, 1) * 1000
results["mobilefacenet (NPU)"] = (*score(npu_vectors), npu_ms, 1.0)
print(f"{'mobilefacenet (NPU)':24} {results['mobilefacenet (NPU)'][0]:6.1%}")

for label, repo, arch in CANDIDATES:
    try:
        model = load_cvlface(repo, arch)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        t0 = time.perf_counter()
        vectors = {c: embed(model, [e["crop"] for e in pools[c]]) for c in CLIPS}
        ms = (time.perf_counter() - t0) / max(n_total, 1) * 1000
        acc, sem, cov = score(vectors)
        results[label] = (acc, sem, cov, ms, params)
        print(f"{label:24} {acc:6.1%} +/-{sem:4.1%}  100%-precision coverage {cov:5.1%}  "
              f"{ms:6.0f} ms  ({params:.0f}M)")
        del model
    except Exception as exc:
        print(f"{label:24} FAILED: {type(exc).__name__}: {exc}")

print("\n" + "=" * 78)
print(f"{'model':24} {'top-1':>8} {'noise':>7} {'cov@100%':>10} {'ms/face':>9} {'params':>8}")
for label, (acc, sem, cov, ms, params) in sorted(results.items(), key=lambda kv: -kv[1][0]):
    print(f"{label:24} {acc:8.1%} {sem:7.1%} {cov:10.1%} {ms:9.0f} {params:7.0f}M")

best = max(results, key=lambda k: results[k][0])
print(f"\nBEST: {best} -> this is the model worth compiling to HEF")
with open("/home/grazzy/media/sweep_tiled.json", "w") as fh:
    json.dump({k: {"acc": v[0], "sem": v[1], "cov100": v[2], "ms": v[3], "params": v[4]}
               for k, v in results.items()}, fh, indent=2)
