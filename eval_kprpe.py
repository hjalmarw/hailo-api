"""Measure the KPRPE ViT — the last unevaluated model.

Keypoint-relative position encoding needs the five facial landmarks alongside the image.
We already have them from SCRFD in original-frame coordinates; align_face computes a
similarity transform to the canonical 112x112 crop, so applying that same matrix to the
landmarks puts them in the frame the model expects. No extra detection, no aligner.pt.

Worth testing despite my earlier dismissal: I ruled it out assuming ViTs would lose here,
and vit_base then tied on top-1 while winning coverage at 100% precision by 8 points.
KPRPE is the only model trained on WebFace12M - 3x the data - and training data has
predicted performance better than parameter count in every comparison today (IR-50 beat
IR-101; AdaFace beat ArcFace on identical architecture).

The keypoint format is not documented in the repo, so several plausible conventions are
tried and the first that produces a finite 512-d embedding is used. Whichever works is
reported, so the result is reproducible.
"""
import importlib
import json
import os
import sys
import time

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
ALLOW = ["*.py", "*.json", "*.yaml", "pretrained_model/model.pt"]

import cv2
import numpy as np
import torch
from PIL import Image

import face

CLIPS = {
    "clip3":        ("hjalmar", "/home/grazzy/media/clip3.mp4"),
    "clip9":        ("hjalmar", "/home/grazzy/media/clip9.mp4"),
    "yard_hjalmar": ("hjalmar", "/home/grazzy/media/crusty/hjalmar_271514.mp4"),
    "clip4":        ("johanna", "/home/grazzy/media/clip4.mp4"),
    "clip5":        ("johanna", "/home/grazzy/media/clip5.mp4"),
    "ring_johanna": ("johanna", "/home/grazzy/media/crusty/ring_johanna_271570.mp4"),
    "ring_joanna":  ("joanna",  "/home/grazzy/media/crusty/ring_joanna_265110.mp4"),
    "ring_joanna2": ("joanna",  "/home/grazzy/media/crusty/ring_joanna2_264994.mp4"),
}
QUALITY_FLOOR = 0.30

detector = face.get_face_detector()


def align_with_landmarks(rgb, landmarks):
    """Aligned 112x112 crop plus the landmarks carried into that frame."""
    src = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    matrix, _ = cv2.estimateAffinePartial2D(src, face.ARCFACE_TEMPLATE,
                                            method=cv2.LMEDS)
    if matrix is None:
        return None, None
    crop = cv2.warpAffine(rgb, matrix, (112, 112), borderValue=0.0,
                          flags=cv2.INTER_LINEAR)
    ones = np.ones((5, 1), dtype=np.float32)
    moved = (np.hstack([src, ones]) @ matrix.T).astype(np.float32)   # 5x2 in crop space
    return crop, moved


print("harvesting crops + aligned landmarks ...")
pools = {}
for name, (who, path) in CLIPS.items():
    if not os.path.exists(path):
        pools[name] = []
        continue
    cap = cv2.VideoCapture(path)
    kept = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for d in detector.detect(Image.fromarray(rgb), min_confidence=0.3):
            if d["area_px"] < 1500:
                continue
            crop, pts = align_with_landmarks(rgb, d["landmarks"])
            if crop is None:
                continue
            s = face.blur_score(crop)
            if s < 30.0:
                continue
            q = face.quality_score(d["area_px"], s, d["confidence"],
                                   face.frontality(d["landmarks"]))
            if q >= QUALITY_FLOOR:
                kept.append({"crop": crop, "kps": pts, "q": q})
    cap.release()
    pools[name] = kept
    print(f"  {name:14} {who:8} {len(kept):4}")

people = sorted({w for w, _ in CLIPS.values()})
n_total = sum(len(v) for v in pools.values())
print(f"  {n_total} faces, {len(people)} identities\n")

# ---- load ------------------------------------------------------------------------
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

REPO = "minchul/cvlface_adaface_vit_base_kprpe_webface12m"
local = snapshot_download(REPO, allow_patterns=ALLOW)
if local not in sys.path:
    sys.path.insert(0, local)
cfg = OmegaConf.load(os.path.join(local, "pretrained_model", "model.yaml"))
if "yaml_path" in cfg:
    cfg.yaml_path = os.path.join(local, str(cfg.yaml_path))
net = importlib.import_module("models.vit_kprpe").load_model(cfg)

raw = torch.load(os.path.join(local, "pretrained_model", "model.pt"),
                 map_location="cpu", weights_only=False)
st = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
want = set(net.state_dict().keys())
options = {"as-is": st,
           "strip net.": {k[len("net."):]: v for k, v in st.items()
                          if k.startswith("net.")}}
label, best = max(options.items(), key=lambda kv: len(set(kv[1]) & want))
overlap = len(set(best) & want)
print(f"key mapping: {label} -> {overlap}/{len(want)}")
if overlap < len(want):
    raise SystemExit("partial weight load; refusing to report a noise number")
net.load_state_dict(best, strict=False)
net.eval()

# ---- find the keypoint convention -------------------------------------------------
sample_crop = pools[next(iter(pools))][0]["crop"]
sample_kps = pools[next(iter(pools))][0]["kps"]
x = torch.from_numpy(
    ((sample_crop.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None])

CONVENTIONS = {
    "pixels 5x2":      lambda k: torch.from_numpy(k[None]),
    "normalised 5x2":  lambda k: torch.from_numpy((k / 112.0)[None]),
    "flat 10":         lambda k: torch.from_numpy(k.reshape(1, 10)),
    "flat 10 norm":    lambda k: torch.from_numpy((k / 112.0).reshape(1, 10)),
}
chosen = None
for cname, fn in CONVENTIONS.items():
    try:
        with torch.no_grad():
            y = net(x, fn(sample_kps.astype(np.float32)))
        y = y[0] if isinstance(y, (tuple, list)) else y
        v = y.numpy().reshape(-1)
        if v.shape[0] == 512 and np.isfinite(v).all():
            chosen = (cname, fn)
            print(f"keypoint convention: {cname} -> {tuple(y.shape)}")
            break
    except Exception as exc:
        print(f"  {cname}: {type(exc).__name__}: {str(exc)[:80]}")
if chosen is None:
    raise SystemExit("no keypoint convention worked; not evaluable")

cname, kfn = chosen


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def embed(entries):
    out = []
    with torch.no_grad():
        for e in entries:
            xx = torch.from_numpy(
                ((e["crop"].astype(np.float32) / 255.0 - 0.5) / 0.5)
                .transpose(2, 0, 1)[None])
            y = net(xx, kfn(e["kps"].astype(np.float32)))
            y = y[0] if isinstance(y, (tuple, list)) else y
            out.append(_unit(y.numpy().reshape(-1)))
    return out


t0 = time.perf_counter()
vectors = {c: embed(pools[c]) for c in CLIPS}
ms = (time.perf_counter() - t0) / max(n_total, 1) * 1000

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
        ok = names[int(o[0])] == truth
        scored.append((ok, float(s[o[0]] - s[o[1]])))
        hits += int(ok)
    per_fold.append(hits / len(vectors[probe]))

mean = float(np.mean(per_fold))
sem = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
cov = 0.0
for floor in np.arange(0.0, 0.9, 0.02):
    kept = [c for c, m in scored if m >= floor]
    if kept and all(kept):
        cov = len(kept) / len(scored)
        break

params = sum(p.numel() for p in net.parameters()) / 1e6
print(f"\nvit_kprpe      {mean:6.1%} +/-{sem:4.1%}  cov@100% {cov:5.1%}  "
      f"{ms:6.0f} ms  ({params:.0f}M)  [{cname}]")
print("\nfor comparison, same protocol:")
print("  adaface_ir50   94.9%  cov@100% 89.2%   233 ms   44M")
print("  vit_base       94.7%  cov@100% 97.1%   519 ms  115M")
print("  adaface_ir18   94.3%  cov@100% 87.8%   101 ms   24M")

json.dump({"acc": mean, "sem": sem, "cov100": cov, "ms": ms, "params": params,
           "kp_convention": cname},
          open("/home/grazzy/media/kprpe_eval.json", "w"), indent=2)
print("DONE")
