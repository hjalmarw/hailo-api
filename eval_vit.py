"""Measure the ViT AdaFace variants — the last unevaluated models.

  cvlface_adaface_vit_base_webface4m         ViT-B, no KPRPE, drop-in
  cvlface_adaface_vit_base_kprpe_webface12m  ViT-B + keypoint-relative position
                                             encoding, WebFace12M (3x the training data)

Prior expectation, stated before measuring: IR-101 at 65M params lost to IR-50 at 44M on
this footage, so a ViT-B at ~86M is not obviously going to win. Training data mattered
more than capacity every time so far — and the KPRPE model is the only one trained on
WebFace12M, which is the one axis where it genuinely has an edge.

KPRPE needs landmarks fed alongside the image (that is what the shipped aligner.pt is
for), so it may not be a straight drop-in. Handled by trying the plain ViT first.

Weights are ~460MB per repo, so HF_HOME points at the QNAP mount: the Pi is at 90% and
two of these would fill it.
"""
import importlib
import json
import os
import sys
import time

# The QNAP is CIFS-mounted and has no symlinks, which huggingface_hub's cache requires;
# HF_HUB_DISABLE_SYMLINKS only silences the warning, it does not change the behaviour.
# So cache locally instead and fetch only what is needed: the checkpoint plus the repo's
# own architecture code, skipping the redundant safetensors copy of the same weights.
# That is ~460MB rather than ~920MB, which fits the Pi's remaining space.
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
print(f"HF cache: local (CIFS has no symlinks); selective download {ALLOW}")
print("harvesting (quality floor 0.30) ...")
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
            a = face.align_face(rgb, np.array(d["landmarks"], dtype=np.float32))
            s = face.blur_score(a)
            if s < 30.0:
                continue
            q = face.quality_score(d["area_px"], s, d["confidence"],
                                   face.frontality(d["landmarks"]))
            if q >= QUALITY_FLOOR:
                kept.append({"crop": a, "q": q})
    cap.release()
    pools[name] = kept
    print(f"  {name:14} {who:8} {len(kept):4}")

people = sorted({w for w, _ in CLIPS.values()})
n_total = sum(len(v) for v in pools.values())
print(f"  {n_total} faces, {len(people)} identities\n")


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
            ok = names[int(o[0])] == truth
            scored.append((ok, float(s[o[0]] - s[o[1]])))
            hits += int(ok)
        per_fold.append(hits / len(vectors[probe]))
    mean = float(np.mean(per_fold)) if per_fold else 0.0
    sem = (float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
           if len(per_fold) > 1 else 0.0)
    cov = 0.0
    for floor in np.arange(0.0, 0.9, 0.02):
        kept = [c for c, m in scored if m >= floor]
        if kept and all(kept):
            cov = len(kept) / len(scored)
            break
    return mean, sem, cov


def load(repo, module_path, builder):
    """Use the repo's own loader rather than guessing constructor arguments.

    vit.py exposes only a VisionTransformer class, no builder functions; the package
    __init__ has load_model(config) -> ViTModel.from_config, and the config lives in
    pretrained_model/model.yaml alongside the checkpoint. Going through their loader
    means the architecture matches the weights by construction.
    """
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    local = snapshot_download(repo, allow_patterns=ALLOW)
    if local not in sys.path:
        sys.path.insert(0, local)
    cfg = OmegaConf.load(os.path.join(local, "pretrained_model", "model.yaml"))
    # yaml_path inside the config is relative to the repo root
    if "yaml_path" in cfg:
        cfg.yaml_path = os.path.join(local, str(cfg.yaml_path))
    m = importlib.import_module(module_path)
    net = m.load_model(cfg)
    raw = torch.load(os.path.join(local, "pretrained_model", "model.pt"),
                     map_location="cpu", weights_only=False)
    st = raw.get("state_dict", raw) if isinstance(raw, dict) else raw

    # The IResNet repos hand back a bare network, so their checkpoints need the "net."
    # prefix stripped. models.vit.load_model returns a WRAPPER whose own state dict
    # already carries that prefix - stripping there breaks every key. Rather than encode
    # either assumption, pick whichever mapping actually matches the architecture.
    want = set(net.state_dict().keys())
    candidates = {
        "as-is": st,
        "strip net.": {k[len("net."):]: v for k, v in st.items() if k.startswith("net.")},
    }
    name, best = max(candidates.items(),
                     key=lambda kv: len(set(kv[1]) & want))
    overlap = len(set(best) & want)
    print(f"    key mapping: {name} -> {overlap}/{len(want)} match")
    if overlap < len(want):
        raise RuntimeError(
            f"only {overlap}/{len(want)} weights matched; refusing to evaluate a "
            "partially-initialised model - the number would be noise, not a result")
    missing, unexpected = net.load_state_dict(best, strict=False)
    print(f"    loaded ({len(missing)} missing, {len(unexpected)} unexpected)")
    net.eval()
    return net


def embed(model, crops):
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), 4):
            arr = np.stack(crops[i:i + 4]).astype(np.float32)
            arr = (arr / 255.0 - 0.5) / 0.5
            t = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))
            y = model(t)
            if isinstance(y, (tuple, list)):
                y = y[0]
            out.extend(y.cpu().numpy())
    return [_unit(v.astype(np.float32).reshape(-1)) for v in out]


CANDIDATES = [
    ("vit_base", "minchul/cvlface_adaface_vit_base_webface4m", "models.vit", None),
]

results = {}
for label, repo, modpath, builder in CANDIDATES:
    try:
        print(f"loading {label} ...")
        model = load(repo, modpath, builder)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        t0 = time.perf_counter()
        vecs = {c: embed(model, [e["crop"] for e in pools[c]]) for c in CLIPS}
        ms = (time.perf_counter() - t0) / max(n_total, 1) * 1000
        acc, sem, cov = score(vecs)
        results[label] = (acc, sem, cov, ms, params)
        print(f"  {label:14} {acc:6.1%} +/-{sem:4.1%}  cov@100% {cov:5.1%}  "
              f"{ms:6.0f} ms  ({params:.0f}M)")
        del model
    except Exception as exc:
        print(f"  {label} FAILED: {type(exc).__name__}: {exc}")

print("\n" + "=" * 70)
print("for comparison, same protocol and floor:")
print("  adaface_ir50    94.9%  cov@100% 89.2%   233 ms   44M")
print("  adaface_ir18    94.3%  cov@100% 87.8%   101 ms   24M")
for k, (a, s, c, ms, p) in results.items():
    print(f"  {k:14}  {a:.1%}  cov@100% {c:.1%}  {ms:.0f} ms  {p:.0f}M")

if results:
    json.dump({k: {"acc": v[0], "sem": v[1], "cov100": v[2], "ms": v[3], "params": v[4]}
               for k, v in results.items()},
              open("/home/grazzy/media/vit_eval.json", "w"), indent=2)
print("DONE")
