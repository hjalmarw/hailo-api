"""Prepare everything the DFC needs, so compiling is short once an AVX host exists.

Produces three artefacts on the Pi, then they get copied to whichever box compiles:

  adaface_ir50.onnx        the model, opset 11, static 1x3x112x112
  calib_faces.npy          calibration set for INT8 quantisation
  reference_embedding.npy  known-good output for a fixed input

The calibration set is the part worth care. Quantisation picks scale factors from
whatever data you show it, so the closer that data is to production input, the smaller
the accuracy loss. These are real aligned crops from grazzy's own cameras, drawn across
all three subjects and every clip, and filtered by the calibrated 0.30 quality floor -
so no flower pots.

The reference embedding exists so that after compiling we can prove the HEF reproduces
the PyTorch model on identical input, rather than assuming it.
"""
import importlib
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

import face

OUT = "/home/grazzy/media/compile"
os.makedirs(OUT, exist_ok=True)

CLIPS = [
    ("hjalmar", "/home/grazzy/media/clip3.mp4"),
    ("hjalmar", "/home/grazzy/media/clip9.mp4"),
    ("hjalmar", "/home/grazzy/media/crusty/hjalmar_271514.mp4"),
    ("johanna", "/home/grazzy/media/clip5.mp4"),
    ("johanna", "/home/grazzy/media/crusty/ring_johanna_271570.mp4"),
    ("joanna",  "/home/grazzy/media/crusty/ring_joanna_265110.mp4"),
    ("joanna",  "/home/grazzy/media/crusty/ring_joanna2_264994.mp4"),
]
QUALITY_FLOOR = 0.30
TARGET_CALIB = 256          # DFC guidance is 64-1024; 256 is a sane middle

# ---- 1. calibration crops from real footage ------------------------------------
detector = face.get_face_detector()
crops = []
print("harvesting calibration crops (quality floor 0.30, real camera footage)")
for who, path in CLIPS:
    if not os.path.exists(path):
        continue
    cap = cv2.VideoCapture(path)
    got = 0
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
            if q < QUALITY_FLOOR:
                continue
            crops.append(a)
            got += 1
    cap.release()
    print(f"  {os.path.basename(path):32} {who:8} +{got}")

print(f"  {len(crops)} crops total")
if len(crops) > TARGET_CALIB:
    # spread the sample across the whole set rather than taking the first N,
    # so every clip and subject is represented
    idx = np.linspace(0, len(crops) - 1, TARGET_CALIB).astype(int)
    crops = [crops[i] for i in idx]

calib = np.stack(crops).astype(np.uint8)          # N,112,112,3 RGB
np.save(f"{OUT}/calib_faces.npy", calib)
print(f"  wrote calib_faces.npy {calib.shape}")

# ---- 2. ONNX export -------------------------------------------------------------
REPO, ARCH = "minchul/cvlface_adaface_ir50_webface4m", "IR_50"
from huggingface_hub import snapshot_download

local = snapshot_download(REPO)
if local not in sys.path:
    sys.path.insert(0, local)
mod = importlib.import_module("models.iresnet.model")
net = getattr(mod, ARCH)([112, 112])
raw = torch.load(os.path.join(local, "pretrained_model", "model.pt"),
                 map_location="cpu", weights_only=False)
st = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
st = {k[len("net."):]: v for k, v in st.items() if k.startswith("net.")}
net.load_state_dict(st, strict=False)
net.eval()


class EmbeddingOnly(torch.nn.Module):
    """cvlface forwards can return a tuple; DFC wants exactly one output tensor."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        y = self.inner(x)
        return y[0] if isinstance(y, (tuple, list)) else y


wrapped = EmbeddingOnly(net)
dummy = torch.zeros(1, 3, 112, 112)
onnx_path = f"{OUT}/adaface_ir50.onnx"
torch.onnx.export(wrapped, dummy, onnx_path,
                  input_names=["input"], output_names=["embedding"],
                  opset_version=11, do_constant_folding=True, dynamic_axes=None)
print(f"  wrote adaface_ir50.onnx ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

# ---- 3. reference embedding for post-compile verification -----------------------
probe = calib[0]
tensor = torch.from_numpy(
    ((probe.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None])
with torch.no_grad():
    ref = wrapped(tensor).numpy().reshape(-1)
ref = ref / (np.linalg.norm(ref) or 1.0)
np.save(f"{OUT}/reference_input.npy", probe)
np.save(f"{OUT}/reference_embedding.npy", ref)

json.dump({
    "repo": REPO, "arch": ARCH, "opset": 11,
    "input": "1x3x112x112 float32, RGB, (x/255-0.5)/0.5",
    "output": "512-d, L2-normalise after inference",
    "calibration": f"{calib.shape[0]} real aligned crops, quality>={QUALITY_FLOOR}",
    "hw_arch": "hailo10h",
    "cpu_accuracy": {"top1": 0.949, "coverage_at_100pct_precision": 0.892,
                     "identities": 3, "clips": 12},
}, open(f"{OUT}/compile_meta.json", "w"), indent=2)

print(f"  reference embedding: norm={np.linalg.norm(ref):.4f} "
      f"first5={ref[:5].round(4).tolist()}")
print(f"\nready in {OUT}:")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}  {os.path.getsize(os.path.join(OUT, f))/1e6:.1f} MB")
