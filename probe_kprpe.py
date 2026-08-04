"""Can the KPRPE ViT be evaluated with the crops we already produce?

Reason for testing it now, having earlier deprioritised it: I assumed ViTs would lose on
this footage. vit_base then tied on top-1 and won decisively on coverage at 100%
precision (97.1% vs 89.2%). The KPRPE variant is the only model trained on WebFace12M -
3x the data - and training data has been the strongest predictor of performance here all
day. So the assumption that ruled it out is gone.

The obstacle is interface, not capability: KPRPE means keypoint-relative position
encoding, so forward() likely wants facial landmarks alongside the image. We have those
from SCRFD, but in original-frame coordinates, whereas the model will want them relative
to the aligned 112x112 crop.

This probes the interface before committing to a full evaluation - checks the forward
signature and whether a plain image-only call works.
"""
import glob
import importlib
import inspect
import os
import sys

import torch

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
REPO = "minchul/cvlface_adaface_vit_base_kprpe_webface12m"
ALLOW = ["*.py", "*.json", "*.yaml", "pretrained_model/model.pt",
         "pretrained_model/aligner.pt"]

from huggingface_hub import snapshot_download

local = snapshot_download(REPO, allow_patterns=ALLOW)
print(f"snapshot: {local}")
print("files:")
for f in sorted(glob.glob(os.path.join(local, "**", "*.py"), recursive=True))[:10]:
    print(f"   {os.path.relpath(f, local)}")

if local not in sys.path:
    sys.path.insert(0, local)

from omegaconf import OmegaConf

cfg = OmegaConf.load(os.path.join(local, "pretrained_model", "model.yaml"))
print(f"\nconfig: {dict(cfg)}")
if "yaml_path" in cfg:
    cfg.yaml_path = os.path.join(local, str(cfg.yaml_path))

mod = importlib.import_module("models.vit_kprpe")
net = mod.load_model(cfg)
print(f"\nbuilt: {type(net).__name__}, "
      f"{sum(p.numel() for p in net.parameters())/1e6:.0f}M params")

sig = inspect.signature(net.forward)
print(f"forward signature: {sig}")

# key mapping, chosen by measurement as before
raw = torch.load(os.path.join(local, "pretrained_model", "model.pt"),
                 map_location="cpu", weights_only=False)
st = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
want = set(net.state_dict().keys())
options = {
    "as-is": st,
    "strip net.": {k[len("net."):]: v for k, v in st.items() if k.startswith("net.")},
}
name, best = max(options.items(), key=lambda kv: len(set(kv[1]) & want))
print(f"key mapping: {name} -> {len(set(best) & want)}/{len(want)}")

if len(set(best) & want) == len(want):
    net.load_state_dict(best, strict=False)
    net.eval()
    x = torch.zeros(1, 3, 112, 112)
    try:
        with torch.no_grad():
            y = net(x)
        y = y[0] if isinstance(y, (tuple, list)) else y
        print(f"IMAGE-ONLY FORWARD WORKS -> {tuple(y.shape)}")
        print("VERDICT: evaluable with our existing crops, no landmarks needed")
    except Exception as exc:
        print(f"image-only forward failed: {type(exc).__name__}: {exc}")
        print("VERDICT: needs landmarks; would require mapping SCRFD points into the "
              "aligned 112x112 frame")
else:
    print("VERDICT: key mismatch, not evaluable as-is")
print("DONE")
