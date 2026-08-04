"""Why did the ViT checkpoint load with 279 missing / 281 unexpected keys?

That mismatch means load_state_dict(strict=False) accepted a checkpoint that does not
correspond to the constructed architecture — leaving a near-randomly-initialised network.
Any accuracy measured from it would be noise reported as a result, which is worse than
no result.

Most likely cause: I strip a "net." prefix, copied from the IResNet repos. The ViT
checkpoint probably uses a different prefix, or none.
"""
import glob
import os

import torch

snap = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--minchul--cvlface_adaface_vit_base_webface4m"
    "/snapshots/*/"))[0]
raw = torch.load(os.path.join(snap, "pretrained_model", "model.pt"),
                 map_location="cpu", weights_only=False)
sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw

keys = list(sd.keys())
print(f"checkpoint tensors: {len(keys)}")
print("first 8:")
for k in keys[:8]:
    print(f"   {k}")
print("last 4:")
for k in keys[-4:]:
    print(f"   {k}")

prefixes = {}
for k in keys:
    prefixes[k.split(".")[0]] = prefixes.get(k.split(".")[0], 0) + 1
print("top-level prefixes:", dict(sorted(prefixes.items(), key=lambda kv: -kv[1])[:8]))

# what does the architecture expect?
import importlib
import sys

if snap not in sys.path:
    sys.path.insert(0, snap)
from omegaconf import OmegaConf

cfg = OmegaConf.load(os.path.join(snap, "pretrained_model", "model.yaml"))
if "yaml_path" in cfg:
    cfg.yaml_path = os.path.join(snap, str(cfg.yaml_path))
mod = importlib.import_module("models.vit")
net = mod.load_model(cfg)
want = list(net.state_dict().keys())
print(f"\narchitecture tensors: {len(want)}")
print("first 8:")
for k in want[:8]:
    print(f"   {k}")

# find the prefix transform that maximises overlap
best = None
for strip in ("", "net.", "model.", "module.", "backbone."):
    mapped = {k[len(strip):] if k.startswith(strip) else k: v for k, v in sd.items()}
    overlap = len(set(mapped) & set(want))
    print(f"  strip {strip!r:12} -> {overlap}/{len(want)} match")
    if best is None or overlap > best[1]:
        best = (strip, overlap)
print(f"\nbest: strip {best[0]!r} giving {best[1]}/{len(want)}")
