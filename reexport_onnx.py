"""Re-export AdaFace IR-50 in a form the Hailo parser will accept.

The first attempt produced opset 18 with external data. Both are problems:

  * torch 2.9's torch.onnx.export defaults to the dynamo exporter, which ignored
    opset_version=11 once its downgrade converter hit "No Adapter From Version 16 for
    Identity". It emitted 18 instead and reported success.
  * dynamo also wrote weights to a separate .data sidecar. A self-contained file is far
    less to go wrong when handing it to another toolchain on another host.

dynamo=False selects the legacy TorchScript exporter, which honours opset_version
properly and inlines the weights.

Verifies the result afterwards rather than trusting the exporter's own success message -
that is exactly what misled the first run.
"""
import importlib
import os
import sys

import numpy as np
import onnx
import torch

OUT = "/home/grazzy/media/compile"
REPO, ARCH = "minchul/cvlface_adaface_ir50_webface4m", "IR_50"
TARGET_OPSET = 11

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
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        y = self.inner(x)
        return y[0] if isinstance(y, (tuple, list)) else y


wrapped = EmbeddingOnly(net)
dummy = torch.zeros(1, 3, 112, 112)

for opset in (TARGET_OPSET, 12, 13):
    path = f"{OUT}/adaface_ir50_op{opset}.onnx"
    try:
        torch.onnx.export(
            wrapped, dummy, path,
            input_names=["input"], output_names=["embedding"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,               # legacy exporter: honours opset, inlines weights
        )
    except Exception as exc:
        print(f"opset {opset}: export failed - {type(exc).__name__}: {exc}")
        continue

    # verify rather than trust the success message
    m = onnx.load(path, load_external_data=False)
    got = {op.version for op in m.opset_import if op.domain in ("", "ai.onnx")}
    external = any(t.HasField("data_location") and t.data_location == 1
                   for t in m.graph.initializer)
    size = os.path.getsize(path) / 1e6
    try:
        onnx.checker.check_model(path)
        valid = "valid"
    except Exception as exc:
        valid = f"CHECKER FAILED: {exc}"
    print(f"opset {opset}: emitted={got} external_data={external} "
          f"{size:.1f} MB  {valid}")

    if opset in got and not external:
        # numerical parity against the PyTorch model
        try:
            import onnxruntime as ort
            probe = np.load(f"{OUT}/reference_input.npy")
            x = ((probe.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            got_vec = sess.run(None, {"input": x})[0].reshape(-1)
            got_vec = got_vec / (np.linalg.norm(got_vec) or 1.0)
            ref = np.load(f"{OUT}/reference_embedding.npy")
            print(f"           cosine vs pytorch reference: {float(ref @ got_vec):.6f}")
        except ImportError:
            print("           (onnxruntime not installed - parity check skipped)")
        os.replace(path, f"{OUT}/adaface_ir50.onnx")
        for stale in (f"{OUT}/adaface_ir50.onnx.data",):
            if os.path.exists(stale):
                os.remove(stale)
        print(f"           -> adopted as adaface_ir50.onnx")
        break

print("\nfinal artefacts:")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}  {os.path.getsize(os.path.join(OUT, f))/1e6:.1f} MB")
