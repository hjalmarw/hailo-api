"""Face embedding backends, selectable per deployment.

Two options with a genuine trade-off:

  mobilefacenet  ~1M params   NPU    ~2ms    74.5% on real CCTV
  adaface_ir50   43.6M params CPU  ~216ms    95.9% on the same footage

AdaFace was built for low-quality faces (state of the art on IJB-S and TinyFace, which
are surveillance benchmarks) and it shows: +21pp on grazzy's cameras, with the margin
between candidates roughly doubling, which matters just as much because abstention keys
off margin.

216ms on a Pi CPU is slower than the NPU by two orders of magnitude, but for
event-triggered identification - a person detector fires, we answer "who?" - a fifth of
a second is irrelevant. Use mobilefacenet only where per-frame realtime is genuinely
required.

DANGER, and the reason models are tagged everywhere below: both produce 512-d unit
vectors, so mixing them is dimensionally valid and semantically meaningless. A gallery
enrolled with one model and queried with the other yields plausible-looking similarity
scores that are pure noise. Every embedding carries its model name, and matching refuses
to cross that line.
"""

import threading
from typing import List, Optional

import numpy as np

MOBILEFACENET = "mobilefacenet"
ADAFACE_IR50 = "adaface_ir50"
ADAFACE_IR18 = "adaface_ir18"

EMBEDDING_DIM = 512

_ADAFACE_REPOS = {
    ADAFACE_IR50: ("minchul/cvlface_adaface_ir50_webface4m", "IR_50"),
    ADAFACE_IR18: ("minchul/cvlface_adaface_ir18_webface4m", "IR_18"),
}


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


class AdaFaceEmbedder:
    """AdaFace IResNet on CPU via torch.

    Loading these cvlface repos through transformers' AutoModel does not work: the
    wrapper predates the v5 API and reads its config by relative path. The repo ships
    the architecture and a plain state dict, so we build it directly - fewer moving
    parts and no version sensitivity.
    """

    def __init__(self, variant: str = ADAFACE_IR50, threads: Optional[int] = None):
        if variant not in _ADAFACE_REPOS:
            raise ValueError(f"unknown AdaFace variant: {variant}")
        self.variant = variant
        self.model_name = variant
        self._threads = threads
        self._net = None
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        import importlib
        import os
        import sys

        import torch
        from huggingface_hub import snapshot_download

        repo, arch = _ADAFACE_REPOS[self.variant]
        try:
            local = snapshot_download(repo)
            if local not in sys.path:
                sys.path.insert(0, local)

            module = importlib.import_module("models.iresnet.model")
            net = getattr(module, arch)([112, 112])

            raw = torch.load(os.path.join(local, "pretrained_model", "model.pt"),
                             map_location="cpu", weights_only=False)
            state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
            state = {k[len("net."):]: v for k, v in state.items()
                     if k.startswith("net.")}

            missing, unexpected = net.load_state_dict(state, strict=False)
            if missing:
                print(f"{self.variant}: {len(missing)} missing keys on load")
            net.eval()

            if self._threads:
                torch.set_num_threads(self._threads)

            self._net = net
            print(f"{self.variant} initialized "
                  f"({sum(p.numel() for p in net.parameters())/1e6:.1f}M params, CPU)")
            return True
        except Exception as exc:
            print(f"Failed to initialize {self.variant}: {exc}")
            return False

    @property
    def is_ready(self) -> bool:
        return self._net is not None

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        return self.embed_batch([aligned_face])[0]

    def embed_batch(self, crops: List[np.ndarray], batch_size: int = 8) -> List[np.ndarray]:
        """Batched embedding — meaningfully faster than one-at-a-time on CPU."""
        if not self.is_ready:
            raise RuntimeError(f"{self.variant} not initialized")
        if not crops:
            return []

        import cv2
        import torch

        prepared = []
        for crop in crops:
            if crop.shape[:2] != (112, 112):
                crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
            prepared.append(crop)

        out = []
        with self._lock, torch.no_grad():
            for i in range(0, len(prepared), batch_size):
                batch = np.stack(prepared[i:i + batch_size]).astype(np.float32)
                # The shipped config says color_space: RGB. Feeding BGR here produces
                # confident, meaningless embeddings — silent and very hard to spot.
                batch = (batch / 255.0 - 0.5) / 0.5
                tensor = torch.from_numpy(
                    np.ascontiguousarray(batch.transpose(0, 3, 1, 2)))
                result = self._net(tensor)
                if isinstance(result, (tuple, list)):
                    result = result[0]
                out.extend(result.cpu().numpy())

        return [_unit(v.astype(np.float32).reshape(-1)) for v in out]

    def close(self):
        self._net = None


_adaface_cache: dict = {}
_cache_lock = threading.Lock()


def get_adaface(variant: str = ADAFACE_IR50) -> AdaFaceEmbedder:
    with _cache_lock:
        embedder = _adaface_cache.get(variant)
        if embedder is None:
            embedder = AdaFaceEmbedder(variant)
            if not embedder.initialize():
                raise RuntimeError(f"could not initialize {variant}")
            _adaface_cache[variant] = embedder
    return embedder


def close_adaface():
    with _cache_lock:
        for embedder in _adaface_cache.values():
            embedder.close()
        _adaface_cache.clear()


def available_embedders() -> List[str]:
    return [MOBILEFACENET, ADAFACE_IR18, ADAFACE_IR50]
