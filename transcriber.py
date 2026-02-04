"""Whisper speech-to-text transcriber using transformers on CPU."""

import subprocess
import time
import tempfile
import threading
from typing import Optional

import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

SAMPLE_RATE = 16000

# Supported variants (HuggingFace model names)
WHISPER_VARIANTS = ["tiny", "tiny.en", "base", "base.en"]


class WhisperTranscriber:
    """Whisper speech-to-text running on CPU via HuggingFace transformers."""

    def __init__(self, variant: str = "tiny.en"):
        self.variant = variant
        self._inference_lock = threading.Lock()
        self._ready = False
        self.start_time = time.time()
        self._processor = None
        self._model = None
        self._initialize(variant)

    def _initialize(self, variant: str):
        """Load whisper model and processor from HuggingFace."""
        if variant not in WHISPER_VARIANTS:
            raise ValueError(f"Unknown whisper variant: {variant}. Available: {WHISPER_VARIANTS}")

        hf_name = f"openai/whisper-{variant}"
        print(f"Loading Whisper {variant} (CPU)...")
        self._processor = WhisperProcessor.from_pretrained(hf_name)
        self._model = WhisperForConditionalGeneration.from_pretrained(hf_name)
        self._model.eval()
        self._ready = True
        print(f"Whisper {variant} ready (CPU)")

    @property
    def is_ready(self) -> bool:
        return self._ready and self._model is not None

    @property
    def uptime_seconds(self) -> int:
        return int(time.time() - self.start_time)

    def transcribe(self, audio: np.ndarray, language: str = "en") -> tuple[str, int, float]:
        """Transcribe audio numpy array to text.

        Args:
            audio: float32 numpy array normalized to [-1, 1], 16kHz mono
            language: Language code (used for info, model determines capability)

        Returns:
            Tuple of (text, processing_ms, audio_duration_seconds)
        """
        if not self.is_ready:
            raise RuntimeError("Transcriber not ready")

        start_time = time.perf_counter()
        audio_duration_s = round(len(audio) / SAMPLE_RATE, 2)

        with self._inference_lock:
            input_features = self._processor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            ).input_features

            with torch.no_grad():
                generated_ids = self._model.generate(
                    input_features, max_new_tokens=128
                )

            text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

        processing_ms = int((time.perf_counter() - start_time) * 1000)
        return text, processing_ms, audio_duration_s

    def stop(self):
        """Release model resources."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._ready = False
            print(f"Whisper {self.variant} stopped")


# --- Module-level singleton management ---

_transcriber: Optional[WhisperTranscriber] = None
_transcriber_lock = threading.Lock()


def get_transcriber(variant: Optional[str] = None) -> WhisperTranscriber:
    """Get or create the transcriber singleton.

    Args:
        variant: Whisper model variant. If different from current, restarts pipeline.
    """
    global _transcriber

    if variant is None:
        variant = "tiny.en"

    with _transcriber_lock:
        if _transcriber is not None and _transcriber.variant == variant:
            return _transcriber

        # Different variant requested or first init - (re)create
        if _transcriber is not None:
            _transcriber.stop()

        _transcriber = WhisperTranscriber(variant=variant)
        return _transcriber


def close_transcriber():
    """Shut down the transcriber singleton."""
    global _transcriber
    with _transcriber_lock:
        if _transcriber is not None:
            _transcriber.stop()
            _transcriber = None


def list_whisper_models() -> list[str]:
    """Return available whisper variants."""
    return sorted(WHISPER_VARIANTS)


def audio_bytes_to_numpy(audio_bytes: bytes) -> np.ndarray:
    """Convert raw audio bytes (WAV, OGG, MP3, or any ffmpeg-supported format) to float32 numpy array.

    Uses ffmpeg for robust format handling and resampling to 16kHz mono.
    """
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        result = subprocess.run(
            ["ffmpeg", "-i", tmp.name, "-ar", str(SAMPLE_RATE), "-ac", "1",
             "-f", "f32le", "-"],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
        return np.frombuffer(result.stdout, dtype=np.float32)
