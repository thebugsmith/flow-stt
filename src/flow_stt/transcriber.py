"""Warm faster-whisper backend on CUDA."""

from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path

import numpy as np

from . import audio as audio_mod
from .config import Config

log = logging.getLogger(__name__)

# Whisper loves to invent these out of silence.
SILENCE_HALLUCINATIONS = {
    "thank you",
    "thank you.",
    "thanks for watching",
    "thanks for watching!",
    "thank you for watching",
    "thank you for watching.",
    "please subscribe",
    "bye",
    "bye.",
    "you",
    ".",
}

SILENT_RMS = 0.004


def _preload_cuda_libs() -> None:
    """dlopen the pip nvidia wheels so CTranslate2 finds cublas/cudnn without
    LD_LIBRARY_PATH. Without this, import/usage dies with libcublas.so.12 missing.
    """
    try:
        import nvidia
    except ImportError:
        return
    # `nvidia` is a namespace package: __file__ is None, so use __path__.
    bases = [Path(p) for p in nvidia.__path__]
    libs = sorted(str(lib) for base in bases for lib in base.glob("*/lib/*.so*"))
    # A few passes: libcudnn_ops needs libcublas and the cudnn core loaded first.
    pending = libs
    for _ in range(4):
        failed = []
        for lib in pending:
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(lib)
        if len(failed) == len(pending):
            break
        pending = failed


class Transcriber:
    def __init__(self, cfg: Config):
        from faster_whisper import WhisperModel

        _preload_cuda_libs()
        self.cfg = cfg
        self._model = None
        for device, compute in self._attempts():
            try:
                self._model = WhisperModel(
                    cfg.model, device=device, compute_type=compute, num_workers=1
                )
                self.device = device
                self.compute_type = compute
                log.info("whisper %s on %s/%s", cfg.model, device, compute)
                break
            except Exception as e:  # RuntimeError on missing cudnn, ValueError on bad quant
                log.warning("%s/%s failed: %s", device, compute, e)
        if self._model is None:
            raise RuntimeError(f"could not load {cfg.model} on gpu or cpu")

    def _attempts(self):
        # int8_float32 = int8 weights, fp32 math. On pre-Ampere GPUs the INT8 GEMM
        # path is ~2x slower than fp32 math over the same int8 weights (measured
        # 1455 ms vs 784 ms on a 1650 Ti), so it is the safer default.
        for compute in (self.cfg.compute_type, "int8_float32", "int8"):
            yield (self.cfg.device, compute)
        yield ("cpu", "int8")

    def transcribe(self, samples: np.ndarray) -> str:
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        assert self._model is not None
        hotwords = ", ".join(self.cfg.vocabulary) or None
        segments, _info = self._model.transcribe(
            audio,
            language=self.cfg.language or None,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            initial_prompt=self.cfg.initial_prompt or None,
            hotwords=hotwords,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()

        if audio_mod.rms(audio) < SILENT_RMS and text.lower().strip() in SILENCE_HALLUCINATIONS:
            return ""
        if text.strip(" .!?,").lower() in {"", "you"}:
            return ""
        return text


def main() -> int:
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = Config()
    print(f"loading {cfg.model} ({cfg.compute_type})...")
    t0 = time.perf_counter()
    tr = Transcriber(cfg)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s on {tr.device}")

    rec = audio_mod.Recorder(min_duration_ms=cfg.min_duration_ms)
    rec.open()
    print("speak now: recording 4s...")
    rec.start()
    time.sleep(4.0)
    samples = rec.stop()
    rec.close()
    print(f"captured {samples.size / rec.sample_rate:.2f}s rms={audio_mod.rms(samples):.4f}")

    t1 = time.perf_counter()
    text = tr.transcribe(samples)
    print(f"stt {1000 * (time.perf_counter() - t1):.0f} ms")
    print(f"transcript: {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
