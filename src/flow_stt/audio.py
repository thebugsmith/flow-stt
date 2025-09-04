"""PipeWire mic capture straight into RAM (16 kHz mono float32)."""

from __future__ import annotations

import threading
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000


class Recorder:
    """Keeps one InputStream open; start/stop only gate collection.

    Opening a stream per utterance loses the first ~50 ms of speech while
    PipeWire connects, so the stream is opened once and left running.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, min_duration_ms: int = 250):
        self.sample_rate = sample_rate
        self.min_duration_ms = min_duration_ms
        self._chunks: list[np.ndarray] = []
        self._recording = False
        self._stream: sd.InputStream | None = None
        self._last_cb = 0.0
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        self._last_cb = time.monotonic()
        if not self._recording:
            return
        # copy: PortAudio reuses the buffer between callbacks
        with self._lock:
            self._chunks.append(indata[:, 0].copy())

    def open(self) -> None:
        """Open the stream and wait until PipeWire actually delivers samples."""
        if self._stream is not None:
            return
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            # 10 ms blocks: a 64 ms block would drop that much off the start and
            # end of every hold, since PortAudio only hands over filled blocks.
            blocksize=160,
        )
        stream.start()
        self._stream = stream
        deadline = time.monotonic() + 1.0
        while self._last_cb == 0.0 and time.monotonic() < deadline:
            time.sleep(0.005)

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    def start(self) -> None:
        self.open()
        with self._lock:
            self._chunks = []
        self._recording = True

    def stop(self) -> np.ndarray:
        self._recording = False
        with self._lock:
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def too_short(self, audio: np.ndarray) -> bool:
        return audio.size < self.sample_rate * self.min_duration_ms // 1000


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


if __name__ == "__main__":
    rec = Recorder()
    print("opening mic...")
    rec.open()
    print("recording 3s...")
    t0 = time.monotonic()
    rec.start()
    time.sleep(3.0)
    audio = rec.stop()
    elapsed = time.monotonic() - t0

    peak = float(np.abs(audio).max()) if audio.size else 0.0
    print(f"captured {audio.size} samples = {audio.size / rec.sample_rate:.3f}s "
          f"(wall {elapsed:.3f}s, peak {peak:.3f})")
    print(f"rms {rms(audio):.4f}  too_short={rec.too_short(audio)}")
    rec.close()
