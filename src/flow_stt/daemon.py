"""Main daemon: hotkey -> record -> transcribe -> paste, model kept warm in VRAM."""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import sys
import threading
import time

import numpy as np

from . import audio, cues, injector
from .config import Config, load
from .hotkey import HotkeyListener
from .transcriber import Transcriber

log = logging.getLogger("flow_stt")


class Daemon:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rec = audio.Recorder(min_duration_ms=cfg.min_duration_ms)
        self.tr: Transcriber | None = None
        self.listener: HotkeyListener | None = None
        self.jobs: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self._recording = False
        self._busy = False
        self._lock = threading.Lock()

    def on_press(self, mode: str) -> None:
        with self._lock:
            if self._recording or self._busy:
                log.info("ignoring %s: previous utterance still in flight", mode)
                return
            self._recording = True
            self._mode = mode
        cues.play(cues.START)  # 40 ms, below VAD's min speech duration, so it never transcribes
        self.rec.start()

    def on_release(self, mode: str) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False
        released = time.monotonic()
        samples = self.rec.stop()
        cues.play(cues.STOP)
        if self.rec.too_short(samples):
            log.info("ignored %.0f ms tap", 1000 * samples.size / self.rec.sample_rate)
            return
        with self._lock:
            self._busy = True
        self.jobs.put((samples, mode, released))

    def _worker(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            samples, mode, released = job
            try:
                self._handle(samples, mode, released)
            except Exception:
                log.exception("utterance failed")
            finally:
                with self._lock:
                    self._busy = False

    def _handle(self, samples: np.ndarray, mode: str, released: float) -> None:
        assert self.tr is not None
        dur = samples.size / self.rec.sample_rate
        t0 = time.monotonic()
        text = self.tr.transcribe(samples)
        stt_ms = 1000 * (time.monotonic() - t0)
        if not text:
            log.info("audio: %.1fs | stt: %.0f ms | empty transcript", dur, stt_ms)
            return
        injector.inject(text, self.cfg)
        total_ms = 1000 * (time.monotonic() - released)
        log.info("audio: %.1fs | stt: %.0f ms | total: %.0f ms | %s",
                 dur, stt_ms, total_ms, text)

    def run(self) -> int:
        log.info("loading %s (%s)...", self.cfg.model, self.cfg.compute_type)
        self.tr = Transcriber(self.cfg)
        log.info("model ready on %s/%s", self.tr.device, self.tr.compute_type)

        self.rec.open()
        self.listener = HotkeyListener(self.cfg, self.on_press, self.on_release)
        threading.Thread(target=self._worker, name="stt", daemon=True).start()

        def shutdown(signum, _frame):
            log.info("signal %d, shutting down", signum)
            self.stop.set()
            if self.listener:
                self.listener.stop()

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        log.info("listening: hold %s to dictate, %s for cleanup", self.cfg.hotkey, self.cfg.cleanup_hotkey)
        try:
            self.listener.run()
        finally:
            self.rec.close()
            self.jobs.put(None)
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flow-stt", description="push-to-talk dictation daemon")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("faster_whisper", "httpx2", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load()
    try:
        return Daemon(cfg).run()
    except RuntimeError as e:
        log.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
