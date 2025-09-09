"""Tiny in-memory blips so dictation feels tactile (no wav files)."""

from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

RATE = 48000


def _tone(f0: float, f1: float, ms: int, amp: float) -> np.ndarray:
    n = int(RATE * ms / 1000)
    t = np.arange(n) / RATE
    wave = np.sin(2 * np.pi * np.linspace(f0, f1, n) * t)
    fade = max(8, min(n // 4, int(0.005 * RATE)))  # avoid the speaker click
    env = np.ones(n)
    env[:fade] = np.linspace(0.0, 1.0, fade)
    env[-fade:] = np.linspace(1.0, 0.0, fade)
    return (wave * env * amp).astype(np.float32)


START = _tone(660, 990, 40, 0.10)  # rising
STOP = _tone(420, 330, 35, 0.07)   # softer, falling


def play(blip: np.ndarray) -> None:
    try:
        sd.play(blip, samplerate=RATE, blocking=False)
    except Exception as e:  # no output device / PipeWire hiccup: never block dictation
        log.warning("cannot play cue: %s", e)
