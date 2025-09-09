"""Global push-to-talk hotkey listener over /dev/input (compositor-agnostic)."""

from __future__ import annotations

import glob
import logging
import os
import threading
import time

import evdev
from evdev import ecodes

from .config import Config

log = logging.getLogger(__name__)

# Keyboards expose dozens of keys; mice, power/sleep buttons and lid switches a handful.
# (A REL axis does not disqualify a device: kanata and other remappers create
# virtual keyboards that also advertise REL for mouse emulation.)
MIN_KEYS = 40
RESCAN_S = 2.0


def is_keyboard(dev: evdev.InputDevice) -> bool:
    caps = dev.capabilities()
    if len(caps.get(ecodes.EV_KEY, [])) < MIN_KEYS:
        return False
    if ecodes.ABS_MT_POSITION_X in caps.get(ecodes.EV_ABS, []):
        return False
    return True


class HotkeyListener:
    def __init__(self, cfg: Config, on_press, on_release):
        self.cfg = cfg
        self.on_press = on_press
        self.on_release = on_release
        self.codes = {
            self._code(cfg.hotkey): "raw",
            self._code(cfg.cleanup_hotkey): "clean",
        }
        self._active: set[str] = set()
        self._devices: dict[str, evdev.InputDevice] = {}
        self._stop = threading.Event()
        self._last_scan = 0.0

    @staticmethod
    def _code(name: str) -> int:
        try:
            return ecodes.ecodes[name]
        except KeyError:
            raise ValueError(f"unknown key name {name!r} — use names like KEY_RIGHTALT") from None

    def scan(self) -> None:
        for path in sorted(glob.glob("/dev/input/event*")):
            if path in self._devices:
                continue
            try:
                dev = evdev.InputDevice(path)
            except PermissionError as e:
                raise RuntimeError(
                    "no permission to read /dev/input — run: sudo usermod -aG input $USER "
                    "then log out and back in (or reboot)."
                ) from e
            except OSError:
                continue
            if not is_keyboard(dev):
                dev.close()
                continue
            self._devices[path] = dev
            log.info("listening on %s (%s)", path, dev.name)
        self._last_scan = time.monotonic()

    def _handle(self, ev) -> None:
        mode = self.codes.get(ev.code)
        if mode is None or ev.value == 2:  # 2 = autorepeat
            return
        if ev.value == 1:
            if mode in self._active:  # same key from a second device
                return
            self._active.add(mode)
            self.on_press(mode)
        elif mode in self._active:
            self._active.discard(mode)
            self.on_release(mode)

    def run(self) -> None:
        import selectors

        self.scan()
        if not self._devices:
            raise RuntimeError("no keyboard found under /dev/input — is the input group set?")
        sel = selectors.DefaultSelector()
        for dev in self._devices.values():
            sel.register(dev, selectors.EVENT_READ, dev.path)

        while not self._stop.is_set():
            for key, _ in sel.select(timeout=0.5):
                path = key.data
                dev = self._devices.get(path)
                if dev is None:
                    continue
                try:
                    # read() is a lazy generator in evdev 1.9: pull it inside the try,
                    # otherwise a disappearing device raises mid-iteration and kills the loop.
                    events = list(dev.read())
                except OSError:
                    log.warning("lost %s (%s), will rescan", path, dev.name)
                    sel.unregister(dev)
                    del self._devices[path]
                    dev.close()
                    continue
                for ev in events:
                    if os.environ.get("FLOW_STT_DEBUG"):
                        print(f"  raw {path} type={ev.type} code={ev.code} value={ev.value}", flush=True)
                    if ev.type == ecodes.EV_KEY:
                        self._handle(ev)
            if time.monotonic() - self._last_scan > RESCAN_S:
                before = set(self._devices)
                try:
                    self.scan()
                except RuntimeError as e:
                    log.error("%s", e)
                for path in set(self._devices) - before:
                    sel.register(self._devices[path], selectors.EVENT_READ, path)

    def stop(self) -> None:
        self._stop.set()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = Config()
    print(f"hotkeys: {cfg.hotkey} -> raw, {cfg.cleanup_hotkey} -> clean.  Ctrl+C to quit.")

    listener = HotkeyListener(
        cfg,
        on_press=lambda m: print(f"press {m}", flush=True),
        on_release=lambda m: print(f"release {m}", flush=True),
    )
    try:
        listener.run()
    except KeyboardInterrupt:
        listener.stop()
