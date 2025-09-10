"""`flow-stt --check`: verify everything the daemon needs, without starting it."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess

import evdev

from .config import Config
from .hotkey import is_keyboard


def _cuda(cfg: Config) -> tuple[bool, str]:
    try:
        import ctranslate2

        from .transcriber import _preload_cuda_libs

        _preload_cuda_libs()
        count = ctranslate2.get_cuda_device_count()
        if count == 0:
            return False, "CTranslate2 sees no CUDA device"
        types = ctranslate2.get_supported_compute_types("cuda")
        if cfg.compute_type not in types:
            return False, f"{cfg.compute_type} unsupported here, have {sorted(types)}"
        return True, f"{count} device(s), {cfg.compute_type} supported"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _input(_cfg: Config) -> tuple[bool, str]:
    seen = []
    denied = False
    for path in sorted(glob.glob("/dev/input/event*")):
        try:
            dev = evdev.InputDevice(path)
        except PermissionError:
            denied = True
            continue
        except OSError:
            continue
        if is_keyboard(dev):
            seen.append(f"{path} ({dev.name})")
        dev.close()
    if seen:
        return True, ", ".join(seen)
    if denied:
        return False, "permission denied — sudo usermod -aG input $USER, then log out/in"
    return False, "no keyboard readable under /dev/input"


def _wayland(_cfg: Config) -> tuple[bool, str]:
    display = os.environ.get("WAYLAND_DISPLAY")
    if display:
        if not os.path.exists(f"/run/user/{os.getuid()}/{display}"):
            return False, f"WAYLAND_DISPLAY={display} but that socket does not exist"
    else:
        found = sorted(glob.glob(f"/run/user/{os.getuid()}/wayland-?"))
        if not found:
            return False, "no Wayland socket — injector needs a Wayland session"
        display = os.path.basename(found[0])
    missing = [t for t in ("wl-copy", "wl-paste", "wtype") if shutil.which(t) is None]
    if missing:
        return False, f"missing {', '.join(missing)} — pacman -S wl-clipboard wtype"
    return True, f"{display}, wl-clipboard + wtype present"


def _mic(_cfg: Config) -> tuple[bool, str]:
    try:
        import sounddevice as sd

        sd.check_input_settings(samplerate=16000, channels=1, dtype="float32")
        dev = sd.query_devices(kind="input")
        name = dev["name"]
        try:  # PipeWire's real default source, not the "default" alias
            src = subprocess.run(["pactl", "get-default-source"], capture_output=True,
                                 text=True, timeout=2).stdout.strip()
            name = src or name
        except Exception:
            pass
        return True, f"{name} @ {int(dev['default_samplerate'])} Hz"
    except Exception as e:
        return False, str(e)


def _model(cfg: Config) -> tuple[bool, str]:
    hits = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--*{cfg.model}*"))
    if hits:
        return True, f"{cfg.model} cached"
    return False, f"{cfg.model} not cached — first start downloads it (needs network)"


CHECKS = (
    ("cuda", _cuda),
    ("model", _model),
    ("input", _input),
    ("wayland", _wayland),
    ("mic", _mic),
)


def run(cfg: Config) -> bool:
    print(f"flow-stt check ({cfg.model} on {cfg.device}/{cfg.compute_type})\n")
    failed = 0
    for name, fn in CHECKS:
        ok, detail = fn(cfg)
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:8s} {detail}")
        failed += 0 if ok else 1
    print()
    if failed:
        print(f"{failed} check(s) failed")
    return failed == 0
