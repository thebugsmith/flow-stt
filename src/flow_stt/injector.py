"""Wayland text injection: put the transcript on the clipboard, send Ctrl+V, restore.

Uses wl-copy/wl-paste + wtype (virtual-keyboard protocol — works on wlroots
compositors: Hyprland, Sway, river. Mutter/GNOME does not expose it).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time

from .config import Config

log = logging.getLogger(__name__)

# wtype key sequences per paste_method
PASTE_KEYS = {
    "ctrl+v": ["-M", "ctrl", "-k", "v", "-m", "ctrl"],
    "ctrl+shift+v": ["-M", "ctrl", "-M", "shift", "-k", "v", "-m", "shift", "-m", "ctrl"],
}


def _run(args: list[str], data: bytes | None = None, timeout: float = 5.0,
         capture: bool = True) -> subprocess.CompletedProcess:
    if capture:
        return subprocess.run(args, input=data, capture_output=True, timeout=timeout)
    # wl-copy forks a daemon that inherits our pipes, so capturing its output
    # hangs until the selection owner dies. Send it to /dev/null instead.
    return subprocess.run(args, input=data, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, timeout=timeout)


def read_clipboard() -> bytes | None:
    """Current clipboard text, or None when it is empty or holds non-text data."""
    types = _run(["wl-paste", "--list-types"])
    if types.returncode != 0:
        return None
    if b"text/" not in types.stdout and b"UTF8" not in types.stdout:
        log.info("clipboard holds non-text data, leaving it alone")
        return None
    got = _run(["wl-paste", "--no-newline"])
    if got.returncode != 0:
        return None
    return got.stdout


def write_clipboard(data: bytes) -> None:
    if data:
        _run(["wl-copy"], data=data, capture=False)
    else:
        # wl-copy keeps the old selection alive if fed empty stdin
        _run(["wl-copy", "--clear"], capture=False)


def inject(text: str, cfg: Config) -> bool:
    if not text.strip():
        return False
    for tool in ("wl-copy", "wl-paste", "wtype"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found — pacman -S wl-clipboard wtype")

    if cfg.paste_method == "type":
        _run(["wtype", "--", text])
        return True
    if cfg.paste_method not in PASTE_KEYS:
        raise ValueError(f"unknown paste_method {cfg.paste_method!r}")

    previous = read_clipboard()
    write_clipboard(text.encode())
    # wl-copy forks a daemon that owns the selection; give it a moment before the paste
    time.sleep(0.04)
    _run(["wtype", *PASTE_KEYS[cfg.paste_method]])
    time.sleep(cfg.paste_delay_ms / 1000)
    if previous is not None:
        write_clipboard(previous)
    return True


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("usage: python -m flow_stt.injector <text>   # focus a text field first")
        raise SystemExit(2)
    cfg = Config()
    print(f"injecting via {cfg.paste_method}: {sys.argv[1]!r}")
    inject(" ".join(sys.argv[1:]), cfg)
