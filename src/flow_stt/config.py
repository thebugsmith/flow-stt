"""Config loading for flow-stt (~/.config/flow-stt/config.toml)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

DEFAULT_PATH = Path("~/.config/flow-stt/config.toml").expanduser()


@dataclass
class Config:
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "int8_float32"
    language: str = "en"

    # evdev key names, e.g. KEY_RIGHTALT
    hotkey: str = "KEY_RIGHTALT"
    cleanup_hotkey: str = "KEY_RIGHTCTRL"
    always_clean: bool = False

    # paste_method: "ctrl+v" | "ctrl+shift+v" | "type"
    paste_method: str = "ctrl+v"
    paste_delay_ms: int = 150

    min_duration_ms: int = 250

    llama_url: str = "http://127.0.0.1:8080/v1/chat/completions"
    llama_model: str = "local"
    llama_timeout: float = 2.5

    vocabulary: list[str] = field(default_factory=list)

    @property
    def initial_prompt(self) -> str:
        if not self.vocabulary:
            return ""
        return "Glossary: " + ", ".join(self.vocabulary) + "."


def load(path: Path | str | None = None) -> Config:
    path = Path(os.environ.get("FLOW_STT_CONFIG", path or DEFAULT_PATH)).expanduser()
    cfg = Config()
    if not path.exists():
        return cfg

    with path.open("rb") as f:
        data = tomllib.load(f)

    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys in {path}: {', '.join(sorted(unknown))}")

    for key, value in data.items():
        setattr(cfg, key, value)
    return cfg
