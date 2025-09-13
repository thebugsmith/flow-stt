# flow-stt

Push-to-talk dictation for Wayland: hold a key, speak, and the transcript is pasted into
whatever window has focus. Local only — faster-whisper on your GPU, no cloud, no network calls
at runtime.

Measured on a GTX 1650 Ti (4 GB) + 8 GB RAM laptop with `large-v3-turbo` at `int8_float32`:
**~800 ms** from key release to text on screen, ~1.05 GB VRAM resident. See
[Latency](#latency) if you want it faster.

## Requirements

- Arch Linux, Wayland with a **wlroots compositor** (Hyprland, Sway, river…). Our injector uses
  `wtype`, i.e. the `virtual-keyboard` protocol — this does **not** work on GNOME/Mutter.
- NVIDIA GPU, a couple of GB of free VRAM.
- Python 3.11 or 3.12. `ctranslate2` has no wheels for 3.13+/3.14 yet.

## Install

```bash
sudo pacman -S --needed wl-clipboard wtype portaudio python

# read /dev/input (keyboard) and create virtual devices for the test harness
sudo usermod -aG input,uinput "$USER"      # then log out and back in

git clone <this repo> ~/flow-stt && cd ~/flow-stt
python3.11 -m venv .venv                   # or: ~/.pyenv/versions/3.11.15/bin/python3 -m venv .venv
.venv/bin/pip install -e ".[cuda]"         # .[cuda] pulls nvidia-cublas-cu12 + nvidia-cudnn-cu12

mkdir -p ~/.config/flow-stt
cp config.example.toml ~/.config/flow-stt/config.toml
```

`.[cuda]` matters: CTranslate2 dlopens `libcublas.so.12`/`libcudnn_ops.so.9` from those wheels and
does not declare them itself. Without them the model loads and the first transcription dies with
`Library libcublas.so.12 is not found`.

First start downloads ~1.6 GB of weights to `~/.cache/huggingface`; after that it is offline.

## Use

```bash
.venv/bin/flow-stt --check     # cuda / model / input / wayland / mic
.venv/bin/flow-stt             # run in the foreground
```

Hold **RightAlt**, speak, release. Two blips and the text appears at the cursor. Ctrl+C or
`SIGTERM` shuts down cleanly.

`--check` output on a working machine:

```
  ok    cuda     1 device(s), int8_float32 supported
  ok    model    large-v3-turbo cached
  ok    input    /dev/input/event20 (kanata), /dev/input/event4 (AT Translated Set 2 keyboard)
  ok    wayland  wayland-1, wl-clipboard + wtype present
  ok    mic      alsa_input.pci-0000_00_1f.3.analog-stereo @ 44100 Hz
```

## Configuration

`~/.config/flow-stt/config.toml` — every key optional, unknown keys are rejected loudly.

| key | default | notes |
|---|---|---|
| `model` | `large-v3-turbo` | any faster-whisper model name |
| `device` | `cuda` | falls back to cpu if CUDA fails |
| `compute_type` | `int8_float32` | see [Latency](#latency) before changing |
| `language` | `en` | `""` = autodetect |
| `hotkey` | `KEY_RIGHTALT` | evdev name |
| `cleanup_hotkey` | `KEY_RIGHTCTRL` | second key; identical to `hotkey` until the LLM cleanup stage is added |
| `paste_method` | `ctrl+v` | `ctrl+shift+v` for terminals, `type` to type text without touching the clipboard |
| `paste_delay_ms` | `150` | wait before restoring the old clipboard |
| `min_duration_ms` | `250` | shorter holds are ignored as accidental taps |
| `vocabulary` | `[]` | names/jargon fed to Whisper as prompt + hotwords |

## Run as a service

User unit (keeps the model warm, so there is no cold start on the first press):

```bash
mkdir -p ~/.config/systemd/user
cp deploy/flow-stt.service ~/.config/systemd/user/
# edit ExecStart/WorkingDirectory/LD_LIBRARY_PATH if the repo is not at ~/flow-stt
systemctl --user daemon-reload
systemctl --user enable --now flow-stt
```

A user service inherits no session environment, so the daemon resolves the Wayland socket itself
(first `/run/user/$UID/wayland-?`) every time it injects. If `wl-copy` still cannot connect,
import the variable and restart the unit:

```bash
# ~/.config/hypr/hyprland.conf (or your compositor's equivalent)
exec-once = systemctl --user import-environment WAYLAND_DISPLAY
exec-once = systemctl --user restart flow-stt
```

Check it took:

```bash
journalctl --user -u flow-stt -n 20
```

Optional udev rule so group `uinput` persists across reboots:

```bash
sudo cp deploy/99-uinput.rules /etc/udev/rules.d/ && sudo udevadm control --reload
```

## Latency

Whisper pads every clip to a 30 s encoder window, so STT cost is flat: 816 ms for a 2 s
clip, 883 ms for an 11 s one. Compute type is the single biggest lever, and the intuitive choice is
wrong on older cards — measured on the 1650 Ti (Turing), same 1048 MiB VRAM in all three cases:

| `compute_type` | 3 s clip |
|---|---|
| `int8_float32` | **784 ms** |
| `int8` | 1455 ms |
| `int8_float16` | 1461 ms |
| `float16` | ~6000 ms |
| `float32` | OOM on 4 GB |

All three "int8" variants store int8 weights; only the math precision differs. Pre-Ampere GPUs have
no fast INT8 GEMM path in CTranslate2, so `int8`/`int8_float16` are the slow options there. Ampere
or newer: `int8_float16` is the usual best pick — benchmark it.

To go under ~500 ms the model has to change: `small.en` measures 338 ms on the same clip, at an
accuracy cost that matters more for jargon than for plain dictation.

## Troubleshooting

**`Library libcublas.so.12 is not found`** — the nvidia wheels are missing
(`pip install -e ".[cuda]"`), or you are running under systemd without the `LD_LIBRARY_PATH` line
in the unit.

**`no permission to read /dev/input`** — `sudo usermod -aG input $USER`, then log out and back in;
`id` must show `input`.

**Hotkey does nothing, but keys work elsewhere** — a remapper (kanata, keyd) has grabbed the
physical keyboard, so events only appear on its virtual device. flow-stt listens to every keyboard
it can find (that is what the `kanata` line in `--check` is), but the remapper must forward the key:
keep `process-unmapped-keys yes` and do not map `ralt` in `defsrc`.

**Nothing pastes, but text appears in the clipboard** — `wtype` could not type into the focused
window: check `WAYLAND_DISPLAY` is set, and that you are not on GNOME. Some terminals need
`paste_method = "ctrl+shift+v"`.

**First press is slow, later ones are fast** — the model was not preloaded; run under the systemd
unit instead of starting the daemon by hand.

## Notes

- Audio stays in RAM: no temp files, no disk writes.
- The model stays resident; `Restart=on-failure` covers GPU driver hiccups.
- Optional LLM cleanup (`llama-server`) is not wired up yet — `cleanup_hotkey` currently behaves
  like `hotkey`.
