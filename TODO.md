# TODO

## Optional local LLM cleanup stage (llama-server)

`cleanup_hotkey` (RightCtrl) is currently accepted and logged but does nothing — both keys paste the
raw transcript. The goal is a second hotkey that runs the transcript through a local
`llama-server` (`http://127.0.0.1:8080/v1/chat/completions`) to strip filler words, resolve
mid-sentence self-corrections and fix punctuation.

- [ ] `src/flow_stt/cleaner.py` using stdlib `urllib.request` — no new dependencies
- [ ] tight system prompt: remove "um"/"uh"/"like", apply self-corrections, fix punctuation and
      capitalisation, return only the cleaned text
- [ ] 2.5 s timeout, silent fallback to the raw transcript so typing is never blocked
- [ ] wire `mode == "clean"` and `always_clean = true` in `Daemon._handle`
- [ ] README: `llama-server` invocation with `-ngl` tuned to leave room for Whisper in 4 GB VRAM
