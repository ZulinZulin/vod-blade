# Developing VOD BLADE

## Prerequisites

- Python 3.14.x (the packaged release bundles this exact version - see
  `build_release.ps1`'s `$PyVersion`; keep them in sync if you upgrade).
- `ffmpeg`/`ffprobe` on PATH. (The packaged release bundles static binaries instead -
  see "Building a release" below - but a source checkout expects them on PATH, per
  `config.py`'s `FFMPEG_BINARY`/`FFPROBE_BINARY` defaults.)
- [Ollama](https://ollama.com) (optional) - only needed to exercise AI Arbitration
  locally; pull a model with `ollama pull qwen2.5:14b-instruct` (the default) or point
  `LLM_MODEL` in `.env` at something else you've already pulled.

## Setup

```bash
git clone https://github.com/ZulinZulin/StreamCutter.git
cd StreamCutter
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

Two binaries aren't checked into git (see `.gitignore`) and need to be placed under
`bin/` by hand:

- `bin/TwitchDownloaderCLI.exe` - from
  [lay295/TwitchDownloader](https://github.com/lay295/TwitchDownloader)'s releases
  (grab the CLI-only build for Windows).
- `bin/models/yamnet.onnx` - the YAMNet sound-event classification model, ONNX format,
  from [andrelgomes/yamnet-onnx](https://huggingface.co/andrelgomes/yamnet-onnx) on
  Hugging Face.
- `bin/models/yamnet_class_map.csv` - the official class map (521 classes) from
  [tensorflow/models](https://github.com/tensorflow/models) (`research/audioset/yamnet`).

Run it with `run_app.bat`, or directly via `.venv\Scripts\python.exe app.py`. The app
listens on `http://localhost:7863`.

## Repository layout

- `app.py` - Gradio UI and all the button/event wiring.
- `config.py` - every tunable, all `.env`-overridable with sensible defaults.
- `core/` - the actual pipeline: chat/audio/sound-event analysis, LLM arbitration,
  fetchers, session persistence, Ollama setup automation.
- `exporters/` - FCPXML/EDL generation and the DaVinci Resolve scripting bridge.
- `ui/` - CSS and static assets.
- `tests/` - pytest suite.

## Testing packaging changes safely

- `build_release.ps1` only ever reads from the repo root and writes to `dist/`
  (gitignored) - safe to delete and rebuild from scratch at any time.
- Never test the Ollama install/uninstall automation
  (`core/ollama_setup.py`, wired into the Settings panel) against your real Ollama
  setup - use a fresh Windows account or [Windows
  Sandbox](https://learn.microsoft.com/windows/security/application-security/application-isolation/windows-sandbox/windows-sandbox-overview)
  instead, since that's the one part of this app that reaches outside its own folder.
- `VOD_BLADE_DATA_DIR` / `VOD_BLADE_DOWNLOADS_DIR` env vars redirect the small-state
  and downloads directories respectively, without needing a full packaged build - handy
  for testing the path-redirection logic in isolation.

## Building a release

```powershell
.\build_release.ps1 [-Version 0.2.0] [-VendorPath E:\vod-blade-vendor\bin]
```

Downloads an embeddable Python matching the pinned version, installs all dependencies
into it, bundles `ffmpeg`/`ffprobe` (fetched fresh from BtbN's static builds) plus
`TwitchDownloaderCLI.exe`/the YAMNet model (copied from `-VendorPath`, since neither has
a confirmed stable "latest" download URL to fetch automatically), and zips the result
into `dist/VOD-BLADE-v<version>-win64.zip`.

`-VendorPath` defaults to `vendor\bin` next to the script (gitignored) - a folder you
maintain by hand with the same two binaries `bin/` needs for dev (see Setup above).

After building, smoke-test by unzipping the output somewhere outside this repo and
running its `run_app.bat` - it should work with zero dev tools on PATH, since that's
the whole point of bundling everything.
