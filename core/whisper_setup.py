"""
core/whisper_setup.py

Optional, opt-in local transcription setup: detects whether the whisper.cpp CLI
binary and a chosen ggml model are present, and can download/remove them on the
user's behalf so "Generate transcript locally" doesn't require knowing what
whisper.cpp even is. Mirrors core/ollama_setup.py's "live detection, never a
stored flag" philosophy - every check here hits the actual filesystem each call.

Simpler than the Ollama precedent in one respect: whisper-cli has no installer,
no daemon, no registry uninstall entry - it's a standalone binary dropped in
bin/, exactly like TwitchDownloaderCLI.exe/ffmpeg.exe already are. So there's no
INSTALLED_NOT_RUNNING-equivalent state, and "remove" is just deleting a file.

Nothing here runs automatically; app.py only calls these from explicit button
clicks. A bug in the automated path should degrade to "get the binary/model
yourself from GitHub/Hugging Face" (always shown alongside), never strand the
user.
"""

from __future__ import annotations

import logging
import platform
import zipfile
from enum import Enum, auto
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import requests

from config import BIN_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

# Multilingual tiers only - deliberately NOT the ".en" variants most whisper.cpp
# tutorials default to. See config.WhisperConfig's docstring: this project's real
# content is Russian-language, and an English-only model would silently
# mistranscribe it, not just underperform.
MODEL_TIERS: List[Tuple[str, str]] = [
    ("tiny", "~75 MB - fastest, roughest"),
    ("base", "~142 MB"),
    ("small", "~466 MB - recommended default"),
    ("medium", "~1.5 GB"),
    ("large-v3", "~2.9 GB - most accurate"),
]

# NEEDS RE-VERIFICATION against the current release before this ships - whisper.cpp's
# Windows asset naming has changed before (e.g. CPU vs CUDA/Vulkan build variants),
# same caveat build_release.ps1 already carries for BtbN's ffmpeg asset name.
_WHISPER_RELEASE_ZIP_URL = "https://github.com/ggml-org/whisper.cpp/releases/latest/download/whisper-bin-x64.zip"
_HF_MODEL_URL_TMPL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"


class WhisperState(Enum):
    BINARY_MISSING = auto()
    MODEL_MISSING = auto()
    READY = auto()


class WhisperSetupError(Exception):
    """Raised for setup actions (install/download-model/remove) that fail outright -
    never for "not present yet", which is a normal detect_state() result."""


def model_path_for(model_name: str, models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / f"ggml-{model_name}.bin"


def is_binary_present(binary_path: Path) -> bool:
    return binary_path.exists()


def is_model_present(model_name: str, models_dir: Path = MODELS_DIR) -> bool:
    return model_path_for(model_name, models_dir).exists()


def detect_state(binary_path: Path, model_name: str, models_dir: Path = MODELS_DIR) -> WhisperState:
    if not is_binary_present(binary_path):
        return WhisperState.BINARY_MISSING
    if not is_model_present(model_name, models_dir):
        return WhisperState.MODEL_MISSING
    return WhisperState.READY


def check_gpu_vram_mb() -> Optional[int]:
    """Same nvidia-smi query as core.ollama_setup.check_gpu_vram_mb - reused
    directly rather than duplicated, since it's pure GPU-detection with nothing
    Ollama-specific about it."""
    from core.ollama_setup import check_gpu_vram_mb as _check_gpu_vram_mb
    return _check_gpu_vram_mb()


def download_whisper_release_zip(dest: Path) -> Iterator[dict]:
    """Streams the official whisper.cpp Windows release zip to `dest`, yielding
    {'downloaded_mb': ..., 'total_mb': ... | None, 'percent': ... | None} as it
    goes - same shape as ollama_setup.download_ollama_installer."""
    response = requests.get(_WHISPER_RELEASE_ZIP_URL, stream=True, timeout=30)
    response.raise_for_status()
    total = response.headers.get("Content-Length")
    total_bytes = int(total) if total else None

    downloaded = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            yield {
                "downloaded_mb": downloaded / (1024 * 1024),
                "total_mb": (total_bytes / (1024 * 1024)) if total_bytes else None,
                "percent": (downloaded / total_bytes * 100) if total_bytes else None,
            }


def install_whisper_binary(zip_path: Path, bin_dir: Path = BIN_DIR) -> Path:
    """
    Extracts whisper-cli(.exe) AND every DLL sitting alongside it in the archive.
    Unlike ffmpeg's single static exe, whisper.cpp's official Windows release ships
    the CLI dynamically linked against several DLLs (ggml*.dll, whisper.dll,
    possibly SDL2.dll depending on build variant) - all of them have to land next
    to the exe or it fails to launch at all. This is exactly the class of bug this
    project already hit once with TwitchDownloaderCLI/ffmpeg ("Unable to find
    FFmpeg" - see config._default_ffmpeg_binary's docstring) - the exact zip
    layout needs re-confirming against the current release before this ships.
    """
    exe_name = "whisper-cli.exe" if platform.system() == "Windows" else "whisper-cli"
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        exe_members = [n for n in names if n.lower().endswith(exe_name.lower())]
        if not exe_members:
            raise WhisperSetupError(
                f"'{exe_name}' not found inside the downloaded whisper.cpp release archive - "
                "its layout may have changed since this was written."
            )
        exe_dir_in_zip = str(Path(exe_members[0]).parent)
        to_extract = [
            n for n in names
            if str(Path(n).parent) == exe_dir_in_zip and Path(n).suffix.lower() in {".exe", ".dll"}
        ]
        bin_dir.mkdir(parents=True, exist_ok=True)
        for member in to_extract:
            (bin_dir / Path(member).name).write_bytes(zf.read(member))

    result = bin_dir / exe_name
    if not result.exists():
        raise WhisperSetupError(f"Extraction completed but '{result}' is still missing.")
    return result


def download_model(model_name: str, dest: Path) -> Iterator[dict]:
    """Streams ggml-{model_name}.bin from Hugging Face - same streaming shape as
    download_whisper_release_zip. Written to a .part file and only renamed onto
    `dest` once the download completes fully, so an interrupted download (browser
    closed, network drop) can never leave a corrupt file that is_model_present()
    would wrongly report as ready."""
    url = _HF_MODEL_URL_TMPL.format(name=model_name)
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = response.headers.get("Content-Length")
    total_bytes = int(total) if total else None

    downloaded = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    with open(tmp_dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            yield {
                "downloaded_mb": downloaded / (1024 * 1024),
                "total_mb": (total_bytes / (1024 * 1024)) if total_bytes else None,
                "percent": (downloaded / total_bytes * 100) if total_bytes else None,
            }
    tmp_dest.replace(dest)


def remove_model(model_name: str, models_dir: Path = MODELS_DIR) -> None:
    model_path_for(model_name, models_dir).unlink(missing_ok=True)


def remove_binary(binary_path: Path) -> None:
    binary_path.unlink(missing_ok=True)
