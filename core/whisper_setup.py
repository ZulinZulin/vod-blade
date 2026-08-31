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
import os
import platform
import shutil
import subprocess
import zipfile
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

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

# Rough VRAM needed to run each tier on the GPU build, in MB. The model weights
# dominate (measured: 'small' reported "CUDA0 total size = 487.01 MB", matching its
# file size), so these are the weight sizes plus ~35% headroom for activations and
# the compute buffer. Deliberately approximate and deliberately generous - this only
# ever drives a non-blocking warning, so over-warning slightly is much cheaper than
# letting someone start a multi-hour transcription that dies on an OOM partway.
_MODEL_VRAM_MB: Dict[str, int] = {
    "tiny": 150,
    "base": 250,
    "small": 700,
    "medium": 2100,
    "large-v3": 4000,
}

# Pinned to an exact tag rather than /releases/latest on purpose. The CPU build is
# bundled into the release zip by build_release.ps1 at build time, while a CUDA build
# is downloaded by the user later - possibly months later. Under "latest" those two
# could be different whisper.cpp versions with incompatible ggml backend ABIs, which
# with GGML_BACKEND_DL (see WhisperVariant) is a crash path, not a cosmetic mismatch.
# Bumping this is a deliberate, reviewable act; keep it in sync with build_release.ps1.
_WHISPER_RELEASE_TAG = "b4938"  # whisper.cpp 1.9.3
_RELEASE_BASE_URL = f"https://github.com/ggml-org/whisper.cpp/releases/download/{_WHISPER_RELEASE_TAG}"
_HF_MODEL_URL_TMPL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"


class WhisperVariant(Enum):
    """
    Which whisper.cpp build to use. These are NOT interchangeable in one directory:
    both ship same-named-but-different ggml.dll / whisper.dll / ggml-base.dll /
    ggml-cpu-*.dll, so they get separate install dirs (see binary_path_for).

    The releases are built with -DGGML_BACKEND_DL=ON, meaning backends are
    LoadLibrary'd at runtime by scanning for ggml-*.dll next to the exe - and a
    backend that fails to load is logged and *swallowed*, with the process quietly
    continuing on CPU. So merely having the CUDA build installed proves nothing about
    whether it's actually being used; core.transcriber watches the output stream for
    the real confirmation. Measured on an RTX 5070 Ti: ~7.6x faster than CPU once warm.
    """

    CPU = auto()
    CUDA = auto()


_RELEASE_ZIP_NAMES = {
    WhisperVariant.CPU: "whisper-bin-x64.zip",
    # CUDA 12.4, deliberately not the smaller 11.8 build: both embed the identical
    # ggml PTX (90-virtual), so 11.8's only real difference is an older bundled cuBLAS -
    # i.e. it optimizes against the exact component that carries the compatibility risk
    # on newer GPUs. Upstream's own CI also passes -allow-unsupported-compiler only for
    # the 11.8 job, i.e. treats that toolchain as unsupported.
    WhisperVariant.CUDA: "whisper-cublas-12.4.0-bin-x64.zip",
}
_RELEASE_ZIP_URLS = {v: f"{_RELEASE_BASE_URL}/{name}" for v, name in _RELEASE_ZIP_NAMES.items()}

# Rough on-disk cost of each variant, for preflight checks and honest UI copy.
_VARIANT_DOWNLOAD_MB = {WhisperVariant.CPU: 8, WhisperVariant.CUDA: 640}
_VARIANT_EXTRACTED_MB = {WhisperVariant.CPU: 30, WhisperVariant.CUDA: 1150}

CUDA_BIN_DIR = BIN_DIR / "whisper-cuda"


class WhisperState(Enum):
    BINARY_MISSING = auto()
    MODEL_MISSING = auto()
    READY = auto()


class WhisperSetupError(Exception):
    """Raised for setup actions (install/download-model/remove) that fail outright -
    never for "not present yet", which is a normal detect_state() result."""


def model_path_for(model_name: str, models_dir: Path = MODELS_DIR) -> Path:
    """Models are variant-independent - a ggml-*.bin works with the CPU and CUDA
    builds alike, so switching variants never re-downloads a multi-GB model."""
    return models_dir / f"ggml-{model_name}.bin"


def _exe_name() -> str:
    return "whisper-cli.exe" if platform.system() == "Windows" else "whisper-cli"


def binary_path_for(variant: WhisperVariant) -> Path:
    return (CUDA_BIN_DIR if variant is WhisperVariant.CUDA else BIN_DIR) / _exe_name()


def is_cuda_binary(binary_path: Path) -> bool:
    """Whether `binary_path` is the GPU build - i.e. lives under CUDA_BIN_DIR. Used to
    decide GPU-only command-line flags (see config.WhisperConfig.flash_attention)
    without assuming the caller went through active_binary_path()."""
    try:
        return CUDA_BIN_DIR.resolve() in binary_path.resolve().parents
    except OSError:
        return False


def active_variant() -> WhisperVariant:
    """
    Presence-driven, deliberately NOT a stored preference: whichever build is
    installed is the one that runs. This matches this module's "live detection,
    never a stored flag" contract - it stays correct when the user adds or deletes
    an install behind our back, and there's no way for a saved setting to disagree
    with what's actually on disk. If you want to force CPU while the CUDA build is
    installed, that's a *runtime* choice (core.transcriber's use_gpu -> -ng), not a
    reinstall.
    """
    return WhisperVariant.CUDA if binary_path_for(WhisperVariant.CUDA).exists() else WhisperVariant.CPU


def active_binary_path() -> Path:
    """
    The whisper-cli that should actually run. An explicitly-set WHISPER_CLI_PATH wins
    outright (it's the power-user escape hatch - e.g. a hand-built GPU binary); after
    that it's whatever active_variant() resolves to.

    Note there is deliberately no fallback from "CUDA selected but missing" to the CPU
    path: silently substituting a different backend is precisely the failure mode this
    whole feature guards against. If the CUDA dir vanishes, active_variant() simply
    reports CPU again and detect_state() speaks for itself.
    """
    if os.getenv("WHISPER_CLI_PATH"):
        return Path(os.environ["WHISPER_CLI_PATH"])
    return binary_path_for(active_variant())


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


def check_gpu_info() -> Optional[dict]:
    """
    One nvidia-smi call describing the first NVIDIA GPU, or None if there isn't one
    (AMD/Intel/no dGPU) or anything about the call fails - callers should read None as
    "couldn't determine", not "no GPU". Returns
    {name, compute_cap, total_mb, free_mb, driver}.

    free_mb matters as much as total here: this app can have a ~9GB Ollama model
    resident on the same card during AI Arbitration, so total VRAM alone would
    overstate what's actually available to a large Whisper model.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, check=True,
        )
        name, cap, total, free, driver = (p.strip() for p in result.stdout.strip().splitlines()[0].split(","))
        return {
            "name": name,
            "compute_cap": cap,
            "total_mb": int(total),
            "free_mb": int(free),
            "driver": driver,
        }
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        logger.info("Could not query the GPU (%s) - proceeding without the check.", exc)
        return None


def check_gpu_vram_mb() -> Optional[int]:
    """Total VRAM on the first NVIDIA GPU, or None - thin wrapper over
    check_gpu_info() kept for callers that only need the one number."""
    info = check_gpu_info()
    return info["total_mb"] if info else None


def check_vram_headroom(model_name: str) -> Optional[str]:
    """
    Returns a human-readable warning if the GPU likely lacks free VRAM for
    `model_name`, or None if there's enough (or if we can't tell).

    Only meaningful when the CUDA build is what will actually run - on the CPU
    build VRAM is irrelevant, so this returns None rather than warning about a
    constraint that doesn't apply.

    Checks FREE VRAM, not total: this app can have a ~9GB Ollama model resident on
    the same card during AI Arbitration, so total capacity badly overstates what's
    actually available. Advisory only - never blocks the run, since the user may
    well know something we don't (Ollama idle-unloaded, another app just closed).
    """
    if active_variant() is not WhisperVariant.CUDA:
        return None
    needed = _MODEL_VRAM_MB.get(model_name)
    if needed is None:
        return None
    info = check_gpu_info()
    if info is None:
        return None

    free = info["free_mb"]
    if free >= needed:
        return None
    return (
        f"Low GPU memory: the '{model_name}' model needs roughly {needed / 1024:.1f}GB of "
        f"VRAM but only {free / 1024:.1f}GB is free on your {info['name']} "
        f"({info['total_mb'] / 1024:.1f}GB total). Transcription may fail or fall back to "
        "the CPU. Closing other GPU apps - or stopping Ollama if AI Arbitration isn't "
        "running - would free some up."
    )


def has_free_space(path: Path, need_bytes: int) -> bool:
    """Whether `path`'s volume has room, walking up to the nearest existing parent
    (the target dir usually doesn't exist yet on a first install)."""
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free >= need_bytes
    except OSError:
        return True  # can't tell - don't block the install on a failed check


def download_whisper_release_zip(
    dest: Path, variant: WhisperVariant = WhisperVariant.CPU,
) -> Iterator[dict]:
    """Streams the official whisper.cpp Windows release zip for `variant` to `dest`,
    yielding {'downloaded_mb': ..., 'total_mb': ... | None, 'percent': ... | None} as
    it goes - same shape as ollama_setup.download_ollama_installer."""
    response = requests.get(_RELEASE_ZIP_URLS[variant], stream=True, timeout=30)
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


def install_whisper_binary_streaming(zip_path: Path, bin_dir: Path = BIN_DIR) -> Iterator[dict]:
    """
    Extracts whisper-cli(.exe) AND every DLL sitting alongside it in the archive,
    yielding {'extracted_mb', 'total_mb', 'percent', 'name'} per member.

    Unlike ffmpeg's single static exe, whisper.cpp's Windows releases ship the CLI
    dynamically linked against several DLLs (ggml*.dll, whisper.dll, and for the CUDA
    build cudart/cublas/cublasLt/nvrtc too) - all of them have to land next to the exe
    or it fails to launch at all. This is exactly the class of bug this project already
    hit once with TwitchDownloaderCLI/ffmpeg ("Unable to find FFmpeg" - see
    config._default_ffmpeg_binary's docstring).

    Copied via copyfileobj rather than read()/write_bytes: the CUDA build's
    ggml-cuda.dll is ~512MB and cublasLt64_12.dll ~450MB uncompressed, and reading
    either into a single bytes object inside a long-lived Gradio process is a real
    memory hazard. The final yield carries {'done': True, 'path': <exe>}.
    """
    exe_name = _exe_name()
    try:
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
            total_bytes = sum(zf.getinfo(n).file_size for n in to_extract) or 1
            bin_dir.mkdir(parents=True, exist_ok=True)
            written = 0
            for member in to_extract:
                with zf.open(member) as src, open(bin_dir / Path(member).name, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                written += zf.getinfo(member).file_size
                yield {
                    "extracted_mb": written / (1024 * 1024),
                    "total_mb": total_bytes / (1024 * 1024),
                    "percent": written / total_bytes * 100,
                    "name": Path(member).name,
                }
    except OSError as exc:
        # Disk full mid-extract, or a read-only bin/ in a packaged install. Surfaced as
        # WhisperSetupError so callers handle it like any other setup-action failure.
        raise WhisperSetupError(f"Could not extract the whisper.cpp archive: {exc}") from exc

    result = bin_dir / exe_name
    if not result.exists():
        raise WhisperSetupError(f"Extraction completed but '{result}' is still missing.")
    yield {"done": True, "path": result}


def install_whisper_binary(zip_path: Path, bin_dir: Path = BIN_DIR) -> Path:
    """Non-streaming wrapper over install_whisper_binary_streaming, for callers that
    don't want per-member progress."""
    final = None
    for progress in install_whisper_binary_streaming(zip_path, bin_dir):
        final = progress
    return final["path"]


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


def remove_variant(variant: WhisperVariant) -> None:
    """
    CUDA lives in its own directory (~1.1GB of exes and DLLs), so removing it is an
    rmtree - unlinking just the exe would strand the rest. The CPU build shares bin/
    with ffmpeg, TwitchDownloaderCLI and the models dir, so that one can only remove
    the files it actually owns.
    """
    if variant is WhisperVariant.CUDA:
        shutil.rmtree(CUDA_BIN_DIR, ignore_errors=True)
        return
    remove_binary(binary_path_for(WhisperVariant.CPU))


def remove_binary(binary_path: Path) -> None:
    binary_path.unlink(missing_ok=True)
