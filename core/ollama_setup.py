"""
core/ollama_setup.py

Optional, opt-in local-AI setup: detects whether Ollama + the target model are
available, and can install/pull/remove them on the user's behalf so "AI
Arbitration" doesn't require knowing what Ollama even is. Every check here is
live (hits the actual filesystem/API each call) rather than a stored "setup
complete" flag, so it naturally handles a fresh machine, a machine that
already has Ollama for something else, and an already-fully-set-up machine
with the same code path - and stays correct even if the user installs or
removes Ollama themselves outside this app.

Nothing here runs automatically; app.py only calls these from explicit button
clicks. A bug in the automated path should degrade to "install it yourself
from ollama.com" (always shown alongside), never strand the user.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
from enum import Enum, auto
from pathlib import Path
from typing import Iterator, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "http://localhost:11434"   # mirrors LLMConfig.DEFAULT_API_BASE
DEFAULT_MODEL = "qwen2.5:14b-instruct"        # bare name, mirrors LLMConfig.DEFAULT_MODEL

_OLLAMA_INSTALLER_URL = "https://github.com/ollama/ollama/releases/latest/download/OllamaSetup.exe"
_REQUEST_TIMEOUT_S = 15


class OllamaState(Enum):
    NOT_INSTALLED = auto()
    INSTALLED_NOT_RUNNING = auto()
    RUNNING_MODEL_MISSING = auto()
    READY = auto()


class OllamaSetupError(Exception):
    """Raised for setup actions (install/pull/remove) that fail outright -
    never for "not installed yet", which is a normal detect_state() result."""


def is_ollama_on_path() -> bool:
    return shutil.which("ollama") is not None


def is_server_running(api_base: str = DEFAULT_API_BASE) -> bool:
    try:
        requests.get(f"{api_base.rstrip('/')}/api/tags", timeout=5)
        return True
    except requests.exceptions.RequestException:
        return False


def get_installed_models(api_base: str = DEFAULT_API_BASE) -> List[str]:
    """Same /api/tags call app.py's do_fetch_models already made inline - kept here
    as the single implementation so the Settings-tab dropdown and the live
    setup-detection below never disagree. Raises requests.RequestException /
    ValueError on failure; callers that just want a yes/no should use detect_state()."""
    base = api_base.rstrip("/")
    resp = requests.get(f"{base}/api/tags", timeout=_REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    payload = resp.json()
    return sorted({
        str(entry.get("model") or entry.get("name"))
        for entry in payload.get("models", [])
        if entry.get("model") or entry.get("name")
    })


def detect_state(model_name: str = DEFAULT_MODEL, api_base: str = DEFAULT_API_BASE) -> OllamaState:
    if not is_ollama_on_path() and not is_server_running(api_base):
        return OllamaState.NOT_INSTALLED
    try:
        models = get_installed_models(api_base)
    except (requests.exceptions.RequestException, ValueError):
        return OllamaState.INSTALLED_NOT_RUNNING
    return OllamaState.READY if model_name in models else OllamaState.RUNNING_MODEL_MISSING


def check_gpu_vram_mb() -> Optional[int]:
    """Total VRAM on the first NVIDIA GPU, or None if nvidia-smi isn't available
    (AMD/Intel/no dGPU) or anything about the call fails - callers should treat
    None as "couldn't determine", not "no GPU"."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        first_line = result.stdout.strip().splitlines()[0]
        return int(first_line.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        logger.info("Could not determine GPU VRAM (%s) - proceeding without the check.", exc)
        return None


def download_ollama_installer(dest: Path) -> Iterator[dict]:
    """Streams the official Windows installer to `dest`, yielding
    {'downloaded_mb': ..., 'total_mb': ... | None, 'percent': ... | None} as it goes."""
    response = requests.get(_OLLAMA_INSTALLER_URL, stream=True, timeout=30)
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


def install_ollama_silent(installer_path: Path) -> None:
    """Standard per-user, no-admin Inno Setup silent install. The one known silent-
    install bug (ollama/ollama#7969) is specifically about all-users/enterprise
    deployment - doesn't apply to a normal single-user install like this."""
    result = subprocess.run(
        [str(installer_path), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise OllamaSetupError(f"Ollama installer exited with code {result.returncode}: {result.stderr}")


def pull_model(model_name: str = DEFAULT_MODEL, api_base: str = DEFAULT_API_BASE) -> Iterator[dict]:
    """Streams Ollama's own /api/pull progress straight through - each line is
    already a small JSON dict with 'status' and, once download starts, 'completed'/
    'total' byte counts."""
    base = api_base.rstrip("/")
    response = requests.post(
        f"{base}/api/pull", json={"name": model_name, "stream": True}, stream=True, timeout=30,
    )
    response.raise_for_status()
    for line in response.iter_lines():
        if not line:
            continue
        yield json.loads(line)


def remove_model(model_name: str = DEFAULT_MODEL, api_base: str = DEFAULT_API_BASE) -> None:
    base = api_base.rstrip("/")
    resp = requests.delete(f"{base}/api/delete", json={"name": model_name}, timeout=_REQUEST_TIMEOUT_S)
    resp.raise_for_status()


def find_ollama_uninstaller() -> Optional[Path]:
    """Inno Setup convention (%LOCALAPPDATA%\\Programs\\Ollama\\unins000.exe), falling
    back to the registry uninstall string if the app was installed somewhere else."""
    if platform.system() != "Windows":
        return None
    import winreg  # Windows-only stdlib module - local import so this file still imports fine elsewhere.

    default_path = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "unins000.exe"
    if default_path.exists():
        return default_path

    uninstall_key = r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, uninstall_key) as root:
                for i in range(winreg.QueryInfoKey(root)[0]):
                    subkey_name = winreg.EnumKey(root, i)
                    with winreg.OpenKey(root, subkey_name) as subkey:
                        try:
                            display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                        except OSError:
                            continue
                        if display_name != "Ollama":
                            continue
                        try:
                            uninstall_string = winreg.QueryValueEx(subkey, "UninstallString")[0]
                        except OSError:
                            continue
                        candidate = Path(uninstall_string.strip('"'))
                        if candidate.exists():
                            return candidate
        except OSError:
            continue
    return None


def uninstall_ollama_silent() -> None:
    uninstaller = find_ollama_uninstaller()
    if uninstaller is None:
        raise OllamaSetupError("Could not locate Ollama's uninstaller - remove it from Windows Settings > Apps instead.")
    result = subprocess.run([str(uninstaller), "/VERYSILENT"], capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise OllamaSetupError(f"Ollama uninstaller exited with code {result.returncode}: {result.stderr}")
