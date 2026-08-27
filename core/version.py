"""
core/version.py

App release version and GitHub-Releases update check. Deliberately manual-only
(no automatic check on launch) - a background network call on every startup is
a surprise a locally-run tool shouldn't spring on someone without asking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import requests

from config import BASE_DIR

logger = logging.getLogger(__name__)

_VERSION_FILE = BASE_DIR / "VERSION"
_UNKNOWN_VERSION = "0.0.0-dev"
_REQUEST_TIMEOUT_S = 10


def get_version() -> str:
    """The packaged app's own version, or a dev-checkout placeholder if VERSION
    doesn't exist yet (e.g. a fresh git clone with no release built from it)."""
    try:
        return _VERSION_FILE.read_text(encoding="utf-8").strip() or _UNKNOWN_VERSION
    except OSError:
        return _UNKNOWN_VERSION


def _parse_version(version: str) -> Tuple[int, ...]:
    """'v1.2.3' / '1.2.3-dev' -> (1, 2, 3). Non-numeric trailing parts are dropped
    rather than raising, since a hand-edited VERSION file or an unusual tag name
    shouldn't crash the comparison."""
    core = version.lstrip("v").split("-", 1)[0]
    parts = []
    for piece in core.split("."):
        if not piece.isdigit():
            break
        parts.append(int(piece))
    return tuple(parts)


@dataclass(frozen=True)
class UpdateCheckResult:
    current: str
    latest: Optional[str]
    is_newer: bool
    release_url: Optional[str]
    error: Optional[str] = None


def check_for_update(repo: str) -> UpdateCheckResult:
    """Compares the local VERSION against the given GitHub repo's latest release.
    `repo` is "owner/name". Never raises - any failure (offline, rate-limited,
    repo not public yet) just reports no update available, same defensive style
    as the existing Ollama /api/tags lookup in app.py's do_fetch_models."""
    current = get_version()
    try:
        response = requests.get(
            f"https://api.github.com/repos/{repo}/releases/latest",
            timeout=_REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        logger.warning("Update check failed: %s", exc)
        return UpdateCheckResult(current=current, latest=None, is_newer=False, release_url=None, error=str(exc))

    latest = payload.get("tag_name", "")
    release_url = payload.get("html_url")
    is_newer = _parse_version(latest) > _parse_version(current)
    return UpdateCheckResult(current=current, latest=latest, is_newer=is_newer, release_url=release_url)
