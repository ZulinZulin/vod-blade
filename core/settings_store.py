"""
core/settings_store.py

Small persisted key-value settings for UI state that should survive an app
restart (e.g. a user-chosen downloads folder) but doesn't belong in .env
(that's dev/deploy config, loaded once at process start via load_dotenv() -
see config.py) or in a session file (that's per-review-run state, see
core/session_store.py). Stored as one small JSON file under DATA_DIR.

Deliberately never raises: a missing or corrupt settings file just means
"nothing persisted yet" - it should never block the app from starting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from config import DATA_DIR

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_SETTINGS_PATH = DATA_DIR / "settings.json"


def load_settings() -> dict:
    """Returns the persisted settings dict, or {} if missing/corrupt."""
    if not _SETTINGS_PATH.exists():
        return {}
    try:
        payload = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read settings file '%s', ignoring: %s", _SETTINGS_PATH, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def save_settings(partial: dict) -> None:
    """Read-modify-write merges `partial` into the persisted settings."""
    current = load_settings()
    current.update(partial)
    current["schema_version"] = _SCHEMA_VERSION
    try:
        _SETTINGS_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write settings file '%s': %s", _SETTINGS_PATH, exc)


def get_downloads_dir_override() -> Optional[Path]:
    """The last user-chosen downloads folder, or None if never set/blank."""
    value = load_settings().get("downloads_dir", "")
    return Path(value) if value and value.strip() else None


def set_downloads_dir_override(path: Path) -> None:
    save_settings({"downloads_dir": str(path)})


def is_onboarding_completed() -> bool:
    """Whether the first-run setup panel has been finished or skipped -
    default False so a fresh install shows it on the very first page load."""
    return bool(load_settings().get("onboarding_completed", False))


def mark_onboarding_completed() -> None:
    save_settings({"onboarding_completed": True})
