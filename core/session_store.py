"""
core/session_store.py

Saves/restores a review session (clip candidates plus the source metadata
needed to keep Preview/Export working) to a local JSON file under
data/sessions/. Protects a real analysis run - which can take a long time
(chat/subtitle fetching over a multi-hour VOD, dozens of real LLM judgment
calls) - from being lost to an app restart or crash, and lets an operator
checkpoint manual review edits (accept/reject overrides, boundary tweaks)
before export.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config import SESSIONS_DIR
from core.llm_agent import CandidateClip

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


class SessionError(Exception):
    """Raised when a session can't be saved or loaded."""


def _slugify(text: str, max_len: int = 40) -> str:
    slug = _SLUG_RE.sub("_", text).strip("_")
    return slug[:max_len] or "session"


def save_session(
    clips: List[CandidateClip],
    source_video_path: str,
    youtube_source: str,
    twitch_source: str,
    chat_offset: float,
    session_path: Optional[str] = None,
) -> Path:
    """
    Writes the current review state to a JSON file. If session_path is given,
    overwrites that file (checkpointing an existing session in place) instead
    of creating a new one - keeps repeated manual saves during review from
    piling up near-duplicate files.
    """
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "youtube_source": youtube_source,
        "twitch_source": twitch_source,
        "chat_offset": chat_offset,
        "source_video_path": source_video_path,
        "clips": [asdict(c) for c in clips],
    }

    if session_path:
        out_path = Path(session_path)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = _slugify(twitch_source or youtube_source or "session")
        out_path = SESSIONS_DIR / f"{slug}_{stamp}.json"

    try:
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        raise SessionError(f"Could not write session file '{out_path}': {exc}") from exc

    return out_path


def load_session(session_path: str) -> dict:
    """Returns {"clips", "source_video_path", "youtube_source", "twitch_source", "chat_offset"}."""
    path = Path(session_path)
    if not path.exists():
        raise SessionError(f"Session file not found: {session_path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionError(f"Could not read session file '{path.name}': {exc}") from exc

    try:
        clips = [CandidateClip(**c) for c in payload["clips"]]
    except (KeyError, TypeError) as exc:
        raise SessionError(f"Session file '{path.name}' has an unrecognized/incompatible format: {exc}") from exc

    return {
        "clips": clips,
        "source_video_path": payload.get("source_video_path", ""),
        "youtube_source": payload.get("youtube_source", ""),
        "twitch_source": payload.get("twitch_source", ""),
        "chat_offset": payload.get("chat_offset", 0.0),
    }


def list_sessions() -> List[Path]:
    """All saved session files, most recently modified first."""
    return sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
