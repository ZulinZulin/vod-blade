"""
core/preview.py

On-demand short clip extraction for the operator preview feature. Uses
ffmpeg stream-copy (no re-encode) to remux a candidate's [start, end] range
out of the already-downloaded source VOD into a small temp file the UI can
hand to a <video> player - lets an operator quickly see/hear a candidate
beyond its subtitle snippet, without waiting on a transcode.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

from config import ExportConfig, PREVIEWS_DIR, THUMBNAILS_DIR, settings

logger = logging.getLogger(__name__)


class PreviewError(Exception):
    """Raised when a preview clip can't be extracted."""


def extract_preview_clip(
    source_video_path: str,
    start_time: float,
    end_time: float,
    cfg: ExportConfig = None,
) -> Path:
    """
    Extracts [start_time, end_time] from source_video_path into a small mp4 via
    ffmpeg stream-copy. This is a fast remux, not a re-encode, so it stays
    near-instant even on a multi-hour VOD - but stream-copy can only cut at the
    nearest keyframe, so the actual boundaries may drift a second or two from
    what was asked. That's fine for a quick preview; the real exports compute
    frame-accurate boundaries against the original file and are unaffected by this.

    Content-addressed and cached under data/cache/previews/ (hash of source path +
    start + end), same pattern as extract_thumbnail below - re-previewing the exact
    same range is then free instead of re-running ffmpeg and leaking another
    same-content file. This used to write a brand-new uniquely-named file to the
    OS temp dir on every single click with no cleanup at all; this both fixes
    that leak and gets clip reuse for free by construction, rather than needing
    to remember to delete the previous one.
    """
    cfg = cfg or settings.export
    source_path = Path(source_video_path) if source_video_path else None
    if not source_path or not source_path.exists():
        raise PreviewError(f"Source video not found: {source_video_path or '(none set)'}")
    if end_time <= start_time:
        raise PreviewError(f"Invalid clip range: start={start_time}, end={end_time}")

    key = hashlib.sha1(
        f"{source_path.resolve()}|{start_time:.3f}|{end_time:.3f}".encode("utf-8")
    ).hexdigest()[:16]
    out_path = PREVIEWS_DIR / f"{key}.mp4"
    if out_path.exists():
        return out_path
    duration = end_time - start_time

    try:
        result = subprocess.run(
            [
                cfg.ffmpeg_binary, "-y",
                "-ss", f"{start_time:.3f}",
                "-i", str(source_path),
                "-t", f"{duration:.3f}",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(out_path),
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreviewError(f"Could not run ffmpeg ('{cfg.ffmpeg_binary}'): {exc}") from exc

    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        stderr_tail = result.stderr.strip()[-500:] if result.stderr else "(no stderr)"
        raise PreviewError(f"ffmpeg failed to extract preview clip (exit {result.returncode}): {stderr_tail}")

    return out_path


def extract_thumbnail(source_video_path: str, timestamp: float, cfg: ExportConfig = None) -> Path:
    """
    Extracts a single JPEG frame near `timestamp`, used as a lightweight visual
    stand-in for a candidate card so an operator can scan a page of cards
    without playing every preview video. Content-addressed (hash of source
    path + timestamp) and cached under data/cache/thumbnails/, since - unlike
    the video preview, which only runs on an explicit button click - this
    runs every time the card list re-renders (page nav, accept/reject toggle,
    etc.); without caching, that would re-invoke ffmpeg a dozen times per click.
    """
    cfg = cfg or settings.export
    source_path = Path(source_video_path) if source_video_path else None
    if not source_path or not source_path.exists():
        raise PreviewError(f"Source video not found: {source_video_path or '(none set)'}")

    key = hashlib.sha1(f"{source_path.resolve()}|{timestamp:.2f}".encode("utf-8")).hexdigest()[:16]
    out_path = THUMBNAILS_DIR / f"{key}.jpg"
    if out_path.exists():
        return out_path

    try:
        result = subprocess.run(
            [
                cfg.ffmpeg_binary, "-y",
                "-ss", f"{timestamp:.3f}",
                "-i", str(source_path),
                "-frames:v", "1",
                "-q:v", "4",
                str(out_path),
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreviewError(f"Could not run ffmpeg ('{cfg.ffmpeg_binary}'): {exc}") from exc

    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        stderr_tail = result.stderr.strip()[-500:] if result.stderr else "(no stderr)"
        raise PreviewError(f"ffmpeg failed to extract thumbnail (exit {result.returncode}): {stderr_tail}")

    return out_path
