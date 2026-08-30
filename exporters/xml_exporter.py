"""
exporters/xml_exporter.py

Generates FCPXML (Final Cut Pro XML, DaVinci Resolve-compatible) and CMX
3600 EDL files from refined clip candidates. Each clip becomes its own
project/sequence (FCPXML) or its own cut (EDL) referencing the shared
source media, so editors can import the whole batch and pick clips
individually once inside Resolve.

FCPXML is the primary, richer target (carries clip titles/notes and can
hold many named sequences in one file). EDL is offered as a lowest-common-
denominator fallback for other NLEs, at the cost of losing titles/notes.
"""

from __future__ import annotations

import logging
import re
import subprocess
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional

from lxml import etree

from config import ExportConfig, settings
from core.llm_agent import CandidateClip

logger = logging.getLogger(__name__)


class ExportError(Exception):
    """Raised when clip data can't be turned into a valid export file."""


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

# (frame-duration numerator, denominator) for common broadcast rates, seconds/frame.
_FRAME_DURATIONS = {
    23.976: (1001, 24000),
    24: (1, 24),
    25: (1, 25),
    29.97: (1001, 30000),
    30: (1, 30),
    50: (1, 50),
    59.94: (1001, 60000),
    60: (1, 60),
}


def _frame_duration_fraction(fps: float) -> Fraction:
    closest = min(_FRAME_DURATIONS, key=lambda f: abs(f - fps))
    if abs(closest - fps) > 0.05:
        return Fraction(1, round(fps))
    num, den = _FRAME_DURATIONS[closest]
    return Fraction(num, den)


def _is_ntsc_rate(fps: float) -> bool:
    """True for fractional broadcast rates like 29.97/59.94/23.976."""
    return abs(fps - round(fps)) > 0.001


_DISALLOWED_NAME_CHARS_RE = re.compile(r"[^A-Za-z0-9 _-]")
_WHITESPACE_RE = re.compile(r"\s+")


def _safe_name(name: str, fallback: str, max_len: int = 80) -> str:
    cleaned = _DISALLOWED_NAME_CHARS_RE.sub(" ", name)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return (cleaned or fallback)[:max_len]


def _dedupe_name(base_name: str, used_names: set) -> str:
    name, suffix = base_name, 2
    while name in used_names:
        name = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(name)
    return name


def _validate_clips(clips: List[CandidateClip]) -> None:
    if not clips:
        raise ExportError("No clips to export.")
    for clip in clips:
        if clip.end_time <= clip.start_time:
            raise ExportError(f"Invalid clip '{clip.title}': end_time <= start_time.")


def _timestamped_path(export_dir: Path, extension: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return export_dir / f"streamcutter_clips_{stamp}.{extension}"


def _detect_source_fps(source_video_path: Optional[str], cfg: ExportConfig) -> Optional[float]:
    """
    Probes the actual source file's video stream frame rate via ffprobe, so
    exports use the file's real rate instead of a fixed assumption that's
    silently wrong whenever the actual source differs from it (the same class
    of bug that made injected Resolve clips land in the wrong place: assuming
    a rate instead of reading the file's real one). Returns None (caller falls
    back to cfg.default_fps) if detection fails for any reason - no path given,
    ffprobe missing, file unreadable, no video stream, unparseable output.
    """
    if not source_video_path:
        return None
    source_path = Path(source_video_path)
    if not source_path.exists():
        return None

    try:
        result = subprocess.run(
            [
                cfg.ffprobe_binary, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(source_path),
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "Could not run ffprobe ('%s') to detect the source frame rate: %s. "
            "Falling back to the configured default (%.2f fps).",
            cfg.ffprobe_binary, exc, cfg.default_fps,
        )
        return None

    output = result.stdout.strip()
    if result.returncode != 0 or not output:
        logger.warning(
            "ffprobe couldn't detect the frame rate for '%s' (exit %d): %s. Falling back to %.2f fps.",
            source_path.name, result.returncode, result.stderr.strip(), cfg.default_fps,
        )
        return None

    try:
        if "/" in output:
            num_str, den_str = output.split("/", 1)
            fps = float(num_str) / float(den_str)
        else:
            fps = float(output)
    except (ValueError, ZeroDivisionError) as exc:
        logger.warning(
            "Could not parse ffprobe's frame-rate output %r: %s. Falling back to %.2f fps.",
            output, exc, cfg.default_fps,
        )
        return None

    logger.info("Detected source frame rate for '%s': %.3f fps", source_path.name, fps)
    return fps


# --------------------------------------------------------------------------- #
# FCPXML
# --------------------------------------------------------------------------- #


def _seconds_to_fcp_time(seconds: float, fps: float) -> str:
    """FCPXML times are rational strings ('<num>/<den>s'), snapped to whole frames."""
    frame_dur = _frame_duration_fraction(fps)
    frame_count = round(seconds / float(frame_dur))
    total = frame_dur * frame_count
    return f"{total.numerator}/{total.denominator}s"


def generate_fcpxml(
    clips: List[CandidateClip],
    source_video_path: str,
    export_config: Optional[ExportConfig] = None,
    source_duration_seconds: Optional[float] = None,
) -> str:
    """Returns a complete FCPXML document (UTF-8, pretty-printed) as a string."""
    _validate_clips(clips)
    cfg = export_config or settings.export
    fps = _detect_source_fps(source_video_path, cfg) or cfg.default_fps

    source_path = Path(source_video_path)
    if not source_path.exists():
        logger.warning(
            "Source video not found at '%s'; FCPXML will still reference this path "
            "(fine if it's a placeholder that gets relinked in Resolve).", source_path,
        )

    if source_duration_seconds is None:
        source_duration_seconds = max(clip.end_time for clip in clips) + 60.0
        logger.info(
            "No source_duration_seconds given; estimating asset duration as %.1fs.",
            source_duration_seconds,
        )

    fcpxml = etree.Element("fcpxml", version=cfg.fcpxml_version)
    resources = etree.SubElement(fcpxml, "resources")

    format_id = "r1"
    frame_dur = _frame_duration_fraction(fps)
    etree.SubElement(
        resources, "format",
        id=format_id,
        name=f"FFVideoFormat{cfg.default_timeline_height}p{fps:g}",
        frameDuration=f"{frame_dur.numerator}/{frame_dur.denominator}s",
        width=str(cfg.default_timeline_width),
        height=str(cfg.default_timeline_height),
    )

    asset_id = "a1"
    try:
        asset_src = source_path.resolve().as_uri()
    except (OSError, ValueError):
        # as_uri() requires an absolute path; fall back to a relative file:// reference.
        asset_src = f"file:///{source_path.as_posix().lstrip('/')}"

    etree.SubElement(
        resources, "asset",
        id=asset_id,
        name=source_path.stem or "source",
        src=asset_src,
        start="0s",
        duration=_seconds_to_fcp_time(source_duration_seconds, fps),
        hasVideo="1",
        hasAudio="1",
        format=format_id,
    )

    library = etree.SubElement(fcpxml, "library")
    event = etree.SubElement(library, "event", name="VOD BLADE Clips")

    used_names: set = set()
    for i, clip in enumerate(clips, start=1):
        name = _dedupe_name(_safe_name(clip.title, f"Clip_{i}"), used_names)
        clip_duration = _seconds_to_fcp_time(clip.duration, fps)

        project = etree.SubElement(event, "project", name=name)
        sequence = etree.SubElement(project, "sequence", format=format_id, duration=clip_duration)
        spine = etree.SubElement(sequence, "spine")
        asset_clip = etree.SubElement(
            spine, "asset-clip",
            ref=asset_id,
            name=name,
            offset="0s",
            start=_seconds_to_fcp_time(clip.start_time, fps),
            duration=clip_duration,
        )
        note = etree.SubElement(asset_clip, "note")
        note.text = f"{clip.summary} (viral_score={clip.viral_score}/10)"

    xml_bytes = etree.tostring(
        fcpxml,
        pretty_print=True,
        xml_declaration=True,
        encoding="UTF-8",
        doctype="<!DOCTYPE fcpxml>",
    )
    return xml_bytes.decode("utf-8")


def export_fcpxml_file(
    clips: List[CandidateClip],
    source_video_path: str,
    output_path: Optional[Path] = None,
    export_config: Optional[ExportConfig] = None,
    source_duration_seconds: Optional[float] = None,
) -> Path:
    """Renders FCPXML and writes it to disk, returning the written path."""
    cfg = export_config or settings.export
    xml_str = generate_fcpxml(clips, source_video_path, cfg, source_duration_seconds)

    output_path = Path(output_path) if output_path else _timestamped_path(cfg.export_dir, "fcpxml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml_str, encoding="utf-8")
    logger.info("Wrote FCPXML with %d clip(s) to %s", len(clips), output_path)
    return output_path


# --------------------------------------------------------------------------- #
# EDL (CMX 3600)
# --------------------------------------------------------------------------- #


def _drop_frame_timecode(frame_number: int, fps: float) -> str:
    """SMPTE drop-frame timecode for NTSC rates (29.97/59.94/...)."""
    drop_frames = round(fps * 0.066666)
    frames_per_10_minutes = round(fps * 600)
    nominal_fps = round(fps)
    frames_per_minute = nominal_fps * 60 - drop_frames
    frames_per_24_hours = round(fps * 3600) * 24

    frame_number %= frames_per_24_hours
    d, m = divmod(frame_number, frames_per_10_minutes)
    if m > drop_frames:
        frame_number += (drop_frames * 9 * d) + drop_frames * ((m - drop_frames) // frames_per_minute)
    else:
        frame_number += drop_frames * 9 * d

    frames = frame_number % nominal_fps
    secs_total = frame_number // nominal_fps
    secs = secs_total % 60
    mins = (secs_total // 60) % 60
    hrs = (secs_total // 3600) % 24
    return f"{hrs:02d}:{mins:02d}:{secs:02d};{frames:02d}"


def _seconds_to_timecode(seconds: float, fps: float) -> str:
    if _is_ntsc_rate(fps):
        return _drop_frame_timecode(round(seconds * fps), fps)
    nominal_fps = round(fps)
    total_frames = round(seconds * nominal_fps)
    frames = total_frames % nominal_fps
    secs_total = total_frames // nominal_fps
    secs = secs_total % 60
    mins = (secs_total // 60) % 60
    hrs = secs_total // 3600
    return f"{hrs:02d}:{mins:02d}:{secs:02d}:{frames:02d}"


def generate_edl(
    clips: List[CandidateClip],
    title: str = "VOD BLADE Clips",
    reel_name: str = "AX",
    export_config: Optional[ExportConfig] = None,
    source_video_path: Optional[str] = None,
) -> str:
    """
    Returns a CMX 3600 EDL as a string. Clips are laid back-to-back on the
    record track. `source_video_path`, when given, is probed via ffprobe for
    the real source frame rate - EDL timecodes are frame-rate-dependent, so
    without it a source file that isn't actually cfg.default_fps produces
    timecodes scaled wrong by whatever the ratio is between the two rates.
    """
    _validate_clips(clips)
    cfg = export_config or settings.export
    fps = _detect_source_fps(source_video_path, cfg) or cfg.default_fps
    reel = _safe_name(reel_name, "AX", max_len=8).upper() or "AX"

    lines = [
        f"TITLE: {title}",
        f"FCM: {'DROP FRAME' if _is_ntsc_rate(fps) else 'NON-DROP FRAME'}",
        "",
    ]
    record_cursor = 0.0
    used_names: set = set()
    for i, clip in enumerate(clips, start=1):
        src_in = _seconds_to_timecode(clip.start_time, fps)
        src_out = _seconds_to_timecode(clip.end_time, fps)
        rec_in = _seconds_to_timecode(record_cursor, fps)
        rec_out = _seconds_to_timecode(record_cursor + clip.duration, fps)
        name = _dedupe_name(_safe_name(clip.title, f"Clip_{i}"), used_names)

        lines.append(f"{i:03d}  {reel:<8} V     C        {src_in} {src_out} {rec_in} {rec_out}")
        lines.append(f"* FROM CLIP NAME: {name}")
        lines.append(f"* COMMENT: {clip.summary} (viral_score={clip.viral_score}/10)")
        lines.append("")

        record_cursor += clip.duration

    return "\n".join(lines) + "\n"


def export_edl_file(
    clips: List[CandidateClip],
    output_path: Optional[Path] = None,
    title: str = "VOD BLADE Clips",
    reel_name: str = "AX",
    export_config: Optional[ExportConfig] = None,
    source_video_path: Optional[str] = None,
) -> Path:
    """Renders an EDL and writes it to disk, returning the written path."""
    cfg = export_config or settings.export
    edl_str = generate_edl(clips, title, reel_name, cfg, source_video_path=source_video_path)

    output_path = Path(output_path) if output_path else _timestamped_path(cfg.export_dir, "edl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(edl_str, encoding="utf-8")
    logger.info("Wrote EDL with %d clip(s) to %s", len(clips), output_path)
    return output_path


# --------------------------------------------------------------------------- #
# Convenience module-level API
# --------------------------------------------------------------------------- #


def export_clips(
    clips: List[CandidateClip],
    source_video_path: str,
    formats: tuple = ("fcpxml", "edl"),
) -> Dict[str, Path]:
    """Writes each requested format to disk; returns {format: written_path}."""
    written: Dict[str, Path] = {}
    if "fcpxml" in formats:
        written["fcpxml"] = export_fcpxml_file(clips, source_video_path)
    if "edl" in formats:
        written["edl"] = export_edl_file(clips, source_video_path=source_video_path)
    return written
