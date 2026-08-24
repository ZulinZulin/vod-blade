"""
core/fetchers.py

Data ingestion layer for StreamCutter.

Three independent input paths feed the rest of the pipeline:

1. YouTube subtitles (or a local .srt/.vtt/.txt transcript) -> List[SubtitleSegment]
2. Twitch VOD chat log (via TwitchDownloaderCLI)             -> List[ChatMessage]
3. Twitch VOD video file (via TwitchDownloaderCLI)           -> Path (local .mp4)

All three are pure I/O + parsing: they know nothing about hype scoring or LLM
prompts. `chat_offset_seconds` is applied here, once, as an explicit correction
step so every downstream module can assume timestamps already share the same
clock as the source (YouTube) video.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import srt as srt_lib
import webvtt
import yt_dlp

from config import CACHE_DIR, DOWNLOADS_DIR, settings

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class FetcherError(Exception):
    """Base class for all data-ingestion failures."""


class SubtitleFetchError(FetcherError):
    """Raised when YouTube subtitles cannot be located, downloaded, or parsed."""


class ChatFetchError(FetcherError):
    """Raised when Twitch chat cannot be downloaded or parsed."""


class VideoFetchError(FetcherError):
    """Raised when a Twitch VOD's video file cannot be downloaded."""


class BinaryNotFoundError(FetcherError):
    """Raised when a required local CLI binary is missing."""


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SubtitleSegment:
    """A single subtitle cue aligned to the source video's timeline (seconds)."""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class ChatMessage:
    """A single Twitch chat message aligned to the source video's timeline (seconds)."""

    timestamp: float
    username: str
    text: str
    badges: List[str]
    is_subscriber: bool = False
    is_moderator: bool = False


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})")
TWITCH_VOD_URL_RE = re.compile(r"twitch\.tv/videos/(\d+)")
TWITCH_VOD_ID_ONLY_RE = re.compile(r"^\d+$")

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _cache_key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def extract_twitch_vod_id(vod_url_or_id: str) -> str:
    """Accepts a full Twitch VOD URL or a bare numeric VOD id, returns the id."""
    candidate = vod_url_or_id.strip()
    match = TWITCH_VOD_URL_RE.search(candidate)
    if match:
        return match.group(1)
    if TWITCH_VOD_ID_ONLY_RE.match(candidate):
        return candidate
    raise ChatFetchError(
        f"Could not extract a Twitch VOD id from '{vod_url_or_id}'. "
        "Expected a twitch.tv/videos/<id> URL or a bare numeric id."
    )


def _parse_timestamp(value: str) -> float:
    """Parses 'HH:MM:SS.mmm' or 'HH:MM:SS,mmm' into seconds."""
    value = value.replace(",", ".")
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _clean_subtitle_text(text: str) -> str:
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)  # VTT cue text can carry literal entities, e.g. "&gt;&gt;", "&nbsp;"
    text = text.replace("\xa0", " ")  # &nbsp; unescapes to U+00A0, not a plain space
    text = text.replace("\n", " ").strip()
    return _WHITESPACE_RE.sub(" ", text)


def _dedupe_segments(segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
    """
    YouTube auto-captions render as a rolling 2-line window: consecutive cues
    overlap rather than repeat exactly, e.g.

        cue N:   "...as a cultural phenomenon,"
        cue N+1: "as a cultural phenomenon,"                        (old line, no new words)
        cue N+2: "as a cultural phenomenon, it's a circus that"     (old line + new words)

    Emitting every cue as-is roughly doubles transcript length (and LLM token
    cost) with zero new information. Since each cue is either a prefix of, a
    suffix of, or unrelated to its neighbor, keep only the genuinely NEW text
    each cue adds relative to the previous one, timestamped at that cue's own
    [start, end] - cues that add nothing (pure re-displays of already-seen
    text) are dropped entirely.
    """
    if not segments:
        return segments
    cleaned: List[SubtitleSegment] = []
    prev_text = ""
    for seg in segments:
        text = seg.text
        if not text:
            continue
        if prev_text and text.startswith(prev_text):
            new_part = text[len(prev_text):].strip()
        elif prev_text and prev_text.endswith(text):
            new_part = ""  # fully-seen tail re-displayed as the old line scrolls off
        else:
            new_part = text  # no overlap with the previous cue - genuinely new content
        prev_text = text
        if new_part:
            cleaned.append(SubtitleSegment(start=seg.start, end=seg.end, text=new_part))
    return cleaned


# --------------------------------------------------------------------------- #
# YouTube / local subtitle ingestion
# --------------------------------------------------------------------------- #


class YouTubeSubtitleFetcher:
    """
    Fetches subtitle segments either from a YouTube URL (via yt-dlp) or from
    a pre-existing local .srt / .vtt / .txt transcript file.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cfg = settings.fetcher

    def fetch(self, source: str) -> List[SubtitleSegment]:
        """`source` is either a YouTube URL or a path to a local subtitle file."""
        path = Path(source)
        if path.suffix.lower() in {".srt", ".vtt", ".txt"}:
            if not path.exists():
                raise SubtitleFetchError(f"Local subtitle file not found: {source}")
            return self._parse_local_file(path)
        return self._fetch_from_youtube(source)

    def _fetch_from_youtube(self, url: str) -> List[SubtitleSegment]:
        video_id_match = YOUTUBE_ID_RE.search(url)
        cache_id = video_id_match.group(1) if video_id_match else _cache_key(url)

        if self.cfg.cache_enabled:
            cached = self._select_best_subtitle_file(cache_id)
            if cached is not None:
                logger.info("Using cached subtitles for %s (%s)", cache_id, cached.name)
                return self._parse_vtt(cached)

        out_template = str(self.cache_dir / f"yt_{cache_id}.%(ext)s")
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": self.cfg.prefer_manual_subtitles,
            "writeautomaticsub": True,
            "subtitleslangs": self.cfg.subtitle_langs,
            "subtitlesformat": "vtt",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
            "retries": self.cfg.ytdlp_retries,
            "socket_timeout": self.cfg.ytdlp_socket_timeout_s,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            raise SubtitleFetchError(f"yt-dlp failed to fetch subtitles for '{url}': {exc}") from exc

        best = self._select_best_subtitle_file(cache_id)
        if best is None:
            raise SubtitleFetchError(
                f"No subtitles (manual or automatic) found for languages "
                f"{self.cfg.subtitle_langs} on '{url}'."
            )
        logger.info("Selected '%s' subtitle track for %s", best.name, cache_id)
        return self._parse_vtt(best)

    def _select_best_subtitle_file(self, cache_id: str) -> Optional[Path]:
        """
        yt-dlp downloads one file per matching language in self.cfg.subtitle_langs
        (e.g. 'yt_<id>.ru.vtt' and 'yt_<id>.en.vtt' side by side) — a plain glob
        would pick whichever sorts first alphabetically, silently ignoring our
        configured language priority. Walk the priority list explicitly instead.
        """
        for lang in self.cfg.subtitle_langs:
            matches = sorted(self.cache_dir.glob(f"yt_{cache_id}.{lang}.vtt"))
            if matches:
                return matches[0]
        # yt-dlp sometimes normalizes/renames the language code we asked for;
        # fall back to whatever got downloaded rather than failing outright.
        fallback = sorted(self.cache_dir.glob(f"yt_{cache_id}*.vtt"))
        return fallback[0] if fallback else None

    def _parse_local_file(self, path: Path) -> List[SubtitleSegment]:
        suffix = path.suffix.lower()
        if suffix == ".vtt":
            return self._parse_vtt(path)
        if suffix == ".srt":
            return self._parse_srt(path)
        if suffix == ".txt":
            return self._parse_txt(path)
        raise SubtitleFetchError(f"Unsupported subtitle file type: {suffix}")

    @staticmethod
    def _parse_vtt(path: Path) -> List[SubtitleSegment]:
        try:
            segments = [
                SubtitleSegment(
                    start=caption.start_in_seconds,
                    end=caption.end_in_seconds,
                    text=_clean_subtitle_text(caption.text),
                )
                for caption in webvtt.read(str(path))
            ]
        except Exception as exc:  # webvtt raises its own MalformedFileError subclasses
            raise SubtitleFetchError(f"Failed to parse VTT file '{path}': {exc}") from exc
        return _dedupe_segments(segments)

    @staticmethod
    def _parse_srt(path: Path) -> List[SubtitleSegment]:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            subs = list(srt_lib.parse(content))
        except Exception as exc:
            raise SubtitleFetchError(f"Failed to parse SRT file '{path}': {exc}") from exc

        segments = [
            SubtitleSegment(
                start=sub.start.total_seconds(),
                end=sub.end.total_seconds(),
                text=_clean_subtitle_text(sub.content),
            )
            for sub in subs
        ]
        return _dedupe_segments(segments)

    @staticmethod
    def _parse_txt(path: Path) -> List[SubtitleSegment]:
        """
        Fallback parser for plain-text transcripts using either:
            [HH:MM:SS.mmm --> HH:MM:SS.mmm] Text
        or a single leading timestamp per line (end inferred from the next cue):
            HH:MM:SS.mmm Text
        """
        pair_re = re.compile(
            r"\[?(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\]?\s*(.*)"
        )
        single_re = re.compile(r"\[?(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\]?\s*(.*)")

        raw_cues: List[dict] = []  # {"start": float, "end": Optional[float], "text": str}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            pair_match = pair_re.match(line)
            if pair_match:
                start_s, end_s, text = pair_match.groups()
                raw_cues.append({"start": _parse_timestamp(start_s), "end": _parse_timestamp(end_s), "text": text})
                continue
            single_match = single_re.match(line)
            if single_match:
                start_s, text = single_match.groups()
                raw_cues.append({"start": _parse_timestamp(start_s), "end": None, "text": text})
                continue
            if raw_cues:
                raw_cues[-1]["text"] = f"{raw_cues[-1]['text']} {line}".strip()

        if not raw_cues:
            raise SubtitleFetchError(
                f"Could not find any timestamped lines in '{path}'. "
                "Expected '[HH:MM:SS.mmm --> HH:MM:SS.mmm] text' or 'HH:MM:SS.mmm text' per line."
            )

        raw_cues.sort(key=lambda c: c["start"])
        segments: List[SubtitleSegment] = []
        for i, cue in enumerate(raw_cues):
            end = cue["end"]
            if end is None:
                end = raw_cues[i + 1]["start"] if i + 1 < len(raw_cues) else cue["start"] + 2.0
            text = _clean_subtitle_text(cue["text"])
            if text:
                segments.append(SubtitleSegment(start=cue["start"], end=end, text=text))
        return _dedupe_segments(segments)


# --------------------------------------------------------------------------- #
# Twitch chat ingestion
# --------------------------------------------------------------------------- #


class TwitchChatFetcher:
    """
    Downloads a Twitch VOD's chat log via TwitchDownloaderCLI and parses it
    into timestamp-aligned ChatMessage records.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.binaries = settings.binaries
        self.cfg = settings.fetcher

    def fetch(
        self,
        vod_url_or_id: str,
        chat_offset_seconds: Optional[float] = None,
    ) -> List[ChatMessage]:
        """
        `chat_offset_seconds` corrects for the Twitch VOD clock running ahead
        of the eventual YouTube video clock, e.g. because the YouTube upload
        trimmed N seconds off the front of the stream:

            youtube_time = twitch_time - chat_offset_seconds

        Messages that land before youtube_time == 0 are dropped.
        """
        vod_id = extract_twitch_vod_id(vod_url_or_id)
        json_path = self._download_chat(vod_id)
        messages = self._parse_chat_json(json_path)

        offset = (
            chat_offset_seconds if chat_offset_seconds is not None else self.cfg.default_chat_offset_seconds
        )
        if offset:
            shifted = [
                ChatMessage(
                    timestamp=msg.timestamp - offset,
                    username=msg.username,
                    text=msg.text,
                    badges=msg.badges,
                    is_subscriber=msg.is_subscriber,
                    is_moderator=msg.is_moderator,
                )
                for msg in messages
            ]
            messages = [msg for msg in shifted if msg.timestamp >= 0]
        return messages

    def _download_chat(self, vod_id: str) -> Path:
        cli_path = self.binaries.twitch_downloader_cli
        if not cli_path.exists():
            raise BinaryNotFoundError(
                f"TwitchDownloaderCLI not found at '{cli_path}'. "
                "Set TWITCH_DOWNLOADER_CLI_PATH in your .env or place the binary in ./bin/."
            )

        out_path = self.cache_dir / f"twitch_chat_{vod_id}.json"
        if self.cfg.cache_enabled and out_path.exists():
            logger.info("Using cached chat log for VOD %s", vod_id)
            return out_path

        cmd = [
            str(cli_path),
            "chatdownload",
            "--id", vod_id,
            "--embed-images", "false",
            "-o", str(out_path),
        ]
        logger.info("Downloading Twitch chat for VOD %s ...", vod_id)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.cfg.twitch_download_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ChatFetchError(f"TwitchDownloaderCLI timed out for VOD {vod_id}") from exc

        if result.returncode != 0 or not out_path.exists():
            raise ChatFetchError(
                f"TwitchDownloaderCLI failed for VOD {vod_id} "
                f"(exit {result.returncode}): {result.stderr.strip()}"
            )
        return out_path

    @staticmethod
    def _parse_chat_json(json_path: Path) -> List[ChatMessage]:
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ChatFetchError(f"Failed to read/parse chat JSON '{json_path}': {exc}") from exc

        comments = raw.get("comments", [])
        if not comments:
            raise ChatFetchError(f"Chat JSON '{json_path}' contains no comments.")

        messages: List[ChatMessage] = []
        for comment in comments:
            try:
                timestamp = float(comment["content_offset_seconds"])
                message_obj = comment["message"]
                text = message_obj.get("body", "")
                username = comment.get("commenter", {}).get("display_name", "unknown")
                badge_ids = [b.get("_id", "") for b in (message_obj.get("user_badges") or [])]
            except (KeyError, TypeError, ValueError):
                continue
            messages.append(
                ChatMessage(
                    timestamp=timestamp,
                    username=username,
                    text=text,
                    badges=badge_ids,
                    is_subscriber="subscriber" in badge_ids,
                    is_moderator="moderator" in badge_ids,
                )
            )
        return sorted(messages, key=lambda m: m.timestamp)


def get_twitch_vod_title(vod_url_or_id: str, cache_dir: Path = CACHE_DIR) -> Optional[str]:
    """
    Best-effort lookup of a VOD's stream title from its already-downloaded chat
    JSON (TwitchDownloaderCLI embeds a "video" object with the title alongside
    the comments). Reads only the existing cache file - never triggers a fresh
    download - since this is meant to be called after fetch_twitch_chat has
    already run (e.g. for naming a saved session), not as a standalone fetch.
    Returns None on any failure (no source given, nothing cached yet, unparseable
    file) rather than raising - a title is a nice-to-have, not load-bearing.
    """
    if not vod_url_or_id:
        return None
    try:
        vod_id = extract_twitch_vod_id(vod_url_or_id)
    except ChatFetchError:
        return None

    json_path = cache_dir / f"twitch_chat_{vod_id}.json"
    if not json_path.exists():
        return None
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    title = (raw.get("video") or {}).get("title")
    return title.strip() if isinstance(title, str) and title.strip() else None


# --------------------------------------------------------------------------- #
# Twitch VOD video ingestion
# --------------------------------------------------------------------------- #


class TwitchVideoFetcher:
    """
    Downloads a Twitch VOD's video file via TwitchDownloaderCLI, so exporters
    have a local source file to reference.

    Always downloads the WHOLE VOD, never a trimmed range - even though the
    CLI supports -b/-e trim flags, every clip's start_time/end_time in the
    FCPXML/EDL/Resolve timeline is an offset from the VOD's own t=0, so a
    partial download would desync every clip's timing against the file.
    """

    def __init__(self, downloads_dir: Path = DOWNLOADS_DIR):
        self.downloads_dir = downloads_dir
        self.binaries = settings.binaries
        self.cfg = settings.fetcher

    def fetch(self, vod_url_or_id: str, quality: Optional[str] = None) -> Path:
        vod_id = extract_twitch_vod_id(vod_url_or_id)
        quality = quality or self.cfg.twitch_video_quality
        out_path = self.downloads_dir / f"twitch_vod_{vod_id}.mp4"

        if self.cfg.cache_enabled and out_path.exists():
            logger.info("Using already-downloaded VOD file for %s", vod_id)
            return out_path

        cli_path = self.binaries.twitch_downloader_cli
        if not cli_path.exists():
            raise BinaryNotFoundError(
                f"TwitchDownloaderCLI not found at '{cli_path}'. "
                "Set TWITCH_DOWNLOADER_CLI_PATH in your .env or place the binary in ./bin/."
            )

        cmd = [
            str(cli_path),
            "videodownload",
            "--id", vod_id,
            "--quality", quality,
            "-o", str(out_path),
            # Non-interactive: without this, a stale/partial file from a prior failed run
            # would make TwitchDownloaderCLI block on an interactive overwrite prompt,
            # which would just hang until our timeout fires instead of failing fast.
            "--collision", "Overwrite",
        ]
        logger.info(
            "Downloading Twitch VOD %s at quality '%s' - this can take a while for long streams.",
            vod_id, quality,
        )
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.cfg.twitch_video_download_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise VideoFetchError(
                f"TwitchDownloaderCLI timed out downloading VOD {vod_id} "
                f"(limit {self.cfg.twitch_video_download_timeout_s}s). Increase "
                "TWITCH_VIDEO_DOWNLOAD_TIMEOUT_S in .env for longer streams or slower connections."
            ) from exc

        if result.returncode != 0 or not out_path.exists():
            raise VideoFetchError(
                f"TwitchDownloaderCLI failed to download VOD {vod_id} "
                f"(exit {result.returncode}): {result.stderr.strip()}"
            )
        logger.info("Downloaded VOD %s -> %s", vod_id, out_path)
        return out_path


# --------------------------------------------------------------------------- #
# Convenience module-level API
# --------------------------------------------------------------------------- #


def fetch_subtitles(source: str) -> List[SubtitleSegment]:
    """Fetch subtitles from a YouTube URL or local .srt/.vtt/.txt file."""
    return YouTubeSubtitleFetcher().fetch(source)


def fetch_twitch_chat(
    vod_url_or_id: str,
    chat_offset_seconds: Optional[float] = None,
) -> List[ChatMessage]:
    """Fetch and offset-correct a Twitch VOD's chat log."""
    return TwitchChatFetcher().fetch(vod_url_or_id, chat_offset_seconds=chat_offset_seconds)


def fetch_twitch_vod(vod_url_or_id: str, quality: Optional[str] = None) -> Path:
    """Downloads (or reuses a cached) Twitch VOD video file. Returns the local path."""
    return TwitchVideoFetcher().fetch(vod_url_or_id, quality=quality)
