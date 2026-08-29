"""
core/transcriber.py

Local transcription via a bundled whisper.cpp CLI. core/whisper_setup.py owns
install/detect (does the binary/model exist); this module only ever assumes a
ready binary+model and runs them - same split core/ollama_setup.py (install/
detect) keeps from core/llm_agent.py (actual usage).

Two-step pipeline per video: ffmpeg extracts a 16kHz mono WAV (whisper.cpp's
required input format - reuses the exact ffmpeg invocation shape
core/audio_analyzer.extract_pcm_waveform already uses, but to a real file
rather than piped stdout, since whisper-cli needs one), then whisper-cli
transcribes that WAV to an .srt file. That path is handed straight back to
core/fetchers.YouTubeSubtitleFetcher.fetch(), which already accepts any local
.srt path transparently - nothing downstream needs to know this came from
Whisper rather than YouTube. Its timestamps are natively on source_video_path's
own clock (see core/fetchers.py's module docstring) - never run through
shift_subtitles_to_vod_clock; see core.fetchers.is_local_subtitle_source.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterator, Optional

from config import CACHE_DIR, settings

logger = logging.getLogger(__name__)

_DLL_LOAD_FAILURE_HINT = (
    " This often means the Microsoft Visual C++ Redistributable isn't installed - "
    "get it from https://aka.ms/vs/17/release/vc_redist.x64.exe and try again."
)
# Windows process-crash-style exit codes (STATUS_DLL_NOT_FOUND and friends show up
# as large/negative returncodes, not a clean nonzero exit) - worth a specific hint
# rather than a bare "exited with code -1073741515", the same class of dependency
# issue already hit with DaVinci Resolve's fusionscript.dll earlier this project.
_DLL_LOAD_FAILURE_CODES = {-1073741515, 3221225781}


class TranscriptionError(Exception):
    """Raised when audio extraction or whisper-cli invocation fails."""


def _cache_key(video_path: Path, model_path: Path, language: str) -> str:
    """Mirrors core.audio_analyzer.AudioAnalyzer._cache_path exactly (path + size +
    mtime + the params that change the output) - a different model tier or
    language choice invalidates the cache the same way a different bin_seconds
    invalidates the RMS cache there."""
    try:
        stat = video_path.stat()
        identity = f"{video_path.resolve()}|{stat.st_size}|{stat.st_mtime}|{model_path.name}|{language}"
    except OSError:
        identity = f"{video_path}|{model_path.name}|{language}"
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]


def _srt_cache_path(video_path: Path, model_path: Path, language: str, cache_dir: Path) -> Path:
    return cache_dir / f"whisper_{_cache_key(video_path, model_path, language)}.srt"


def _extract_wav(video_path: Path, dest_wav: Path, ffmpeg_binary: str, timeout_s: int) -> None:
    """16kHz mono 16-bit PCM WAV - whisper.cpp's required input format. '-y' is
    required here (unlike extract_pcm_waveform, which pipes to stdout with nothing
    to collide with) so a stale WAV left over from a crashed prior run doesn't hang
    on an interactive overwrite prompt."""
    try:
        result = subprocess.run(
            [
                ffmpeg_binary, "-y", "-i", str(video_path),
                "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dest_wav),
            ],
            capture_output=True, timeout=timeout_s, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TranscriptionError(f"Could not run ffmpeg ('{ffmpeg_binary}') to extract audio: {exc}") from exc
    if result.returncode != 0 or not dest_wav.exists():
        stderr_tail = (
            result.stderr.decode("utf-8", errors="replace").strip()[-500:] if result.stderr else "(no stderr)"
        )
        raise TranscriptionError(f"ffmpeg failed to extract audio (exit {result.returncode}): {stderr_tail}")


def _run_whisper_cli(
    wav_path: Path, model_path: Path, out_prefix: Path, language: str,
    binary_path: Path, threads: int, timeout_s: int,
) -> Iterator[str]:
    """Yields raw stdout lines as live progress text - whisper-cli prints each
    recognized segment with its timestamp as it transcribes, unlike Ollama's
    structured /api/pull JSON progress, so this is just unstructured text, which
    is fine for a spinner-style status readout."""
    cmd = [
        str(binary_path), "-m", str(model_path), "-f", str(wav_path),
        "-l", language, "-t", str(threads), "-osrt", "-of", str(out_prefix),
    ]
    if settings.whisper.max_context >= 0:
        # See config.WhisperConfig.max_context - unlimited context (whisper-cli's own
        # default) let a single hallucinated caption on a real 5.5h VOD self-perpetuate
        # for 83 minutes straight once triggered.
        cmd.extend(["-mc", str(settings.whisper.max_context)])
    try:
        # encoding="utf-8" explicitly - whisper-cli emits UTF-8 regardless of the
        # system locale, but text=True alone decodes via Python's platform-default
        # encoding (locale.getpreferredencoding()), which on a Russian-locale
        # Windows machine is cp1251, not UTF-8. Left to that default, most UTF-8
        # Cyrillic byte sequences still "successfully" decode under cp1251 - just
        # as mojibake - and the rest raise UnicodeDecodeError outright.
        # errors="replace" as a safety net for any genuinely non-UTF-8 byte that
        # slips in (e.g. a stray console control sequence).
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
    except OSError as exc:
        raise TranscriptionError(f"Could not run whisper-cli ('{binary_path}'): {exc}") from exc

    deadline = time.monotonic() + timeout_s
    lines_tail: list = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            lines_tail.append(line)
            lines_tail = lines_tail[-20:]
            yield line
        if time.monotonic() > deadline:
            proc.kill()
            raise TranscriptionError(f"whisper-cli timed out after {timeout_s}s.")

    returncode = proc.wait()
    if returncode != 0:
        tail = "\n".join(lines_tail)
        hint = _DLL_LOAD_FAILURE_HINT if returncode in _DLL_LOAD_FAILURE_CODES else ""
        raise TranscriptionError(f"whisper-cli exited with code {returncode}:\n{tail}{hint}")


def transcribe_locally(
    video_path: str,
    model_path: Path,
    language: Optional[str] = None,
    binary_path: Optional[Path] = None,
    ffmpeg_binary: Optional[str] = None,
    threads: Optional[int] = None,
    cache_dir: Path = CACHE_DIR,
) -> Iterator[dict]:
    """
    Yields {'status': str} progress lines; the final yield is
    {'done': True, 'srt_path': Path}. Caches the resulting .srt keyed on (video
    identity, model, language) - see _cache_key - so re-running the same video/
    model/language combo (e.g. after tweaking analysis settings and re-clicking)
    is instant instead of re-transcribing.
    """
    cfg = settings.whisper
    language = language or cfg.default_language
    binary_path = binary_path or cfg.binary_path
    ffmpeg_binary = ffmpeg_binary or settings.export.ffmpeg_binary
    threads = threads or cfg.threads

    path = Path(video_path)
    if not path.exists():
        raise TranscriptionError(f"Source video not found: {video_path}")
    if not binary_path.exists():
        raise TranscriptionError(f"whisper-cli not found at '{binary_path}'.")
    if not model_path.exists():
        raise TranscriptionError(f"Whisper model not found at '{model_path}'.")

    out_srt = _srt_cache_path(path, model_path, language, cache_dir)
    if cfg.cache_enabled and out_srt.exists():
        logger.info("Using cached transcript for %s (%s)", path.name, out_srt.name)
        yield {"done": True, "srt_path": out_srt}
        return

    wav_path = Path(tempfile.gettempdir()) / f"vodblade_whisper_{_cache_key(path, model_path, language)}.wav"
    try:
        yield {"status": "Extracting audio..."}
        _extract_wav(path, wav_path, ffmpeg_binary, cfg.extraction_timeout_s)

        yield {"status": "Transcribing..."}
        out_prefix = out_srt.with_suffix("")  # whisper-cli appends .srt itself for -osrt
        for line in _run_whisper_cli(
            wav_path, model_path, out_prefix, language, binary_path, threads, cfg.transcription_timeout_s,
        ):
            yield {"status": line}

        if not out_srt.exists():
            raise TranscriptionError(f"whisper-cli finished but produced no output at '{out_srt}'.")
    finally:
        wav_path.unlink(missing_ok=True)  # only the small .srt is worth keeping in cache, not the WAV

    yield {"done": True, "srt_path": out_srt}
