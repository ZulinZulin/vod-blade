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
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

from config import CACHE_DIR, SCRATCH_DIR, settings
from core import whisper_setup

logger = logging.getLogger(__name__)

_DLL_LOAD_FAILURE_HINT = (
    " This often means the Microsoft Visual C++ Redistributable isn't installed - "
    "get it from https://aka.ms/vs/17/release/vc_redist.x64.exe and try again."
)
_CUDA_FAILURE_HINT = (
    " This looks like a GPU/CUDA problem rather than a problem with the audio. Your GPU "
    "may be too new or too old for this whisper.cpp build's compiled kernels. Remove the "
    "GPU build from 'Local transcription settings' to fall back to the CPU one, which "
    "works on any machine."
)
# Markers whisper.cpp prints on stdout/stderr that prove which backend is really in use.
# Necessary because the releases are built with GGML_BACKEND_DL=ON: a ggml-cuda.dll that
# fails to load is logged and swallowed, and the run silently continues on CPU. So
# "the CUDA build is installed" and "the CUDA build is being used" are different claims,
# and only the second one matters. _CUDA_ACTIVE_MARKER is the strong one - it means the
# model weights were actually allocated in a CUDA buffer, not merely that a DLL loaded.
_CUDA_ACTIVE_MARKER = "CUDA0 total size"
_CUDA_LOADED_MARKER = "loaded CUDA backend"
# Windows process-crash-style exit codes (STATUS_DLL_NOT_FOUND and friends show up
# as large/negative returncodes, not a clean nonzero exit) - worth a specific hint
# rather than a bare "exited with code -1073741515", the same class of dependency
# issue already hit with DaVinci Resolve's fusionscript.dll earlier this project.
_DLL_LOAD_FAILURE_CODES = {-1073741515, 3221225781}


class TranscriptionError(Exception):
    """Raised when audio extraction or whisper-cli invocation fails."""


class TranscriptionCancelled(TranscriptionError):
    """Raised specifically when cancel_active_transcription() killed a running
    whisper-cli process - a subclass of TranscriptionError so existing generic
    error handling still catches it, but callers that want to show a plain
    'cancelled' message instead of a scary failure one can catch this first."""


# Tracks the currently-running whisper-cli process (there's only ever meant to be
# one at a time - the UI disables the Generate button while one is in flight) so a
# separate Cancel button click, running as its own independent Gradio call with no
# direct reference to _run_whisper_cli's local Popen object, can still kill it.
# This is the actual, guaranteed kill mechanism - it doesn't depend on Gradio's own
# generator-cancellation semantics (which stop the UI from updating further, but
# aren't something to rely on alone to prove the child process really died).
_active_proc_lock = threading.Lock()
_active_proc: Optional[subprocess.Popen] = None
_cancel_requested = False


def cancel_active_transcription() -> bool:
    """Kills the currently-running whisper-cli process, if any. Returns whether
    there was one to kill. Safe to call even if nothing is running."""
    global _cancel_requested
    with _active_proc_lock:
        proc = _active_proc
        if proc is None or proc.poll() is not None:
            return False
        _cancel_requested = True
    proc.kill()
    return True


def _cache_key(video_path: Path, model_path: Path, language: str) -> str:
    """Mirrors core.audio_analyzer.AudioAnalyzer._cache_path exactly (path + size +
    mtime + the params that change the output) - a different model tier or
    language choice invalidates the cache the same way a different bin_seconds
    invalidates the RMS cache there.

    Deliberately does NOT include the backend (CPU vs CUDA). Their transcripts are
    not byte-identical - autoregressive decoding amplifies floating-point
    differences, so segment boundaries and the odd word vary - but they're
    quality-equivalent, so keying on backend would only force an expensive
    re-transcription every time someone installs or removes the GPU build, buying
    nothing. Reusing an existing transcript across a variant switch is the
    desirable behaviour here, not a bug."""
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
    binary_path: Path, threads: int, timeout_s: int, use_gpu: bool = True,
) -> Iterator[str]:
    """Yields raw stdout lines as live progress text - whisper-cli prints each
    recognized segment with its timestamp as it transcribes, unlike Ollama's
    structured /api/pull JSON progress, so this is just unstructured text, which
    is fine for a spinner-style status readout."""
    cmd = [
        str(binary_path), "-m", str(model_path), "-f", str(wav_path),
        "-l", language, "-t", str(threads), "-osrt", "-of", str(out_prefix),
    ]
    if not use_gpu:
        # Works on either build: the CUDA build honours -ng and stays on CPU, and the
        # CPU build accepts the flag as a no-op. So this is a pure runtime switch,
        # never a reason to reinstall a different variant.
        cmd.append("-ng")
    elif whisper_setup.is_cuda_binary(binary_path) and not settings.whisper.flash_attention:
        # Not optional in practice - with flash attention on, the GPU build silently
        # produced an entire hour of one repeated hallucinated caption on real audio
        # that the CPU transcribed correctly. See config.WhisperConfig.flash_attention
        # for the measurements. Scoped to the CUDA build and to actual GPU runs, so
        # the CPU path's behaviour is byte-for-byte unchanged.
        cmd.append("-nfa")
    if settings.whisper.max_context >= 0:
        # See config.WhisperConfig.max_context - unlimited context (whisper-cli's own
        # default) let a single hallucinated caption on a real 5.5h VOD self-perpetuate
        # from 04:09 to the end of the file. Applies to both backends: the loop was
        # reproduced identically on CPU on the same audio, so this isn't a GPU-only fix.
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

    global _active_proc, _cancel_requested
    with _active_proc_lock:
        _active_proc = proc
        _cancel_requested = False

    deadline = time.monotonic() + timeout_s
    lines_tail: list = []
    try:
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
    finally:
        # Runs on normal completion, on the timeout raise above, AND if this
        # generator gets closed out from under us (e.g. cancel_active_transcription
        # killed proc while we were mid-yield) - proc.poll() is None only in that
        # last case, since the other two paths already know proc has exited.
        with _active_proc_lock:
            was_cancelled = _cancel_requested
            if _active_proc is proc:
                _active_proc = None
        if proc.poll() is None:
            proc.kill()

    returncode = proc.wait()
    if was_cancelled:
        raise TranscriptionCancelled("Transcription cancelled.")
    if returncode != 0:
        tail = "\n".join(lines_tail)
        if returncode in _DLL_LOAD_FAILURE_CODES:
            hint = _DLL_LOAD_FAILURE_HINT
        elif any(m in tail for m in ("CUDA error", "no kernel image", "cuBLAS", "CUBLAS")):
            hint = _CUDA_FAILURE_HINT
        else:
            hint = ""
        raise TranscriptionError(f"whisper-cli exited with code {returncode}:\n{tail}{hint}")


def transcribe_locally(
    video_path: str,
    model_path: Path,
    language: Optional[str] = None,
    binary_path: Optional[Path] = None,
    ffmpeg_binary: Optional[str] = None,
    threads: Optional[int] = None,
    cache_dir: Path = CACHE_DIR,
    use_gpu: bool = True,
) -> Iterator[dict]:
    """
    Yields {'status': str} progress lines; the final yield is
    {'done': True, 'srt_path': Path}. Caches the resulting .srt keyed on (video
    identity, model, language) - see _cache_key - so re-running the same video/
    model/language combo (e.g. after tweaking analysis settings and re-clicking)
    is instant instead of re-transcribing.

    Which binary runs is resolved by whisper_setup.active_binary_path() (the GPU
    build if it's installed, else the CPU one) unless a caller passes binary_path
    explicitly. `use_gpu=False` forces CPU on either build via -ng, without needing
    a different install.
    """
    cfg = settings.whisper
    language = language or cfg.default_language
    binary_path = binary_path or whisper_setup.active_binary_path()
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

    wav_path = SCRATCH_DIR / f"vodblade_whisper_{_cache_key(path, model_path, language)}.wav"
    try:
        yield {"status": "Extracting audio..."}
        _extract_wav(path, wav_path, ffmpeg_binary, cfg.extraction_timeout_s)

        yield {"status": "Transcribing..."}
        out_prefix = out_srt.with_suffix("")  # whisper-cli appends .srt itself for -osrt
        # Report the backend actually in use exactly once, as soon as the output
        # proves it. See _CUDA_ACTIVE_MARKER: with GGML_BACKEND_DL a failed CUDA load
        # is swallowed and the run continues on CPU, so without this an unusably slow
        # "GPU" run looks identical to a working one.
        backend_reported = False
        for line in _run_whisper_cli(
            wav_path, model_path, out_prefix, language, binary_path, threads,
            cfg.transcription_timeout_s, use_gpu=use_gpu,
        ):
            if not backend_reported and _CUDA_ACTIVE_MARKER in line:
                backend_reported = True
                yield {"status": "Running on GPU (CUDA0)."}
            yield {"status": line}
        if not backend_reported:
            yield {"status": "Ran on CPU."}

        if not out_srt.exists():
            raise TranscriptionError(f"whisper-cli finished but produced no output at '{out_srt}'.")
    finally:
        wav_path.unlink(missing_ok=True)  # only the small .srt is worth keeping in cache, not the WAV

    yield {"done": True, "srt_path": out_srt}
