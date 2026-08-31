"""
core/audio_analyzer.py

Audio peak-energy analysis, run independently of core/chat_analyzer.py so
chat-only analysis (no video downloaded yet) is never affected by whether this
is enabled. Extracts the VOD's audio track via ffmpeg, bins it the same way
chat messages are binned, and Z-score-detects loud moments (RMS energy spikes)
the same way ChatAnalyzer detects chat hype spikes - producing the same
ClipCandidate type so both detectors' outputs merge cleanly (see
merge_with_chat_candidates below) instead of needing a parallel type.

extract_pcm_waveform() is deliberately its own function, not inlined into the
RMS computation: Phase 2 (YAMNet sound-event classification via ONNX) will
need the exact same 16kHz mono waveform this module already extracts, and
should be able to call this directly rather than duplicating the ffmpeg
invocation or requiring a second decode pass.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config import AUDIO_RMS_CACHE_DIR, AudioScoreConfig, settings
from core.chat_analyzer import ClipCandidate

logger = logging.getLogger(__name__)

_SCORED_COLUMNS = ["bin_start", "bin_end", "rms", "rolling_mean", "rolling_std", "z_score"]


class AudioAnalysisError(Exception):
    """Raised when the audio track can't be extracted or analyzed."""


# Single-entry in-memory cache for the decoded waveform, so a pipeline run that has
# both audio peak analysis AND sound event classification enabled decodes the source
# once instead of twice - they run back to back on the same file at the same sample
# rate, producing byte-identical output, so the second decode was pure waste.
#
# Deliberately single-entry rather than an lru_cache: at ~230MB/hour a long VOD's
# waveform is over a gigabyte, so an unbounded memo would be a genuine memory leak
# dressed up as an optimisation. One entry caps the cost at exactly one waveform, a
# different video evicts the previous one, and release_cached_waveform() lets the
# pipeline drop it the moment the audio stages are done.
_waveform_cache_key: Optional[tuple] = None
_waveform_cache_value: Optional[np.ndarray] = None


def release_cached_waveform() -> None:
    """Drops the cached waveform. Call once the audio stages of a run are finished -
    holding a multi-GB array alive for the rest of a long LLM pass is exactly the
    kind of invisible memory growth this cache must not cause."""
    global _waveform_cache_key, _waveform_cache_value
    _waveform_cache_key = None
    _waveform_cache_value = None


def _waveform_identity(path: Path, sample_rate: int) -> Optional[tuple]:
    """Content identity, matching the (path|size|mtime|params) shape used by every
    on-disk cache key in this project. None if the file can't be stat'd, which
    disables caching rather than risking a stale hit."""
    try:
        stat = path.stat()
        return (str(path.resolve()), stat.st_size, stat.st_mtime, sample_rate)
    except OSError:
        return None


def extract_pcm_waveform(
    video_path: str, sample_rate: int = 16000, ffmpeg_binary: str = "ffmpeg", timeout_s: int = 1800,
) -> np.ndarray:
    """
    Decodes video_path's audio track to mono float32 PCM at sample_rate, piped
    directly through stdout - no intermediate file, no video re-encode, so this
    stays fast even on a multi-hour VOD. Returns the raw waveform (values in
    [-1, 1]) as a 1-D array; callers bin/window it themselves.

    Not cached to disk on purpose: at 16kHz mono float32 this is already
    ~230MB/hour, and audio-only decode is fast enough (well under realtime)
    that re-decoding from the source video is cheaper than storing it. What IS
    cached is the much smaller derived RMS timeline - see AudioAnalyzer below.
    It IS cached in memory for the duration of one run - see the single-entry
    cache above.
    """
    global _waveform_cache_key, _waveform_cache_value

    path = Path(video_path) if video_path else None
    if not path or not path.exists():
        raise AudioAnalysisError(f"Source video not found: {video_path or '(none set)'}")

    identity = _waveform_identity(path, sample_rate)
    if identity is not None and identity == _waveform_cache_key and _waveform_cache_value is not None:
        logger.info("Reusing the already-decoded waveform for '%s'.", path.name)
        return _waveform_cache_value

    try:
        result = subprocess.run(
            [
                ffmpeg_binary, "-i", str(path),
                "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-",
            ],
            capture_output=True, timeout=timeout_s, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioAnalysisError(f"Could not run ffmpeg ('{ffmpeg_binary}'): {exc}") from exc

    if result.returncode != 0 or not result.stdout:
        stderr_tail = (
            result.stderr.decode("utf-8", errors="replace").strip()[-500:] if result.stderr else "(no stderr)"
        )
        raise AudioAnalysisError(f"ffmpeg failed to extract audio (exit {result.returncode}): {stderr_tail}")

    waveform = np.frombuffer(result.stdout, dtype=np.float32)
    if identity is not None:
        _waveform_cache_key, _waveform_cache_value = identity, waveform
    return waveform


class AudioAnalyzer:
    """Turns a VOD's audio track into a scored RMS timeline and ranked spike candidates."""

    def __init__(self, config: AudioScoreConfig = None):
        self.cfg = config or settings.audio

    # --- public API ------------------------------------------------------------

    def compute_rms_timeline(self, video_path: str) -> pd.DataFrame:
        """Returns the full per-bin RMS timeline (for the graph overlay), cached per video."""
        cache_path = self._cache_path(video_path)
        if cache_path.exists():
            try:
                with np.load(cache_path) as data:
                    return pd.DataFrame({col: data[col] for col in _SCORED_COLUMNS})
            except Exception as exc:  # corrupt/partial cache file - recompute rather than fail
                logger.warning("Audio RMS cache at '%s' unreadable (%s); recomputing.", cache_path, exc)

        waveform = extract_pcm_waveform(
            video_path, sample_rate=self.cfg.sample_rate,
            ffmpeg_binary=settings.export.ffmpeg_binary, timeout_s=self.cfg.extraction_timeout_s,
        )
        scored = self._bin_and_score(waveform)

        try:
            np.savez(cache_path, **{col: scored[col].to_numpy() for col in _SCORED_COLUMNS})
        except OSError as exc:  # best-effort cache; analysis still works without it
            logger.warning("Could not write audio RMS cache to '%s': %s", cache_path, exc)

        return scored

    def analyze_with_timeline(self, video_path: str) -> Tuple[List[ClipCandidate], pd.DataFrame]:
        """Returns (candidates, scored_timeline) computed in a single pass."""
        scored = self.compute_rms_timeline(video_path)
        if scored.empty:
            return [], scored
        return self._detect_spikes(scored), scored

    def analyze(self, video_path: str) -> List[ClipCandidate]:
        """Convenience wrapper around analyze_with_timeline."""
        candidates, _ = self.analyze_with_timeline(video_path)
        return candidates

    # --- caching -----------------------------------------------------------

    def _cache_path(self, video_path: str) -> Path:
        source = Path(video_path)
        try:
            stat = source.stat()
            identity = (
                f"{source.resolve()}|{stat.st_size}|{stat.st_mtime}|{self.cfg.bin_seconds}|{self.cfg.sample_rate}"
            )
        except OSError:
            identity = f"{video_path}|{self.cfg.bin_seconds}|{self.cfg.sample_rate}"
        key = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
        return AUDIO_RMS_CACHE_DIR / f"{key}.npz"

    # --- binning / scoring ---------------------------------------------------

    def _bin_and_score(self, waveform: np.ndarray) -> pd.DataFrame:
        samples_per_bin = max(1, int(self.cfg.sample_rate * self.cfg.bin_seconds))
        n_bins = len(waveform) // samples_per_bin
        if n_bins == 0:
            return pd.DataFrame(columns=_SCORED_COLUMNS)

        trimmed = waveform[: n_bins * samples_per_bin].reshape(n_bins, samples_per_bin)
        rms = np.sqrt(np.mean(np.square(trimmed, dtype=np.float64), axis=1))

        df = pd.DataFrame({
            "bin_start": np.arange(n_bins) * self.cfg.bin_seconds,
            "bin_end": (np.arange(n_bins) + 1) * self.cfg.bin_seconds,
            "rms": rms,
        })

        window = self.cfg.rolling_window_bins
        rolling = df["rms"].rolling(window=window, min_periods=max(3, window // 4))
        df["rolling_mean"] = rolling.mean()
        rolling_std = rolling.std().replace(0, np.nan)
        df["z_score"] = ((df["rms"] - df["rolling_mean"]) / rolling_std).fillna(0.0)
        df["rolling_mean"] = df["rolling_mean"].fillna(df["rms"])
        df["rolling_std"] = rolling_std.fillna(0.0)
        return df

    # --- spike detection / candidate windows --------------------------------

    def _detect_spikes(self, scored: pd.DataFrame) -> List[ClipCandidate]:
        threshold = self.cfg.z_score_threshold
        spike_rows = scored[scored["z_score"] >= threshold].sort_values("bin_start")
        if spike_rows.empty:
            return []

        min_gap = self.cfg.min_seconds_between_spikes
        selected: List[pd.Series] = []
        for _, row in spike_rows.iterrows():
            if selected and (row["bin_start"] - selected[-1]["bin_start"]) < min_gap:
                if row["z_score"] > selected[-1]["z_score"]:
                    selected[-1] = row
                continue
            selected.append(row)

        candidates = [
            ClipCandidate(
                window_start=float(max(0.0, row["bin_start"] - self.cfg.pre_spike_seconds)),
                window_end=float(row["bin_end"] + self.cfg.post_spike_seconds),
                spike_time=float(row["bin_start"]),
                peak_hype_score=float(row["rms"]),
                peak_z_score=float(row["z_score"]),
                source="audio",
            )
            for row in selected
        ]
        return self._merge_overlapping(candidates, self.cfg.max_merged_duration_seconds)

    @staticmethod
    def _merge_overlapping(candidates: List[ClipCandidate], max_duration: float) -> List[ClipCandidate]:
        """Same chaining logic as ChatAnalyzer._merge_overlapping - see that docstring."""
        if not candidates:
            return candidates
        candidates = sorted(candidates, key=lambda c: c.window_start)
        merged: List[ClipCandidate] = [candidates[0]]
        for cand in candidates[1:]:
            last = merged[-1]
            would_be_end = max(last.window_end, cand.window_end)
            would_be_duration = would_be_end - last.window_start
            if cand.window_start > last.window_end or would_be_duration > max_duration:
                merged.append(cand)
                continue
            stronger = cand if cand.peak_z_score > last.peak_z_score else last
            merged[-1] = replace(
                last,
                window_end=would_be_end,
                spike_time=stronger.spike_time,
                peak_hype_score=stronger.peak_hype_score,
                peak_z_score=stronger.peak_z_score,
            )
        return merged


# Fixed display order for ClipCandidate.source combinations - see _add_source_tag. New
# detectors (e.g. a future third or fourth signal) just add their tag here; nothing else
# needs to change to keep combined source strings ordered consistently.
_SOURCE_TAG_ORDER = ["chat", "audio", "sound_event"]


def _add_source_tag(existing_source: str, tag: str) -> str:
    """
    Adds `tag` to a ClipCandidate.source, keeping the "+"-joined result in a fixed,
    predictable order regardless of which detector ran first or which merge order
    was used - e.g. adding "audio" to "chat" always gives "chat+audio", never
    "audio+chat", so app.py's exact-string checks (and this module's own) stay valid
    no matter which merge function touched the candidate most recently.
    """
    tags = set(existing_source.split("+")) | {tag}
    return "+".join(t for t in _SOURCE_TAG_ORDER if t in tags)


def _same_moment(a: ClipCandidate, b: ClipCandidate, tolerance_s: float) -> bool:
    """
    "Same real-world moment" is decided by spike_time proximity, NOT by comparing
    the two candidates' padded windows - pre_spike_seconds/post_spike_seconds exist
    to give the LLM transcript context, not to define real-world proximity, and
    with 60s/30s padding on each side a padded-window overlap check would let two
    editorially distinct spikes up to ~90s apart falsely "confirm" each other.
    """
    return abs(a.spike_time - b.spike_time) <= tolerance_s


def merge_with_chat_candidates(
    chat_candidates: List[ClipCandidate],
    audio_candidates: List[ClipCandidate],
    allow_new_candidates: bool,
    overlap_tolerance_s: float = 30.0,
) -> List[ClipCandidate]:
    """
    Reconciles the two independently-detected candidate lists by merging their
    OUTPUT windows, not their raw Z-scores pre-detection - each detector stays
    self-contained and independently legible/tunable (chat spike thresholds
    never need retuning because of how sensitive audio is, or vice versa).

    An audio spike within tolerance of a chat candidate enriches that candidate
    (audio_peak_z_score/audio_peak_time set, source gains the "audio" tag) instead of creating a
    near-duplicate entry for the same real moment. An audio spike with no nearby
    chat candidate becomes its own standalone candidate only if
    allow_new_candidates is True; otherwise it's dropped, since with the toggle
    off audio should only ever add context to what chat already flagged, never
    surface a moment chat missed entirely.

    A third detector (e.g. sound-event classification) doesn't extend this
    function - it composes on top via its own merge_sound_events(), applied to
    this function's output. See that function's docstring.
    """

    enriched: List[ClipCandidate] = []
    for cc in chat_candidates:
        nearby = [ac for ac in audio_candidates if _same_moment(ac, cc, overlap_tolerance_s)]
        if nearby:
            best = max(nearby, key=lambda a: a.peak_z_score)
            cc = replace(
                cc, source=_add_source_tag(cc.source, "audio"),
                audio_peak_z_score=best.peak_z_score, audio_peak_time=best.spike_time,
            )
        enriched.append(cc)

    result = enriched
    if allow_new_candidates:
        unmatched_audio = [
            ac for ac in audio_candidates if not any(_same_moment(ac, cc, overlap_tolerance_s) for cc in chat_candidates)
        ]
        result = result + unmatched_audio

    return sorted(result, key=lambda c: c.window_start)
