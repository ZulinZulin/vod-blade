"""
core/sound_event_classifier.py

Acoustic event detection (laughter, screaming, cheering, groaning) via YAMNet
running through onnxruntime. A third independent detector alongside
core/chat_analyzer.py and core/audio_analyzer.py - chat-only and audio-RMS
analysis both keep working unaffected if this is disabled or the model file
isn't present (see config.SoundEventConfig.validate()).

Reuses core.audio_analyzer.extract_pcm_waveform() rather than decoding the
video's audio a third time - both this and the RMS analyzer consume the exact
same 16kHz mono waveform.

Detection here is a plain threshold on a per-event confidence, not a rolling
Z-score like the other two detectors use - see SoundEventConfig's docstring for
why a classifier's already-normalized [0, 1] output doesn't need a
stream-relative baseline the way raw chat volume or audio energy do.

Each configured target "event" (e.g. "Laughter") is actually a SUM of several
raw AudioSet classes (see _EVENT_CLASS_GROUPS below), not one class thresholded
in isolation - confirmed against a real stream that a genuine laugh overlapping
speech splits its confidence across "Laughter"/"Snicker"/"Giggle" rather than
giving any single one a high score.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import SOUND_EVENT_CACHE_DIR, SoundEventConfig, settings
from core.audio_analyzer import _add_source_tag, _same_moment, extract_pcm_waveform
from core.chat_analyzer import ClipCandidate

logger = logging.getLogger(__name__)

# YAMNet scores each of its 521 classes independently (a per-class sigmoid, not one
# softmax distribution), but AudioSet's taxonomy has several near-synonym classes for
# the same real-world event - confirmed against a real stream where audible laughter
# overlapping speech never gave "Laughter" alone more than ~0.29 confidence, while
# "Snicker" simultaneously sat at 0.221 for the same moment. Summing each group's
# scores recovers that fragmented signal instead of thresholding one class in
# isolation and missing real events whose confidence is split across siblings.
_EVENT_CLASS_GROUPS: Dict[str, List[str]] = {
    "Laughter": ["Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh"],
    "Screaming": ["Screaming", "Yell", "Shout", "Children shouting"],
    "Cheering": ["Cheering", "Applause", "Whoop", "Hoot"],
    "Groan": ["Groan", "Wail, moan"],
}


# Bump this whenever a change to how the per-event confidence timeline is COMPUTED
# (e.g. _EVENT_CLASS_GROUPS membership) would make an existing cached .npz silently
# wrong rather than just missing - included in the cache key below so an old cache
# never gets reused across such a change without the video/model themselves changing.
_CACHE_SCHEMA_VERSION = 2


class SoundEventError(Exception):
    """Raised when the model/class map can't be loaded or classification fails."""


def load_class_names(class_map_path: Path) -> List[str]:
    """Returns display_name ordered by index (row 0 -> class 0, etc.) from yamnet_class_map.csv."""
    df = pd.read_csv(class_map_path)
    return df.sort_values("index")["display_name"].tolist()


class SoundEventClassifier:
    """Turns a VOD's audio track into a per-class confidence timeline and ranked event candidates."""

    def __init__(self, config: SoundEventConfig = None):
        self.cfg = config or settings.sound_event
        problems = self.cfg.validate()
        if problems:
            raise SoundEventError(" ".join(problems))

        # Imported lazily (not at module level) so importing this module - e.g. just to
        # reference SoundEventError/SoundEventConfig - never requires onnxruntime to be
        # installed unless sound event detection is actually used.
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(self.cfg.model_path), providers=["CPUExecutionProvider"])
        self._class_names = load_class_names(self.cfg.class_map_path)
        # Falls back to a single-class "group" of just itself for any target_classes entry
        # that isn't one of the known groupings above - keeps this working for an arbitrary
        # class name, not just the 4 shipped defaults.
        self._event_groups: Dict[str, List[str]] = {
            event: _EVENT_CLASS_GROUPS.get(event, [event]) for event in self.cfg.target_classes
        }
        unknown = [c for group in self._event_groups.values() for c in group if c not in self._class_names]
        if unknown:
            raise SoundEventError(
                f"Class names not found in the class map: {unknown}. Check spelling against "
                f"'{self.cfg.class_map_path}'."
            )
        self._target_indices: Dict[str, List[int]] = {
            event: [self._class_names.index(c) for c in group] for event, group in self._event_groups.items()
        }

    # --- public API ------------------------------------------------------------

    def compute_event_timeline(self, video_path: str) -> pd.DataFrame:
        """Per-frame confidence timeline for each configured target class, cached per video+model."""
        cache_path = self._cache_path(video_path)
        columns = ["bin_start", "bin_end", *self.cfg.target_classes]
        if cache_path.exists():
            try:
                with np.load(cache_path) as data:
                    if list(data["target_classes"]) == list(self.cfg.target_classes):
                        return pd.DataFrame({col: data[col] for col in columns})
            except Exception as exc:  # corrupt/partial/schema-mismatched cache - recompute
                logger.warning("Sound event cache at '%s' unreadable (%s); recomputing.", cache_path, exc)

        waveform = extract_pcm_waveform(
            video_path, sample_rate=self.cfg.sample_rate, ffmpeg_binary=settings.export.ffmpeg_binary,
        )
        if len(waveform) == 0:
            return pd.DataFrame(columns=columns)

        df = self._run_inference_chunked(waveform)
        if df.empty:
            return pd.DataFrame(columns=columns)

        try:
            np.savez(
                cache_path,
                target_classes=np.array(self.cfg.target_classes),
                **{col: df[col].to_numpy() for col in columns},
            )
        except OSError as exc:  # best-effort cache; analysis still works without it
            logger.warning("Could not write sound event cache to '%s': %s", cache_path, exc)

        return df

    def _run_inference_chunked(self, waveform: np.ndarray) -> pd.DataFrame:
        """
        Feeding an entire multi-hour waveform into one session.run() call blows up
        memory - confirmed against a real 5.7-hour VOD, where a single intermediate
        conv layer tried to allocate ~8.5GB, scaling with total input length rather
        than just the final per-frame output size. Chunking keeps each inference
        call's working memory bounded regardless of stream length.

        hop_seconds is computed once from the first chunk and reused for the rest -
        it's a fixed property of the model's own windowing, not of how much audio
        is fed in, confirmed by short test clips of very different lengths all
        producing the same ~0.48s hop.
        """
        chunk_samples = max(1, int(self.cfg.chunk_duration_s * self.cfg.sample_rate))
        bin_start_parts: List[np.ndarray] = []
        class_score_parts: Dict[str, List[np.ndarray]] = {cls: [] for cls in self.cfg.target_classes}
        hop_seconds = None

        for chunk_start_sample in range(0, len(waveform), chunk_samples):
            chunk = waveform[chunk_start_sample:chunk_start_sample + chunk_samples]
            scores, _embeddings, _spectrogram = self._session.run(None, {"waveform": chunk})
            n_frames = scores.shape[0]
            if n_frames == 0:
                continue
            if hop_seconds is None:
                hop_seconds = (len(chunk) / self.cfg.sample_rate) / n_frames
            chunk_start_s = chunk_start_sample / self.cfg.sample_rate
            bin_start_parts.append(chunk_start_s + np.arange(n_frames) * hop_seconds)
            for cls, indices in self._target_indices.items():
                # Summed across the group's raw AudioSet classes (see _EVENT_CLASS_GROUPS),
                # then clipped to 1.0 - these are independent per-class sigmoids, not one
                # softmax distribution, so a sum across several elevated siblings can
                # exceed 1.0 without clipping.
                class_score_parts[cls].append(np.clip(scores[:, indices].sum(axis=1), 0.0, 1.0))

        if not bin_start_parts:
            return pd.DataFrame(columns=["bin_start", "bin_end", *self.cfg.target_classes])

        bin_start = np.concatenate(bin_start_parts)
        df = pd.DataFrame({"bin_start": bin_start, "bin_end": bin_start + hop_seconds})
        for cls in self.cfg.target_classes:
            df[cls] = np.concatenate(class_score_parts[cls])
        return df

    def analyze_with_timeline(self, video_path: str) -> Tuple[List[ClipCandidate], pd.DataFrame]:
        """Returns (candidates, scored_timeline) computed in a single pass."""
        timeline = self.compute_event_timeline(video_path)
        if timeline.empty:
            return [], timeline
        return self._detect_events(timeline), timeline

    def analyze(self, video_path: str) -> List[ClipCandidate]:
        """Convenience wrapper around analyze_with_timeline."""
        candidates, _ = self.analyze_with_timeline(video_path)
        return candidates

    # --- caching -----------------------------------------------------------

    def _cache_path(self, video_path: str) -> Path:
        source = Path(video_path)
        try:
            stat = source.stat()
            model_stat = self.cfg.model_path.stat()
            identity = (
                f"{source.resolve()}|{stat.st_size}|{stat.st_mtime}|"
                f"{self.cfg.model_path.resolve()}|{model_stat.st_size}|{model_stat.st_mtime}|"
                f"{self.cfg.sample_rate}|{_CACHE_SCHEMA_VERSION}"
            )
        except OSError:
            identity = f"{video_path}|{self.cfg.model_path}|{self.cfg.sample_rate}|{_CACHE_SCHEMA_VERSION}"
        key = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
        return SOUND_EVENT_CACHE_DIR / f"{key}.npz"

    # --- event detection ------------------------------------------------------

    def _detect_events(self, timeline: pd.DataFrame) -> List[ClipCandidate]:
        per_class_candidates: List[ClipCandidate] = []
        for cls in self.cfg.target_classes:
            per_class_candidates.extend(self._detect_class_runs(timeline, cls))
        per_class_candidates.sort(key=lambda c: c.spike_time)
        return self._merge_overlapping(per_class_candidates)

    def _detect_class_runs(self, timeline: pd.DataFrame, cls: str) -> List[ClipCandidate]:
        above = timeline[cls].to_numpy() >= self.cfg.confidence_threshold
        if not above.any():
            return []

        candidates = []
        run_start = None
        for i, is_above in enumerate(above):
            if is_above and run_start is None:
                run_start = i
            elif not is_above and run_start is not None:
                candidates.append(self._candidate_from_run(timeline, cls, run_start, i))
                run_start = None
        if run_start is not None:
            candidates.append(self._candidate_from_run(timeline, cls, run_start, len(above)))
        return [c for c in candidates if c is not None]

    def _candidate_from_run(self, timeline: pd.DataFrame, cls: str, start_idx: int, end_idx: int):
        run = timeline.iloc[start_idx:end_idx]
        duration = float(run["bin_end"].iloc[-1] - run["bin_start"].iloc[0])
        if duration < self.cfg.min_event_duration_s:
            return None  # single-frame blip, not sustained enough to count as a real event

        peak_row = run.loc[run[cls].idxmax()]
        spike_time = float(peak_row["bin_start"])
        peak_confidence = float(peak_row[cls])
        t_min, t_max = float(timeline["bin_start"].iloc[0]), float(timeline["bin_end"].iloc[-1])
        return ClipCandidate(
            window_start=max(0.0, spike_time - self.cfg.pre_spike_seconds),
            window_end=min(t_max, spike_time + self.cfg.post_spike_seconds),
            spike_time=spike_time,
            # Not a statistical Z-score (see this module's docstring) - both fields carry the
            # peak confidence directly so this candidate is still usable by any code that
            # generically sorts/ranks ClipCandidates by "how strong was the signal".
            peak_hype_score=peak_confidence,
            peak_z_score=peak_confidence,
            source="sound_event",
            sound_events={cls: peak_confidence},
        )

    def _merge_overlapping(self, candidates: List[ClipCandidate]) -> List[ClipCandidate]:
        """
        Same spike-time-proximity + chaining approach as the other two detectors'
        merge steps, but combines sound_events dicts (union, keeping the higher
        confidence per class) instead of just picking one "stronger" candidate's
        fields - two different classes (e.g. laughter and cheering) peaking near
        the same moment should show up together, not silently drop one.
        """
        if not candidates:
            return candidates
        merged: List[ClipCandidate] = [candidates[0]]
        for cand in candidates[1:]:
            last = merged[-1]
            if not _same_moment(cand, last, self.cfg.min_seconds_between_spikes):
                merged.append(cand)
                continue
            combined_events: Dict[str, float] = dict(last.sound_events)
            for cls, conf in cand.sound_events.items():
                combined_events[cls] = max(combined_events.get(cls, 0.0), conf)
            stronger = cand if cand.peak_z_score > last.peak_z_score else last
            merged[-1] = replace(
                last,
                window_start=min(last.window_start, cand.window_start),
                window_end=max(last.window_end, cand.window_end),
                spike_time=stronger.spike_time,
                peak_hype_score=stronger.peak_hype_score,
                peak_z_score=stronger.peak_z_score,
                sound_events=combined_events,
            )
        return merged


def merge_sound_events(
    candidates: List[ClipCandidate],
    sound_event_candidates: List[ClipCandidate],
    allow_new_candidates: bool,
    overlap_tolerance_s: float = 30.0,
) -> List[ClipCandidate]:
    """
    Composes on top of core.audio_analyzer.merge_with_chat_candidates rather than
    extending it into a 3-way merge - `candidates` here is normally that
    function's own output (chat, possibly already enriched with "audio"). A
    nearby sound event adds the "sound_event" tag and its class/confidence data
    to an existing candidate; with no nearby candidate it becomes its own
    standalone entry only if allow_new_candidates is True, same policy as the
    chat/audio merge.
    """
    enriched: List[ClipCandidate] = []
    for cc in candidates:
        nearby = [ec for ec in sound_event_candidates if _same_moment(ec, cc, overlap_tolerance_s)]
        if nearby:
            combined_events: Dict[str, float] = dict(cc.sound_events)
            for ec in nearby:
                for cls, conf in ec.sound_events.items():
                    combined_events[cls] = max(combined_events.get(cls, 0.0), conf)
            cc = replace(cc, source=_add_source_tag(cc.source, "sound_event"), sound_events=combined_events)
        enriched.append(cc)

    result = enriched
    if allow_new_candidates:
        unmatched = [
            ec for ec in sound_event_candidates if not any(_same_moment(ec, cc, overlap_tolerance_s) for cc in candidates)
        ]
        result = result + unmatched

    return sorted(result, key=lambda c: c.window_start)
