"""
core/chat_analyzer.py

Chat analytics engine.

Bins raw Twitch chat messages into fixed-width time windows, computes a
weighted "hype score" per bin, and isolates statistically significant
engagement spikes (rolling mean + Z-score thresholding) into candidate
clip windows for the LLM agent (core/llm_agent.py) to refine.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import HypeScoreConfig, settings
from core.fetchers import ChatMessage

logger = logging.getLogger(__name__)

_SCORED_COLUMNS = [
    "bin_index", "bin_start", "bin_end",
    "message_count", "emote_count", "caps_exclaim_count", "unique_chatters",
    "hype_score", "rolling_mean", "rolling_std", "z_score",
]

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё']+")  # Cyrillic range included: caps detection was Latin-only before
_EXCLAIM_RE = re.compile(r"[!?]")

# Russian chat laughter is written as repeated "ах"/"ха" syllables ("ахах", "хахаха",
# "ахахаха", ...) rather than a fixed word, so it can't live in HypeScoreConfig.hype_emotes
# as a plain token — a token made up of nothing but а/х (or the Latin look-alikes a/x, since
# some type it with a mismatched keyboard layout) of length >=4 is essentially always this.
_RU_LAUGHTER_RE = re.compile(r"^[ахaxАХAX]{4,}$")


@dataclass(frozen=True)
class ChatBin:
    """Aggregated chat activity for a single fixed-width time window."""

    bin_start: float
    bin_end: float
    message_count: int
    emote_count: int
    caps_exclaim_count: int
    unique_chatters: int
    hype_score: float
    rolling_mean: float
    z_score: float


@dataclass(frozen=True)
class ClipCandidate:
    """
    A candidate clip window anchored on a detected spike. `peak_hype_score` is a
    generic "peak value of whatever signal flagged this candidate" - chat hype
    score for chat-sourced candidates, peak RMS for audio-sourced ones (see
    core/audio_analyzer.py) - kept as one field rather than a parallel type so
    both detectors' outputs can be merged/sorted/judged identically downstream.
    """

    window_start: float
    window_end: float
    spike_time: float
    peak_hype_score: float
    peak_z_score: float
    # "+"-joined combination of one or more of "chat" | "audio" | "sound_event", always in
    # that fixed order (see core/audio_analyzer._add_source_tag) - e.g. "chat", "audio",
    # "chat+audio", "chat+audio+sound_event". Left as a plain string, not an enum, since
    # this tag set has already grown once (audio, then sound_event) and is expected to
    # keep growing as more detectors are added.
    source: str = "chat"
    # Set when an overlapping audio-RMS spike also confirmed this candidate - extra context
    # surfaced to the LLM prompt (see llm_agent._build_user_prompt). None if audio analysis
    # is disabled or found no overlap here.
    audio_peak_z_score: Optional[float] = None
    # The enriching audio spike's OWN spike_time (absolute VOD seconds), kept separately
    # from audio_peak_z_score so a UI can point at exactly when it happened, not just how
    # strong it was - this candidate's own spike_time may be a different moment (e.g. a
    # chat spike a few seconds later than the audio that caused it).
    audio_peak_time: Optional[float] = None
    # Sound-event class name -> peak confidence (e.g. {"Laughter": 0.88}) for any YAMNet
    # classes detected at/near this candidate, regardless of whether this candidate
    # originated from chat, audio-RMS, or the sound-event detector itself. Empty when sound
    # event detection is disabled or found nothing here.
    sound_events: Dict[str, float] = field(default_factory=dict)
    # The strongest contributing sound event's OWN spike_time (absolute VOD seconds) - one
    # representative timestamp for the whole sound_events dict, not one per class, since a
    # UI showing "detected: X, Y" only needs a single point to direct attention to.
    sound_event_time: Optional[float] = None

    @property
    def duration(self) -> float:
        return self.window_end - self.window_start


class ChatAnalyzer:
    """Turns a raw chat message stream into a scored timeline and ranked clip candidates."""

    def __init__(self, config: HypeScoreConfig = None):
        self.cfg = config or settings.hype

    # --- public API ----------------------------------------------------------

    def compute_hype_timeline(self, messages: List[ChatMessage]) -> pd.DataFrame:
        """Returns the full per-bin scored timeline (for graphing in the UI)."""
        if not messages:
            logger.warning("ChatAnalyzer received no messages; returning empty timeline.")
            return pd.DataFrame(columns=_SCORED_COLUMNS)
        df = self._to_dataframe(messages)
        if df.empty:
            return pd.DataFrame(columns=_SCORED_COLUMNS)
        binned = self._bin_messages(df)
        return self._score_bins(binned)

    def analyze(self, messages: List[ChatMessage]) -> List[ClipCandidate]:
        """Returns ranked clip candidates. Convenience wrapper around analyze_with_timeline."""
        candidates, _ = self.analyze_with_timeline(messages)
        return candidates

    def analyze_with_timeline(
        self, messages: List[ChatMessage]
    ) -> Tuple[List[ClipCandidate], pd.DataFrame]:
        """Returns (candidates, scored_timeline) computed in a single pass."""
        scored = self.compute_hype_timeline(messages)
        if scored.empty:
            return [], scored
        return self._detect_spikes(scored), scored

    @staticmethod
    def to_chat_bins(scored: pd.DataFrame) -> List[ChatBin]:
        """Converts a scored timeline DataFrame into typed ChatBin records."""
        return [
            ChatBin(
                bin_start=float(row.bin_start),
                bin_end=float(row.bin_end),
                message_count=int(row.message_count),
                emote_count=int(row.emote_count),
                caps_exclaim_count=int(row.caps_exclaim_count),
                unique_chatters=int(row.unique_chatters),
                hype_score=float(row.hype_score),
                rolling_mean=float(row.rolling_mean) if not np.isnan(row.rolling_mean) else 0.0,
                z_score=float(row.z_score),
            )
            for row in scored.itertuples(index=False)
        ]

    # --- dataframe construction ------------------------------------------------

    def _to_dataframe(self, messages: List[ChatMessage]) -> pd.DataFrame:
        records = [
            {
                "timestamp": msg.timestamp,
                "username": msg.username,
                "emote_count": self._count_hype_emotes(msg.text),
                "is_caps_exclaim": self._is_caps_or_exclaim(msg.text),
            }
            for msg in messages
            if msg.timestamp >= 0
        ]
        df = pd.DataFrame.from_records(records)
        if not df.empty:
            df.sort_values("timestamp", inplace=True)
        return df

    def _count_hype_emotes(self, text: str) -> int:
        if not text:
            return 0
        count = 0
        for tok in text.lower().split():
            stripped = tok.strip(".,!?")
            if stripped in self.cfg.hype_emotes or _RU_LAUGHTER_RE.match(stripped):
                count += 1
        return count

    def _is_caps_or_exclaim(self, text: str) -> bool:
        if not text:
            return False
        if _EXCLAIM_RE.search(text):
            return True
        long_words = [w for w in _WORD_RE.findall(text) if len(w) >= self.cfg.caps_min_word_len]
        if not long_words:
            return False
        caps_ratio = sum(1 for w in long_words if w.isupper()) / len(long_words)
        return caps_ratio >= self.cfg.caps_min_ratio

    # --- binning -----------------------------------------------------------

    def _bin_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        bin_s = self.cfg.bin_seconds
        df = df.copy()
        df["bin_index"] = (df["timestamp"] // bin_s).astype(int)

        last_bin = int(df["bin_index"].max())
        full_index = pd.RangeIndex(0, last_bin + 1, name="bin_index")

        grouped = df.groupby("bin_index").agg(
            message_count=("username", "count"),
            emote_count=("emote_count", "sum"),
            caps_exclaim_count=("is_caps_exclaim", "sum"),
            unique_chatters=("username", "nunique"),
        )
        grouped = grouped.reindex(full_index, fill_value=0)
        grouped["bin_start"] = grouped.index.to_series() * bin_s
        grouped["bin_end"] = grouped["bin_start"] + bin_s
        return grouped.reset_index()

    # --- scoring -------------------------------------------------------------

    def _score_bins(self, binned: pd.DataFrame) -> pd.DataFrame:
        df = binned.copy()
        df["hype_score"] = (
            self.cfg.weight_message_volume * df["message_count"]
            + self.cfg.weight_emote_frequency * df["emote_count"]
            + self.cfg.weight_caps_exclaim * df["caps_exclaim_count"]
            + self.cfg.weight_unique_chatters * df["unique_chatters"]
        )

        window = self.cfg.rolling_window_bins
        rolling = df["hype_score"].rolling(window=window, min_periods=max(3, window // 4))
        df["rolling_mean"] = rolling.mean()
        rolling_std = rolling.std().replace(0, np.nan)

        df["z_score"] = ((df["hype_score"] - df["rolling_mean"]) / rolling_std).fillna(0.0)
        df["rolling_mean"] = df["rolling_mean"].fillna(df["hype_score"])
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
                # float(...) matters here, not just style: row[...] is a numpy.float64
                # (from iterating a pandas DataFrame), which duck-types as a plain float
                # almost everywhere but not always - it's silently accepted into these
                # dataclass fields (Python doesn't enforce dataclass type hints at
                # runtime) and can surface much later as a genuine bug, e.g. Gradio's
                # slider preprocessing rejecting it after a value round-trip through the
                # client because it no longer compares as a plain float.
                window_start=float(max(0.0, row["bin_start"] - self.cfg.pre_spike_seconds)),
                window_end=float(row["bin_end"] + self.cfg.post_spike_seconds),
                spike_time=float(row["bin_start"]),
                peak_hype_score=float(row["hype_score"]),
                peak_z_score=float(row["z_score"]),
            )
            for row in selected
        ]
        return self._merge_overlapping(candidates, self.cfg.max_merged_duration_seconds)

    @staticmethod
    def _merge_overlapping(candidates: List[ClipCandidate], max_duration: float) -> List[ClipCandidate]:
        """
        Adjacent/overlapping spike windows collapse into one clip, keeping the
        stronger peak - but never past max_duration. Without a cap, a wide
        pre/post_spike_seconds setting can chain-merge several genuinely
        distinct, unrelated moments (different topics minutes apart) into one
        sprawling multi-topic candidate that no LLM can judge as a single
        self-contained moment - it just anchors on the loudest one and silently
        drops whatever else is riding along in the same window.
        """
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
            merged[-1] = ClipCandidate(
                window_start=last.window_start,
                window_end=would_be_end,
                spike_time=stronger.spike_time,
                peak_hype_score=stronger.peak_hype_score,
                peak_z_score=stronger.peak_z_score,
            )
        return merged


# --------------------------------------------------------------------------- #
# Convenience module-level API
# --------------------------------------------------------------------------- #


def analyze_chat(messages: List[ChatMessage]) -> List[ClipCandidate]:
    """Convenience entry point returning only clip candidates."""
    return ChatAnalyzer().analyze(messages)
