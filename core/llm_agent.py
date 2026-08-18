"""
core/llm_agent.py

Context alignment & LLM agent.

Maps a chat-hype candidate window (core/chat_analyzer.ClipCandidate) onto the
subtitle transcript covering that window, then asks an LLM (via litellm, so
OpenAI / DeepSeek / local Ollama are interchangeable) to judge whether the
window is actually clip-worthy (a chat spike is a statistical signal, not a
guarantee something happened) and, if so, pick the precise hook/joke start
and reaction-resolution end, plus a title, viral confidence score, and short
summary. The LLM's raw text response is never trusted as-is: it is parsed as
JSON and validated through a Pydantic schema before use.

Rejected candidates are kept, not discarded - see CandidateClip.is_clip_worthy -
so a human operator can review them (app.py's "show rejected candidates" toggle)
rather than just trusting the LLM's veto blindly.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import litellm
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from config import LLMConfig, settings
from core.chat_analyzer import ClipCandidate
from core.fetchers import SubtitleSegment

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class LLMAgentError(Exception):
    """Base class for LLM-agent failures."""


class LLMResponseError(LLMAgentError):
    """Raised when the model's response can't be parsed/validated after all retries."""


# --------------------------------------------------------------------------- #
# Schema for LLM output (strictly validated)
# --------------------------------------------------------------------------- #


class ClipSuggestion(BaseModel):
    """Strict schema the LLM's JSON response must conform to."""

    is_clip_worthy: bool = Field(...)
    rejection_reason: Optional[str] = Field(None, max_length=200)
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., ge=0)
    title: str = Field(..., min_length=1, max_length=100)
    viral_score: int = Field(..., ge=1, le=10)
    summary: str = Field(..., min_length=1, max_length=280)

    @field_validator("title", "summary", "rejection_reason")
    @classmethod
    def _strip_whitespace(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _end_after_start(self) -> "ClipSuggestion":
        if self.end_time <= self.start_time:
            raise ValueError(f"end_time ({self.end_time}) must be greater than start_time ({self.start_time})")
        return self


# --------------------------------------------------------------------------- #
# Output data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CandidateClip:
    """
    A fully-refined clip: LLM-adjusted boundaries plus provenance metadata.

    Rejected candidates (is_clip_worthy=False) are NOT dropped - they're kept
    with their rejection_reason attached so a human operator can review the
    LLM's judgment calls instead of just trusting them blindly. Callers that
    only want usable clips (exports, injection) should filter on is_clip_worthy.
    """

    start_time: float
    end_time: float
    title: str
    viral_score: int
    summary: str
    spike_time: float
    peak_hype_score: float
    peak_z_score: float
    transcript_excerpt: str
    used_fallback: bool = False
    is_clip_worthy: bool = True
    rejection_reason: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


# --------------------------------------------------------------------------- #
# Prompting helpers
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """You are a viral video clip editor for Twitch/YouTube content. You are given a \
transcript excerpt covering a moment where CHAT ACTIVITY SPIKED STATISTICALLY, along with the \
approximate time window chat reacted to.

STEP 1 - Judge whether this moment is actually worth clipping. A chat spike is only a statistical signal; \
it does not guarantee anything happened. Read the transcript and decide: is there something identifiable \
here - a joke, a mistake, a reveal, a strong reaction, a notable statement - that plausibly explains the \
spike? Chat spikes can be false positives (raids, copypasta spam, an unrelated meme, chat reacting to \
something off-screen) with nothing worth watching in the transcript itself.
  - If nothing identifiable is happening, set "is_clip_worthy" to false and briefly explain why in \
"rejection_reason".
  - If the transcript for this window is sparse, garbled, or uninformative, do NOT reject just because of \
that alone - a thin transcript doesn't mean nothing happened, and the chat spike is still a real signal. \
When you are unsure, default to "is_clip_worthy": true rather than rejecting.

STEP 2 - If it IS worth clipping, find the precise clip boundaries that make this a satisfying, \
self-contained short clip: start at the beginning of the setup/hook/joke (not mid-sentence), and end \
right after the punchline or reaction resolves (not too early, not lingering).

Write "title" and "summary" in the SAME language as the transcript (e.g. a Russian transcript gets a \
Russian title and summary) - never translate them to English unless the transcript itself is in English. \
All JSON field names and the overall structure must stay exactly as specified below regardless of language.

Respond with ONLY a single JSON object, no markdown fences, no commentary, matching exactly this shape:
{
  "is_clip_worthy": <true or false>,
  "rejection_reason": "<short reason if is_clip_worthy is false, else null>",
  "start_time": <float, seconds, absolute video timeline>,
  "end_time": <float, seconds, absolute video timeline>,
  "title": "<punchy clip title in the transcript's language, <=100 chars>",
  "viral_score": <integer 1-10, confidence this clip goes viral>,
  "summary": "<one or two sentence summary in the transcript's language, <=280 chars>"
}
Even when is_clip_worthy is false, still fill in your best-effort start_time/end_time/title/summary \
(with a low viral_score) instead of omitting them - the caller may still want to review it manually.
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def select_subtitle_window(
    subtitles: List[SubtitleSegment],
    window_start: float,
    window_end: float,
    pad_seconds: float = 10.0,
) -> List[SubtitleSegment]:
    """Subtitle segments overlapping [window_start - pad, window_end + pad]."""
    lo, hi = window_start - pad_seconds, window_end + pad_seconds
    return [seg for seg in subtitles if seg.end >= lo and seg.start <= hi]


def format_transcript(segments: List[SubtitleSegment]) -> str:
    """Renders subtitle segments as '[start-end] text' lines, one per cue."""
    return "\n".join(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text}" for seg in segments) or "(no transcript available)"


def _build_user_prompt(candidate: ClipCandidate, transcript: str) -> str:
    return (
        f"Chat hype spike detected at t={candidate.spike_time:.1f}s "
        f"(peak hype score={candidate.peak_hype_score:.1f}, z-score={candidate.peak_z_score:.2f}).\n"
        f"Suggested rough window: {candidate.window_start:.1f}s to {candidate.window_end:.1f}s.\n\n"
        f"Transcript covering this window:\n{transcript}\n\n"
        "Pick start_time/end_time as absolute seconds on this same timeline, ideally within or close to "
        "the suggested window. Respond with the JSON object only."
    )


def _extract_json(content: str) -> dict:
    cleaned = _MARKDOWN_FENCE_RE.sub("", content.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(cleaned)
    if not match:
        raise LLMResponseError(f"No JSON object found in LLM response: {content[:200]!r}")
    return json.loads(match.group(0))


# --------------------------------------------------------------------------- #
# LLM agent
# --------------------------------------------------------------------------- #


class LLMAgent:
    """Refines chat-spike candidates into precise, titled clip suggestions."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.cfg = config or settings.llm
        self.model = self.cfg.resolve_model()
        self.api_base = self.cfg.resolve_api_base()

    def refine_candidate(
        self,
        candidate: ClipCandidate,
        subtitles: List[SubtitleSegment],
    ) -> CandidateClip:
        """
        Refines a single candidate window. Always returns a CandidateClip -
        rejected candidates are never dropped, just marked (is_clip_worthy=False,
        rejection_reason set) so a human can review the LLM's judgment instead of
        silently losing the candidate. On repeated LLM/validation FAILURE (as
        opposed to an explicit rejection) falls back to the raw chat-spike window,
        since a failure tells us nothing about whether the clip is actually good.
        """
        window_segments = select_subtitle_window(subtitles, candidate.window_start, candidate.window_end)
        transcript = format_transcript(window_segments)

        suggestion = self._call_llm_with_retries(candidate, transcript)
        if suggestion is None:
            return self._fallback_clip(candidate, transcript)

        if not suggestion.is_clip_worthy:
            logger.info(
                "LLM flagged candidate at t=%.1f as not clip-worthy: %s",
                candidate.spike_time, suggestion.rejection_reason or "(no reason given)",
            )

        start_time, end_time = self._clamp_bounds(candidate, suggestion.start_time, suggestion.end_time)
        return CandidateClip(
            start_time=start_time,
            end_time=end_time,
            title=suggestion.title,
            viral_score=suggestion.viral_score,
            summary=suggestion.summary,
            spike_time=candidate.spike_time,
            peak_hype_score=candidate.peak_hype_score,
            peak_z_score=candidate.peak_z_score,
            transcript_excerpt=transcript,
            used_fallback=False,
            is_clip_worthy=suggestion.is_clip_worthy,
            rejection_reason=None if suggestion.is_clip_worthy else suggestion.rejection_reason,
        )

    def refine_candidates(
        self,
        candidates: List[ClipCandidate],
        subtitles: List[SubtitleSegment],
    ) -> List[CandidateClip]:
        """
        Refines a batch of candidates; a single failure never aborts the batch.
        Returns ALL of them (accepted and rejected alike) - filter on
        `.is_clip_worthy` at the call site for anything that should only use
        accepted clips (exports, injection).
        """
        results: List[CandidateClip] = []
        for candidate in candidates:
            try:
                results.append(self.refine_candidate(candidate, subtitles))
            except Exception:
                logger.exception("Unexpected error refining candidate at t=%.1f; skipping.", candidate.spike_time)
        return results

    # --- internals -----------------------------------------------------------

    def _call_llm_with_retries(self, candidate: ClipCandidate, transcript: str) -> Optional[ClipSuggestion]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                content = self._call_llm_once(candidate, transcript)
                payload = _extract_json(content)
                return ClipSuggestion.model_validate(payload)
            except (LLMResponseError, ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "LLM response invalid on attempt %d/%d for spike at t=%.1f: %s",
                    attempt, self.cfg.max_retries, candidate.spike_time, exc,
                )
            except Exception as exc:  # network/provider errors from litellm
                last_error = exc
                logger.warning(
                    "LLM call failed on attempt %d/%d for spike at t=%.1f: %s",
                    attempt, self.cfg.max_retries, candidate.spike_time, exc,
                )
        logger.error(
            "LLM agent exhausted %d retries for spike at t=%.1f; falling back to raw window. Last error: %s",
            self.cfg.max_retries, candidate.spike_time, last_error,
        )
        return None

    def _call_llm_once(self, candidate: ClipCandidate, transcript: str) -> str:
        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(candidate, transcript)},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            timeout=self.cfg.request_timeout_s,
            api_base=self.api_base,
            api_key=self.cfg.api_key,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise LLMResponseError("LLM returned an empty response.")
        return content

    def _clamp_bounds(self, candidate: ClipCandidate, start_time: float, end_time: float) -> tuple[float, float]:
        """Keeps LLM-chosen bounds sane: non-negative, ordered, and duration-bounded."""
        lo = max(0.0, candidate.window_start - 15.0)
        hi = candidate.window_end + 15.0

        start_time = min(max(start_time, lo), hi)
        end_time = min(max(end_time, lo), hi)
        if end_time <= start_time:
            end_time = start_time + self.cfg.min_clip_duration_s

        duration = end_time - start_time
        if duration < self.cfg.min_clip_duration_s:
            end_time = start_time + self.cfg.min_clip_duration_s
        elif duration > self.cfg.max_clip_duration_s:
            end_time = start_time + self.cfg.max_clip_duration_s

        return start_time, end_time

    @staticmethod
    def _fallback_clip(candidate: ClipCandidate, transcript: str) -> CandidateClip:
        """Used when the LLM never produces a valid response: keep the raw chat-spike window."""
        return CandidateClip(
            start_time=candidate.window_start,
            end_time=candidate.window_end,
            title=f"Chat hype spike at {candidate.spike_time:.0f}s",
            viral_score=min(10, max(1, round(candidate.peak_z_score))),
            summary="Auto-generated fallback clip (LLM refinement unavailable).",
            spike_time=candidate.spike_time,
            peak_hype_score=candidate.peak_hype_score,
            peak_z_score=candidate.peak_z_score,
            transcript_excerpt=transcript,
            used_fallback=True,
        )


# --------------------------------------------------------------------------- #
# Convenience module-level API
# --------------------------------------------------------------------------- #


def refine_candidates(
    candidates: List[ClipCandidate],
    subtitles: List[SubtitleSegment],
) -> List[CandidateClip]:
    """Convenience entry point for app.py. Returns ALL clips; filter on .is_clip_worthy as needed."""
    return LLMAgent().refine_candidates(candidates, subtitles)
