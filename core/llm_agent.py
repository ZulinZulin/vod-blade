"""
core/llm_agent.py

Context alignment & LLM agent.

Maps a chat-hype candidate window (core/chat_analyzer.ClipCandidate) onto the
subtitle transcript covering that window, then asks a local Ollama model (via
litellm) to judge whether the window is actually clip-worthy (a chat spike is
a statistical signal, not a guarantee something happened) and, if so, pick
the precise hook/joke start and reaction-resolution end, plus a title, viral
confidence score, and short summary. The LLM's raw text response is never
trusted as-is: it is parsed as JSON and validated through a Pydantic schema
before use.

Rejected candidates are kept, not discarded - see CandidateClip.is_clip_worthy -
so a human operator can review them (app.py's "show rejected candidates" toggle)
rather than just trusting the LLM's veto blindly.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import litellm
import requests
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


class OllamaGpuOffloadError(LLMAgentError):
    """
    Raised when the configured Ollama model isn't sufficiently VRAM-resident -
    i.e. it's (partially) offloaded to CPU/RAM, which "works" but can run an
    order of magnitude slower with no visible indication why. Deliberately NOT
    caught by the retry/fallback machinery: this should stop the run loudly
    rather than silently degrade into a very slow one.
    """


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
    # Carried through from ClipCandidate unchanged - see that class's docstring
    # in core/chat_analyzer.py for why this stays a plain string.
    source: str = "chat"
    audio_peak_z_score: Optional[float] = None
    audio_peak_time: Optional[float] = None
    sound_events: Dict[str, float] = field(default_factory=dict)
    sound_event_time: Optional[float] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


# --------------------------------------------------------------------------- #
# Prompting helpers
# --------------------------------------------------------------------------- #

DEFAULT_SYSTEM_PROMPT = """You are a viral video clip editor for Twitch/YouTube content. You are given a \
transcript excerpt covering a window flagged by one or more automated detectors - a statistical chat \
activity spike, a loud audio moment, and/or a detected acoustic event (e.g. laughter, screaming, \
cheering) - the specific signal(s) for this window are described below the transcript.

IMPORTANT: the reported timestamp marks when the detector's signal peaked, NOT necessarily when the \
actual noteworthy moment happened. For a chat spike especially, chat takes time to read, react, and \
type - the real hook (the joke, the mistake, the reveal) is very often tens of seconds BEFORE the spike \
timestamp, not at or after it. Treat the timestamp as a rough pointer into the window, not the location \
of the moment itself. Read the ENTIRE transcript window below and judge it as a whole - do not anchor \
your search on the content immediately surrounding the timestamp.

STEP 1 - Judge whether the window as a whole contains a moment actually worth clipping. A chat or audio \
spike is only a statistical signal; it does not guarantee anything worth watching happened - a detected \
acoustic event (laughter, screaming, cheering) is stronger evidence, since it's a direct classification \
of the reaction itself rather than a correlated proxy, but still confirm it against the transcript rather \
than accepting it blindly. Apply a HIGH bar: ordinary conversation, routine \
explanation, or coherent-but-unremarkable discussion is NOT enough on its own, even if articulate or \
substantive - most of any stream or podcast is exactly that, and none of it is clip-worthy by default. \
Only call something clip-worthy if it would make a stranger with zero context stop scrolling: a joke that \
actually lands, a mistake, a surprising reveal, a strong emotional reaction, a sharp disagreement or \
conflict, or a genuinely standout line - something with a clear, self-contained hook.
  - Examples that do NOT qualify on their own: a well-reasoned opinion, a normal Q&A answer, routine \
banter, chat spamming an emote as a social/community ritual unrelated to specific on-screen content, or \
a raid/donation/subscriber alert with no accompanying notable moment. Long-form talk (podcasts, co-op \
commentary) is especially prone to sounding "notable" while being ordinary - hold it to the same bar.
  - If nothing like that is happening, set "is_clip_worthy" to false and briefly explain why in \
"rejection_reason".
  - A sparse/garbled/uninformative transcript is not, by itself, grounds for rejection - the chat spike is \
still a real signal even when the transcript is thin. But it is also not an excuse to accept: if you \
can't point to an actual hook, reject it rather than assuming one exists off-screen.

STEP 2 - If it IS worth clipping, find the precise clip boundaries that make this a satisfying, \
self-contained short clip: start at the beginning of the setup/hook/joke (not mid-sentence), and end \
right after the punchline or reaction resolves (not too early, not lingering). Snap start_time/end_time \
to the actual [start-end] boundaries of the subtitle lines shown in the transcript below rather than \
inventing an arbitrary in-between value - this avoids cutting mid-word or mid-sentence.

STEP 3 - "title" and "summary" MUST describe ONLY what actually happens between your own chosen \
start_time and end_time - never something that only appears in the surrounding transcript outside that \
range. If the notable moment you identified falls outside your current start_time/end_time, MOVE the \
boundaries to include it instead of describing something your own chosen range excludes.

STEP 4 - Score "viral_score" honestly across the FULL 1-10 range. Without deliberate effort, rating \
scales like this tend to collapse toward a "safe" 7-8 regardless of actual quality - resist that. Use \
these anchors:
  - 1-2: weak. Barely worth a look; you almost rejected it.
  - 3-4: mildly amusing or notable, but forgettable - unlikely to grab a stranger's attention.
  - 5-6: a solid, genuine reaction. Works for viewers who already like this channel, but isn't the kind \
of thing that gets shared outside that audience.
  - 7-8: strong. A stranger with zero context would likely stop scrolling for it, and it could plausibly \
get shared/reposted.
  - 9-10: exceptional. Genuine highlight-reel material - reserve this for moments you'd bet on.
If most of what you see across a stream is honestly mid-tier, most of your scores should land in the 4-6 \
range - do not inflate scores to sound more decisive than the moment actually warrants. A clip can be \
is_clip_worthy=true with a viral_score as low as 3 or 4 if it clears STEP 1's bar but isn't exceptional; \
"worth keeping" and "amazing" are different questions.

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
  "viral_score": <integer 1-10 per the STEP 4 anchors above - use the full range, not just 7-8>,
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


def build_stat_only_clip(
    candidate: ClipCandidate, transcript: str, summary: str, used_fallback: bool,
) -> CandidateClip:
    """
    Builds a CandidateClip straight from the raw chat-spike window, with no LLM
    involvement at all - shared by the retry-exhaustion fallback path below and
    by the UI's "skip LLM judging" option (for quickly reviewing/tuning the
    statistical spike detection itself without waiting on real LLM calls).
    is_clip_worthy defaults to True either way, since there's no judgment to
    reject them - the operator uses the existing manual accept/reject tools to
    curate a raw candidate list by hand instead.
    """
    return CandidateClip(
        start_time=candidate.window_start,
        end_time=candidate.window_end,
        title=f"{_title_label_for_source(candidate.source)} at {candidate.spike_time:.0f}s",
        viral_score=min(10, max(1, round(candidate.peak_z_score))),
        summary=summary,
        spike_time=candidate.spike_time,
        peak_hype_score=candidate.peak_hype_score,
        peak_z_score=candidate.peak_z_score,
        transcript_excerpt=transcript,
        used_fallback=used_fallback,
        source=candidate.source,
        audio_peak_z_score=candidate.audio_peak_z_score,
        audio_peak_time=candidate.audio_peak_time,
        sound_events=candidate.sound_events,
        sound_event_time=candidate.sound_event_time,
    )


def _title_label_for_source(source: str) -> str:
    """source may be a "+"-joined combination (see ClipCandidate.source) - pick the most
    informative label rather than requiring an exact match for every combination."""
    tags = source.split("+")
    if "chat" in tags and "audio" in tags:
        return "Chat+audio spike"
    if "chat" in tags:
        return "Chat hype spike"
    if "audio" in tags:
        return "Audio peak"
    if "sound_event" in tags:
        return "Sound event"
    return "Spike"


def _build_user_prompt(candidate: ClipCandidate, transcript: str, content_hint: str = "") -> str:
    hint_line = f"\nContext from the operator about this stream: {content_hint}\n" if content_hint and content_hint.strip() else ""
    tags = candidate.source.split("+")
    if "chat" in tags:
        signal_line = (
            f"Chat's reaction peaked at t={candidate.spike_time:.1f}s within this window "
            f"(peak hype score={candidate.peak_hype_score:.1f}, z-score={candidate.peak_z_score:.2f}) - "
            f"remember this is where chat's reaction peaked, not necessarily where the actual moment is; "
            f"scan the whole window above rather than just the content near this timestamp.\n"
        )
    elif "audio" in tags:
        signal_line = (
            f"This window was flagged by a LOUD AUDIO MOMENT at t={candidate.spike_time:.1f}s "
            f"(peak audio energy z-score={candidate.peak_z_score:.2f}), not by chat activity - chat may have "
            f"been quiet here even if something notable happened (e.g. viewers too engaged to type).\n"
        )
    else:
        signal_line = (
            f"This window was flagged by a detected acoustic event at t={candidate.spike_time:.1f}s (see "
            f"below), not by chat activity or raw volume - chat may have been quiet and the audio not "
            f"especially loud, even if something notable happened.\n"
        )
    audio_context_line = (
        f"The audio track also peaked around this same window (energy z-score="
        f"{candidate.audio_peak_z_score:.2f}), reinforcing that something notable likely happened here.\n"
        if candidate.audio_peak_z_score is not None else ""
    )
    sound_event_line = ""
    if candidate.sound_events:
        events_desc = ", ".join(
            f"{cls} (confidence {conf:.2f})"
            for cls, conf in sorted(candidate.sound_events.items(), key=lambda kv: kv[1], reverse=True)
        )
        # A detected acoustic event is more direct evidence than a statistical proxy: a real
        # laugh or scream IS the reaction, not just a signal correlated with one.
        sound_event_line = (
            f"A distinct acoustic event was detected in this window: {events_desc}. Unlike chat/audio "
            f"volume, this is direct evidence a real reaction happened here, not just a proxy for one.\n"
        )
    return (
        f"Transcript window to judge: {candidate.window_start:.1f}s to {candidate.window_end:.1f}s.\n"
        f"{signal_line}"
        f"{audio_context_line}"
        f"{sound_event_line}"
        f"{hint_line}\n"
        f"Transcript covering this window:\n{transcript}\n\n"
        "Pick start_time/end_time as absolute seconds on this same timeline, snapped to real "
        "subtitle-line boundaries shown above, wherever in the window the actual best moment is. "
        "Respond with the JSON object only."
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

    def __init__(self, config: Optional[LLMConfig] = None, system_prompt: Optional[str] = None):
        self.cfg = config or settings.llm
        self.model = self.cfg.resolve_model()
        self.api_base = self.cfg.resolve_api_base()
        # A blank/whitespace-only override falls back to the default rather than sending
        # an empty system prompt to the model, which would break judgment entirely.
        self.system_prompt = system_prompt.strip() if system_prompt and system_prompt.strip() else DEFAULT_SYSTEM_PROMPT
        self._logged_system_prompt = False
        self._checked_gpu_residency = False

    def refine_candidate(
        self,
        candidate: ClipCandidate,
        subtitles: List[SubtitleSegment],
        content_hint: str = "",
    ) -> CandidateClip:
        """
        Refines a single candidate window. Always returns a CandidateClip -
        rejected candidates are never dropped, just marked (is_clip_worthy=False,
        rejection_reason set) so a human can review the LLM's judgment instead of
        silently losing the candidate. On repeated LLM/validation FAILURE (as
        opposed to an explicit rejection) falls back to the raw chat-spike window,
        since a failure tells us nothing about whether the clip is actually good.

        `content_hint` is an optional free-text note from the operator about the
        stream's genre/format (e.g. "podcast, lots of talking - be strict about
        what counts as notable"), folded into the prompt as extra context.
        """
        self._ensure_ollama_ready()
        window_segments = select_subtitle_window(subtitles, candidate.window_start, candidate.window_end)
        transcript = format_transcript(window_segments)

        suggestion = self._call_llm_with_retries(candidate, transcript, content_hint)
        if suggestion is None:
            return build_stat_only_clip(
                candidate, transcript, "Auto-generated fallback clip (LLM refinement unavailable).",
                used_fallback=True,
            )

        is_clip_worthy = suggestion.is_clip_worthy
        rejection_reason = suggestion.rejection_reason if not is_clip_worthy else None
        if not is_clip_worthy:
            logger.info(
                "LLM flagged candidate at t=%.1f as not clip-worthy: %s",
                candidate.spike_time, rejection_reason or "(no reason given)",
            )
        elif suggestion.viral_score < self.cfg.min_viral_score:
            # The model called it worthy, but its own confidence score doesn't clear the
            # operator's bar - treat as rejected rather than silently keeping a weak clip.
            is_clip_worthy = False
            rejection_reason = (
                f"LLM judged this clip-worthy but scored it {suggestion.viral_score}/10, below the "
                f"configured minimum of {self.cfg.min_viral_score}."
            )
            logger.info("Candidate at t=%.1f downgraded by min_viral_score: %s", candidate.spike_time, rejection_reason)

        start_time, end_time = self._clamp_bounds(candidate, suggestion.start_time, suggestion.end_time)

        # Re-extract the transcript for the FINAL chosen range (not the padded candidate
        # window used for judgment), so a reviewing editor can see at a glance whether the
        # title/summary actually matches what's inside the clip's own boundaries.
        final_segments = select_subtitle_window(subtitles, start_time, end_time, pad_seconds=2.0)
        final_transcript = format_transcript(final_segments) if final_segments else transcript

        return CandidateClip(
            start_time=start_time,
            end_time=end_time,
            title=suggestion.title,
            viral_score=suggestion.viral_score,
            summary=suggestion.summary,
            spike_time=candidate.spike_time,
            peak_hype_score=candidate.peak_hype_score,
            peak_z_score=candidate.peak_z_score,
            transcript_excerpt=final_transcript,
            used_fallback=False,
            is_clip_worthy=is_clip_worthy,
            rejection_reason=rejection_reason,
            source=candidate.source,
            audio_peak_z_score=candidate.audio_peak_z_score,
            audio_peak_time=candidate.audio_peak_time,
            sound_events=candidate.sound_events,
            sound_event_time=candidate.sound_event_time,
        )

    def refine_candidates(
        self,
        candidates: List[ClipCandidate],
        subtitles: List[SubtitleSegment],
        content_hint: str = "",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[CandidateClip]:
        """
        Refines a batch of candidates; a single failure never aborts the batch.
        Returns ALL of them (accepted and rejected alike) - filter on
        `.is_clip_worthy` at the call site for anything that should only use
        accepted clips (exports, injection).

        The Ollama GPU-residency check runs once here, before the loop and
        outside its per-candidate try/except, so a confirmed CPU offload raises
        OllamaGpuOffloadError straight out of this call instead of being caught
        and silently skipped candidate-by-candidate.

        `progress_callback`, if given, is called as (completed_count, total)
        after each candidate finishes (success or failure alike) - lets a
        caller (e.g. the Gradio UI) report real per-candidate progress during
        a long batch instead of one static "judging..." message for the
        whole run. Kept as a plain callable rather than importing Gradio here,
        since this module has no business knowing about the UI layer.
        """
        self._ensure_ollama_ready()
        results: List[CandidateClip] = []
        total = len(candidates)
        for i, candidate in enumerate(candidates, start=1):
            try:
                results.append(self.refine_candidate(candidate, subtitles, content_hint))
            except OllamaGpuOffloadError:
                raise
            except Exception:
                logger.exception("Unexpected error refining candidate at t=%.1f; skipping.", candidate.spike_time)
            if progress_callback is not None:
                progress_callback(i, total)
        return results

    # --- internals -----------------------------------------------------------

    def _call_llm_with_retries(
        self, candidate: ClipCandidate, transcript: str, content_hint: str = ""
    ) -> Optional[ClipSuggestion]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                content = self._call_llm_once(candidate, transcript, content_hint)
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

    def _call_llm_once(self, candidate: ClipCandidate, transcript: str, content_hint: str = "") -> str:
        self._log_system_prompt_once()
        user_prompt = _build_user_prompt(candidate, transcript, content_hint)
        logger.info(
            "LLM user prompt for candidate @ spike_time=%.1fs (window %.1f-%.1f), model=%s:\n%s\n%s\n%s",
            candidate.spike_time, candidate.window_start, candidate.window_end, self.model,
            "-" * 80, user_prompt, "-" * 80,
        )

        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            timeout=self.cfg.request_timeout_s,
            api_base=self.api_base,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise LLMResponseError("LLM returned an empty response.")
        logger.info("LLM raw response for candidate @ spike_time=%.1fs:\n%s", candidate.spike_time, content)
        return content

    def _log_system_prompt_once(self) -> None:
        """Logs the (candidate-invariant) system prompt a single time per agent instance."""
        if self._logged_system_prompt:
            return
        logger.info("LLM system prompt (shared across all candidates):\n%s\n%s\n%s", "=" * 80, self.system_prompt, "=" * 80)
        self._logged_system_prompt = True

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

    def _ensure_ollama_ready(self) -> None:
        """
        Refuses to proceed if the model isn't sufficiently VRAM-resident, per
        config.min_ollama_gpu_ratio (<=0 disables this check). Runs once per
        LLMAgent instance (guarded by _checked_gpu_residency). Failures in the check
        itself (Ollama unreachable, API shape changed, etc.) are logged and swallowed -
        only a CONFIRMED offload raises, since the diagnostic probe failing isn't the
        same as the model actually being offloaded, and the real litellm call will
        surface genuine connectivity problems on its own with its own error handling.
        """
        if self._checked_gpu_residency:
            return
        self._checked_gpu_residency = True

        if self.cfg.min_ollama_gpu_ratio <= 0:
            return

        bare_model = self.model.split("/", 1)[1] if self.model.startswith("ollama/") else self.model
        base = (self.api_base or "http://localhost:11434").rstrip("/")

        try:
            entry = self._find_loaded_ollama_model(base, bare_model)
            if entry is None:
                # Not loaded yet - force a load with a minimal generation so there's
                # something in /api/ps to actually check the residency of.
                logger.info(
                    "Ollama model '%s' not yet loaded; sending a warm-up request to check GPU residency.",
                    bare_model,
                )
                requests.post(
                    f"{base}/api/generate",
                    json={"model": bare_model, "prompt": " ", "stream": False, "options": {"num_predict": 1}},
                    timeout=max(120, self.cfg.request_timeout_s * 2),
                )
                entry = self._find_loaded_ollama_model(base, bare_model)
            if entry is None:
                logger.warning(
                    "Could not confirm GPU residency for Ollama model '%s' (not found in /api/ps even "
                    "after a warm-up request); proceeding without the check.", bare_model,
                )
                return
        except requests.exceptions.RequestException as exc:
            logger.warning("Ollama GPU-residency check failed (%s); proceeding without it.", exc)
            return

        size = entry.get("size") or 0
        size_vram = entry.get("size_vram") or 0
        if size <= 0:
            return
        ratio = size_vram / size
        if ratio < self.cfg.min_ollama_gpu_ratio:
            raise OllamaGpuOffloadError(
                f"Ollama model '{bare_model}' is only {ratio:.0%} GPU-resident (needs >= "
                f"{self.cfg.min_ollama_gpu_ratio:.0%}). It's being partially offloaded to CPU/RAM, most "
                "likely because something else on this machine is using VRAM right now - this would run "
                "much slower than usual, so the run is being stopped instead. Close other GPU-heavy "
                "applications and try again, or set LLM_MIN_OLLAMA_GPU_RATIO=0 in .env to allow this."
            )
        logger.info("Ollama model '%s' is %.0f%% GPU-resident - proceeding.", bare_model, ratio * 100)

    @staticmethod
    def _find_loaded_ollama_model(base: str, bare_model: str) -> Optional[dict]:
        resp = requests.get(f"{base}/api/ps", timeout=10)
        resp.raise_for_status()
        for entry in resp.json().get("models", []):
            if entry.get("model") == bare_model or entry.get("name") == bare_model:
                return entry
        return None


# --------------------------------------------------------------------------- #
# Convenience module-level API
# --------------------------------------------------------------------------- #


def refine_candidates(
    candidates: List[ClipCandidate],
    subtitles: List[SubtitleSegment],
    content_hint: str = "",
) -> List[CandidateClip]:
    """Convenience entry point for app.py. Returns ALL clips; filter on .is_clip_worthy as needed."""
    return LLMAgent().refine_candidates(candidates, subtitles, content_hint)
