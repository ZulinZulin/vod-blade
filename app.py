"""
app.py

Gradio front-end tying the whole StreamCutter pipeline together:

    fetchers -> chat_analyzer -> llm_agent -> exporters

Layout:
    1. Sources        - YouTube/subtitle input, Twitch VOD input, chat offset, local source video path
    2. Hype timeline   - interactive Plotly graph of chat hype score with detected spikes
    3. Subtitles       - collapsible full transcript preview, for verifying language/content
    4. Clip candidates - one card per candidate with editable start/end sliders
    5. Export          - "Download FCPXML/EDL" and "Inject into DaVinci Resolve"
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import List, Optional

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import requests

from config import settings
from core.chat_analyzer import ChatAnalyzer, ClipCandidate
from core.fetchers import (
    FetcherError, SubtitleSegment, fetch_subtitles, fetch_twitch_chat, fetch_twitch_vod, get_twitch_vod_title,
)
from core.llm_agent import DEFAULT_SYSTEM_PROMPT, CandidateClip, LLMAgent, OllamaGpuOffloadError
from core.preview import PreviewError, extract_preview_clip, extract_thumbnail
from core.session_store import (
    SessionError, delete_session, list_sessions, load_session, purge_sessions, save_session,
)
from exporters.davinci_api import DavinciAPIError, inject_into_resolve
from exporters.davinci_api import is_available as resolve_is_available
from exporters.xml_exporter import ExportError, export_edl_file, export_fcpxml_file

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


class _BenignProactorResetFilter(logging.Filter):
    """
    Windows' ProactorEventLoop logs a spurious ERROR (WinError 10054, "connection
    forcibly closed") whenever a client aborts an HTTP connection mid-transfer -
    which browsers do constantly while scrubbing/seeking a <video> element, since
    each seek cancels the in-flight range request and issues a new one. The
    socket is already gone by the time cleanup runs; this is a known, harmless
    asyncio/Windows quirk (never fully fixed upstream), not an application bug.
    Only this exact known-benign case is dropped - any other asyncio error still
    logs normally.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        if not isinstance(exc, ConnectionResetError):
            return True
        tb = record.exc_info[2]
        while tb is not None:
            if tb.tb_frame.f_code.co_name == "_call_connection_lost":
                return False
            tb = tb.tb_next
        return True


if sys.platform == "win32":
    logging.getLogger("asyncio").addFilter(_BenignProactorResetFilter())

MAX_CLIP_CARDS = 12  # cards rendered per page
CARD_WINDOW_PADDING_S = 180.0


# --------------------------------------------------------------------------- #
# Chat hype timeline plot
# --------------------------------------------------------------------------- #


def build_hype_timeline_figure(timeline_df: pd.DataFrame, candidates: List[ClipCandidate]) -> go.Figure:
    fig = go.Figure()
    if timeline_df.empty:
        fig.update_layout(title="No chat data yet - run an analysis to populate this graph.", template="plotly_dark")
        return fig

    fig.add_trace(go.Scatter(
        x=timeline_df["bin_start"], y=timeline_df["hype_score"],
        mode="lines", name="Hype score", line=dict(color="#7c5cff", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=timeline_df["bin_start"], y=timeline_df["rolling_mean"],
        mode="lines", name="Rolling baseline", line=dict(color="#888888", width=1, dash="dot"),
    ))
    if candidates:
        fig.add_trace(go.Scatter(
            x=[c.spike_time for c in candidates], y=[c.peak_hype_score for c in candidates],
            mode="markers", name="Detected spikes",
            marker=dict(color="#ff4d6d", size=11, symbol="star"),
        ))
        for c in candidates:
            fig.add_vrect(x0=c.window_start, x1=c.window_end, fillcolor="#ff4d6d", opacity=0.08, line_width=0)

    fig.update_layout(
        title="Chat Hype Timeline",
        xaxis_title="Stream time (s)",
        yaxis_title="Hype score",
        template="plotly_dark",
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# --------------------------------------------------------------------------- #
# Clip candidate cards
# --------------------------------------------------------------------------- #


def _format_hms(seconds: float) -> str:
    """Formats a raw seconds-into-VOD offset as an absolute HH:MM:SS (or MM:SS) timestamp."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _format_full_transcript(subtitles: List[SubtitleSegment]) -> str:
    """
    Renders the whole fetched subtitle track as '[HH:MM:SS -> HH:MM:SS] text' lines,
    so an operator can eyeball the actual language/content pulled for a stream and
    scroll to the context around a clip without leaving the app.
    """
    if not subtitles:
        return "(no subtitles loaded yet - run an analysis)"
    return "\n".join(
        f"[{_format_hms(seg.start)} -> {_format_hms(seg.end)}] {seg.text}" for seg in subtitles
    )


_TIMECODE_PREFIX_RE = re.compile(r"\[\d+(?:\.\d+)?-\d+(?:\.\d+)?\]\s*")
_CARD_TRANSCRIPT_LINES = 6  # fixed visible height; longer excerpts scroll inside it rather than growing the card


def _format_card_markdown(clip: CandidateClip, rank: int) -> str:
    if not clip.is_clip_worthy:
        reason = clip.rejection_reason or "no reason given"
        body = f"_Rejected: {reason}_"
        header = f"**#{rank}. [REJECTED] {clip.title}**"
    else:
        fallback_note = " _(fallback - LLM did not return a valid suggestion)_" if clip.used_fallback else ""
        # rejection_reason is deliberately never cleared on manual approval, so a clip
        # that was flagged (by the LLM or a prior manual reject) before being approved
        # still surfaces that context here instead of silently losing it.
        override_note = (
            f"\n\n_Note: previously flagged before being manually approved - reason: {clip.rejection_reason}_"
            if clip.rejection_reason else ""
        )
        body = f"{clip.summary}{override_note}"
        header = f"**#{rank}. {clip.title}**{fallback_note}"

    return (
        f"{header}\n\n"
        f"{body}\n\n"
        f"**{_format_hms(clip.start_time)} -> {_format_hms(clip.end_time)}**  "
        f"({clip.duration:.1f}s)  |  Viral score: **{clip.viral_score}/10**\n\n"
        f"Chat spike at {_format_hms(clip.spike_time)} (z={clip.peak_z_score:.2f})"
    )


def _format_card_transcript(clip: CandidateClip) -> str:
    """Full subtitle excerpt for a card, timecodes stripped. Not truncated - the UI
    renders this in a fixed-height textbox that scrolls internally for longer excerpts,
    so the card itself doesn't grow but the operator can still read the whole thing."""
    excerpt = _TIMECODE_PREFIX_RE.sub("", clip.transcript_excerpt.replace("\n", " "))
    return excerpt or "(no transcript captured for this range)"


def _visible_clips(clips: List[CandidateClip], show_rejected: bool) -> List[CandidateClip]:
    """Rejected clips always live in clips_state; this only controls what's displayed/paginated."""
    return clips if show_rejected else [c for c in clips if c.is_clip_worthy]


def _total_pages(clips: List[CandidateClip]) -> int:
    return max(1, -(-len(clips) // MAX_CLIP_CARDS))  # ceil division


def _clamp_page(page: int, clips: List[CandidateClip]) -> int:
    return max(0, min(page, _total_pages(clips) - 1))


def _page_label(clips: List[CandidateClip], page: int) -> str:
    if not clips:
        return "No clips yet - run an analysis."
    page = _clamp_page(page, clips)
    lo = page * MAX_CLIP_CARDS + 1
    hi = min(len(clips), (page + 1) * MAX_CLIP_CARDS)
    return f"Showing clips **{lo}-{hi}** of **{len(clips)}** (page {page + 1}/{_total_pages(clips)})"


_TOGGLE_LABEL_ACCEPTED = "Reject this clip"
_TOGGLE_LABEL_REJECTED = "Un-reject (restore)"


def _card_thumbnail_update(clip: CandidateClip, source_video_path: str) -> "gr.update":
    """
    Best-effort thumbnail for one card, taken at the midpoint of the clip's own
    (possibly manually-edited) start/end range. Never raises - a source video
    that isn't set yet, or an extraction failure, just leaves the thumbnail
    hidden rather than breaking the whole page render.
    """
    if not source_video_path:
        return gr.update(value=None, visible=False)
    try:
        thumb_path = extract_thumbnail(source_video_path, (clip.start_time + clip.end_time) / 2.0)
    except PreviewError as exc:
        logger.debug("Thumbnail generation skipped for candidate at t=%.1f: %s", clip.spike_time, exc)
        return gr.update(value=None, visible=False)
    return gr.update(value=str(thumb_path), visible=True)


def _build_card_updates(clips: List[CandidateClip], page: int, source_video_path: str = ""):
    """Returns MAX_CLIP_CARDS * (group, markdown, transcript, thumbnail, start_slider, end_slider, preview_video, toggle_btn) gr.update() tuples for one page."""
    page = _clamp_page(page, clips)
    start = page * MAX_CLIP_CARDS
    page_clips = clips[start:start + MAX_CLIP_CARDS]

    updates = []
    for i in range(MAX_CLIP_CARDS):
        # A card switching to a different clip (new page, filter toggle, re-run) must drop
        # any preview clip generated for whatever candidate previously occupied this slot -
        # otherwise the operator would see a stale preview under the wrong card.
        if i < len(page_clips):
            clip = page_clips[i]
            lo = max(0.0, clip.spike_time - CARD_WINDOW_PADDING_S)
            hi = clip.spike_time + CARD_WINDOW_PADDING_S
            toggle_label = _TOGGLE_LABEL_ACCEPTED if clip.is_clip_worthy else _TOGGLE_LABEL_REJECTED
            updates.extend([
                gr.update(visible=True),
                gr.update(value=_format_card_markdown(clip, start + i + 1)),
                gr.update(value=_format_card_transcript(clip)),
                _card_thumbnail_update(clip, source_video_path),
                gr.update(minimum=lo, maximum=hi, value=clip.start_time, step=0.1),
                gr.update(minimum=lo, maximum=hi, value=clip.end_time, step=0.1),
                gr.update(value=None, visible=False),
                gr.update(value=toggle_label),
            ])
        else:
            updates.extend([
                gr.update(visible=False), gr.update(value=""), gr.update(value=""),
                gr.update(value=None, visible=False), gr.update(), gr.update(),
                gr.update(value=None, visible=False), gr.update(value=_TOGGLE_LABEL_ACCEPTED),
            ])
    return updates


def go_to_page(clips: List[CandidateClip], show_rejected: bool, page: int, source_video_path: str, delta: int):
    """
    Moves `delta` pages (e.g. -1/+1) and returns the new page + refreshed card
    updates. delta=0 is used to just re-render the current page (e.g. after the
    "show rejected" toggle changes what's visible, without changing page).
    """
    visible = _visible_clips(clips, show_rejected)
    new_page = _clamp_page(page + delta, visible)
    return (new_page, _page_label(visible, new_page), *_build_card_updates(visible, new_page, source_video_path))


def _sync_bound(
    clips: List[CandidateClip], show_rejected: bool, page: int, value: float, idx: int, field_name: str
) -> List[CandidateClip]:
    """
    Applies a manual slider edit back into clips_state, guarding against inverted
    bounds. Resolves the edited card's position within the currently-visible
    (possibly filtered) list back to its real position in the full clips_state.
    """
    visible = _visible_clips(clips, show_rejected)
    local_pos = _clamp_page(page, visible) * MAX_CLIP_CARDS + idx
    if local_pos >= len(visible):
        return clips
    target = visible[local_pos]
    if field_name == "start_time" and value >= target.end_time:
        gr.Warning(f"Start time must be before end time ({target.end_time:.1f}s); edit ignored.")
        return clips
    if field_name == "end_time" and value <= target.start_time:
        gr.Warning(f"End time must be after start time ({target.start_time:.1f}s); edit ignored.")
        return clips
    real_idx = next(i for i, c in enumerate(clips) if c is target)
    updated = list(clips)
    updated[real_idx] = replace(target, **{field_name: value})
    return updated


_MANUAL_REJECTION_REASON = "Manually rejected by operator"


def do_toggle_worthy(
    clips: List[CandidateClip], show_rejected: bool, page: int, source_video_path: str, idx: int,
):
    """
    Manually overrides the LLM's accept/reject verdict for one card. Unlike
    _sync_bound, this can change which clips are *visible* (a newly-rejected
    clip vanishes, a newly-approved one appears, whenever "show rejected" is
    off) - so the whole page is re-rendered afterward instead of patching just
    this card, keeping pagination and card slots consistent.
    """
    visible = _visible_clips(clips, show_rejected)
    local_pos = _clamp_page(page, visible) * MAX_CLIP_CARDS + idx
    if local_pos >= len(visible):
        return (clips, page, _page_label(visible, page), *_build_card_updates(visible, page, source_video_path))

    target = visible[local_pos]
    real_idx = next(i for i, c in enumerate(clips) if c is target)
    updated = list(clips)
    if target.is_clip_worthy:
        updated[real_idx] = replace(target, is_clip_worthy=False, rejection_reason=_MANUAL_REJECTION_REASON)
    else:
        # rejection_reason is deliberately left as-is (see _format_card_markdown) so the
        # LLM's original call - or a prior manual rejection - stays visible for reference.
        updated[real_idx] = replace(target, is_clip_worthy=True)

    new_visible = _visible_clips(updated, show_rejected)
    new_page = _clamp_page(page, new_visible)
    return (
        updated, new_page, _page_label(new_visible, new_page),
        *_build_card_updates(new_visible, new_page, source_video_path),
    )


def do_unreject_all_manual(clips: List[CandidateClip], show_rejected: bool, page: int, source_video_path: str):
    """
    Restores every clip the OPERATOR manually rejected, leaving the LLM's own
    rejections untouched. Distinguished via the sentinel rejection_reason
    do_toggle_worthy sets on manual rejection - the LLM never produces that
    exact string on its own, so it's a reliable marker of operator origin.
    """
    is_manual_reject = lambda c: not c.is_clip_worthy and c.rejection_reason == _MANUAL_REJECTION_REASON
    count = sum(1 for c in clips if is_manual_reject(c))
    updated = [replace(c, is_clip_worthy=True) if is_manual_reject(c) else c for c in clips]

    if count == 0:
        gr.Warning("No manually-rejected clips to restore.")
    else:
        gr.Info(f"Restored {count} manually-rejected clip(s). LLM rejections were left untouched.")

    new_visible = _visible_clips(updated, show_rejected)
    new_page = _clamp_page(page, new_visible)
    return (
        updated, new_page, _page_label(new_visible, new_page),
        *_build_card_updates(new_visible, new_page, source_video_path),
    )


# --------------------------------------------------------------------------- #
# Session persistence
# --------------------------------------------------------------------------- #


def _session_choices():
    """(label, value) pairs for the sessions dropdown, newest first."""
    return [(p.name, str(p)) for p in list_sessions()]


def do_save_session(
    clips: List[CandidateClip], source_video_path: str, youtube_source: str,
    twitch_source: str, chat_offset: float, session_path: Optional[str],
):
    if not clips:
        raise gr.Error("No clips to save yet - run an analysis first.")
    try:
        out_path = save_session(
            clips, source_video_path, youtube_source, twitch_source, chat_offset,
            session_path=session_path or None, title_hint=get_twitch_vod_title(twitch_source) or "",
        )
    except SessionError as exc:
        raise gr.Error(str(exc))
    gr.Info(f"Session saved to {out_path.name}.")
    return str(out_path), gr.update(choices=_session_choices(), value=str(out_path))


def do_load_session(session_path: Optional[str]):
    if not session_path:
        raise gr.Error("Select a saved session to load first.")
    try:
        data = load_session(session_path)
    except SessionError as exc:
        raise gr.Error(str(exc))

    clips = data["clips"]
    visible = _visible_clips(clips, show_rejected=False)
    gr.Info(f"Loaded {len(clips)} clip(s) from {Path(session_path).name}.")
    return (
        clips, session_path, data["source_video_path"], data["youtube_source"],
        data["twitch_source"], data["chat_offset"], gr.update(value=False), 0,
        _page_label(visible, 0), *_build_card_updates(visible, 0, data["source_video_path"]),
    )


def do_refresh_sessions():
    return gr.update(choices=_session_choices())


def do_reset_delete_arm():
    """
    Resets the "Delete selected session" button back to its neutral label and
    clears the armed-for-delete state the instant the dropdown selection
    changes. do_delete_session's own armed_path != session_path check already
    guarantees a stale confirm can never delete the wrong file, but without
    this the button would keep visually showing "Confirm delete: <old file>"
    after switching selections until the next click quietly re-arms it.
    """
    return gr.update(value=_DELETE_SESSION_LABEL), None


def do_purge_sessions(confirmed: bool):
    """Permanently deletes every saved session file. Gated behind an explicit
    confirmation checkbox since this is irreversible and can destroy hours of
    real review work - a stray click on the button alone must not delete anything."""
    if not confirmed:
        raise gr.Error("Check the confirmation box first - purging permanently deletes ALL saved sessions.")
    count = purge_sessions()
    if count:
        gr.Info(f"Purged {count} saved session(s).")
    else:
        gr.Warning("No saved sessions to purge.")
    # Reset the confirmation checkbox and drop any tracked session reference so a
    # subsequent save doesn't try to overwrite a file that no longer exists.
    return gr.update(value=False), None, gr.update(choices=_session_choices(), value=None)


_DELETE_SESSION_LABEL = "Delete selected session"


def do_delete_session(session_path: Optional[str], armed_path: Optional[str], active_session_path: Optional[str]):
    """
    Deletes one saved session via a click-twice arm/disarm confirmation: the
    first click on a given selection only arms it (relabels the button), and
    only a second click against that SAME selection actually deletes - if the
    dropdown selection changes in between, this just re-arms for the new one
    instead of deleting the previously-armed file, so a stale confirm can
    never fire against the wrong session.
    """
    if not session_path:
        raise gr.Error("Select a saved session to delete first.")

    if armed_path != session_path:
        gr.Warning(f"Click again to permanently delete '{Path(session_path).name}'.")
        return (
            gr.update(value=f"Confirm delete: {Path(session_path).name}"),
            session_path, gr.update(), active_session_path,
        )

    try:
        delete_session(session_path)
    except SessionError as exc:
        raise gr.Error(str(exc))
    gr.Info(f"Deleted session {Path(session_path).name}.")
    # If the deleted file was the currently-loaded/active session, drop that reference
    # too, so a later "Save session" creates a fresh file instead of quietly reviving
    # the same path.
    new_active = None if active_session_path == session_path else active_session_path
    return (
        gr.update(value=_DELETE_SESSION_LABEL), None,
        gr.update(choices=_session_choices(), value=None), new_active,
    )


def do_preview_clip(source_video_path: str, start_time: float, end_time: float):
    """Extracts and plays the card's current start/end range (live slider values,
    including any manual edits) from the local source video - lets an operator
    quickly see/hear a candidate beyond its subtitle snippet. Triggered either by
    the "Preview clip" button or by clicking the card's thumbnail directly, both
    of which swap the static thumbnail out for the video player."""
    if not source_video_path:
        raise gr.Error("Please provide the local source video path to preview clips.")
    try:
        out_path = extract_preview_clip(source_video_path, start_time, end_time)
    except PreviewError as exc:
        raise gr.Error(str(exc))
    return gr.update(value=str(out_path), visible=True), gr.update(visible=False)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def run_pipeline(
    youtube_source: str,
    twitch_source: str,
    chat_offset: float,
    source_video_path: str,
    z_threshold: float,
    min_gap: float,
    pre_spike: float,
    post_spike: float,
    max_merged_duration: float,
    min_viral_score: float,
    content_hint: str,
    system_prompt: str,
    llm_provider: str,
    llm_model: str,
    llm_api_base: str,
    llm_api_key: str,
    progress=gr.Progress(),
):
    # Gradio Textbox/Dropdown components default to value=None (not "") when left
    # untouched and no explicit `value=""` was set on the component - normalize once
    # here so nothing downstream has to guard against None on what's conceptually an
    # optional *string* field.
    content_hint = content_hint or ""
    system_prompt = system_prompt or ""
    llm_model = llm_model or ""
    llm_api_base = llm_api_base or ""
    llm_api_key = llm_api_key or ""

    if not youtube_source or not twitch_source:
        raise gr.Error("Please provide both a subtitle source and a Twitch VOD source.")
    if z_threshold is None or z_threshold <= 0:
        raise gr.Error("Z-score threshold must be a positive number.")
    if min_gap is None or min_gap < 0 or pre_spike is None or pre_spike < 0 or post_spike is None or post_spike < 0:
        raise gr.Error("Spike spacing/window values must be zero or greater.")
    if max_merged_duration is None or max_merged_duration <= 0:
        raise gr.Error("Max merged candidate duration must be a positive number.")
    if min_viral_score is None or not (1 <= min_viral_score <= 10):
        raise gr.Error("Minimum viral score must be between 1 and 10.")
    if llm_provider in ("openai", "deepseek") and not (llm_api_key.strip() or settings.llm.api_key):
        raise gr.Error(
            f"No API key configured for provider '{llm_provider}'. Enter one above, or set "
            "LLM_API_KEY in .env if you'd rather not paste it into the UI each time."
        )

    progress(0.05, desc="Fetching subtitles...")
    try:
        subtitles = fetch_subtitles(youtube_source)
    except FetcherError as exc:
        raise gr.Error(f"Subtitle fetch failed: {exc}")

    progress(0.3, desc="Downloading Twitch chat log...")
    try:
        messages = fetch_twitch_chat(twitch_source, chat_offset_seconds=chat_offset)
    except FetcherError as exc:
        raise gr.Error(f"Twitch chat fetch failed: {exc}")

    progress(0.55, desc="Scoring chat hype...")
    # Only the spike-detection knobs are overridden here; scoring weights/emote lists/bin
    # width stay at their config.py defaults (those are stable across streams, unlike
    # sensitivity, which an editor will want to retune per chat's chattiness).
    hype_cfg = replace(
        settings.hype,
        z_score_threshold=z_threshold,
        min_seconds_between_spikes=min_gap,
        pre_spike_seconds=pre_spike,
        post_spike_seconds=post_spike,
        max_merged_duration_seconds=max_merged_duration,
    )
    candidates, timeline_df = ChatAnalyzer(config=hype_cfg).analyze_with_timeline(messages)
    if not candidates:
        gr.Warning(
            "No chat hype spikes cleared the Z-score threshold. Try lowering the Z-score threshold "
            "above for a quieter stream, or double check the chat offset."
        )

    progress(0.75, desc="Asking the LLM to judge and refine clip boundaries...")
    llm_cfg = replace(
        settings.llm,
        provider=llm_provider,
        model=llm_model.strip() or None,
        api_base=llm_api_base.strip() or None,
        api_key=llm_api_key.strip() or settings.llm.api_key,
        min_viral_score=int(min_viral_score),
    )

    def _report_judging_progress(completed: int, total: int) -> None:
        # LLM judging is the slowest, most opaque phase of a run (real calls against
        # a local/cloud model, one per candidate) - a per-candidate readout here
        # replaces the single static "judging..." message with visible forward motion.
        fraction = 0.75 + 0.25 * (completed / total if total else 1.0)
        progress(fraction, desc=f"Judging candidate {completed}/{total}...")

    # accepted + rejected, both kept
    agent = LLMAgent(config=llm_cfg, system_prompt=system_prompt)
    try:
        refined_clips = agent.refine_candidates(
            candidates, subtitles, content_hint=content_hint, progress_callback=_report_judging_progress,
        )
    except OllamaGpuOffloadError as exc:
        raise gr.Error(str(exc))
    refined_clips.sort(key=lambda c: c.viral_score, reverse=True)
    rejected_count = sum(1 for c in refined_clips if not c.is_clip_worthy)
    kept_count = len(refined_clips) - rejected_count

    progress(1.0, desc="Done")
    fig = build_hype_timeline_figure(timeline_df, candidates)
    status = (
        f"Analyzed {len(messages)} chat messages -> {len(candidates)} chat-spike candidate(s) -> "
        f"{kept_count} clip(s) kept"
        + (
            f", {rejected_count} rejected (toggle 'Show rejected candidates' to review)."
            if rejected_count else "."
        )
    )
    # Reset the toggle off on every fresh analysis so a stale "show rejected" state
    # from a previous run doesn't leak into a new one.
    visible = _visible_clips(refined_clips, show_rejected=False)
    page_updates = _build_card_updates(visible, page=0, source_video_path=source_video_path)
    transcript_text = _format_full_transcript(subtitles)

    # Auto-save immediately - judging a real VOD can take a long time (dozens of real
    # LLM calls), so the result is checkpointed to disk before the operator even starts
    # reviewing it, rather than living only in the browser's in-memory clips_state.
    try:
        session_path = save_session(
            refined_clips, source_video_path, youtube_source, twitch_source, chat_offset,
            title_hint=get_twitch_vod_title(twitch_source) or "",
        )
        session_update = gr.update(choices=_session_choices(), value=str(session_path))
    except SessionError as exc:
        logger.warning("Auto-save of the finished session failed: %s", exc)
        session_path = None
        session_update = gr.update()

    return (
        fig, refined_clips, status, 0, gr.update(value=False), _page_label(visible, 0),
        transcript_text, str(session_path) if session_path else None, session_update, *page_updates,
    )


# --------------------------------------------------------------------------- #
# LLM provider helpers
# --------------------------------------------------------------------------- #


def do_fetch_models(provider: str, api_base: str, api_key: str):
    """
    Queries the selected provider's model-listing endpoint and returns the
    fetched IDs as Dropdown choices, so the Model field can be picked from a
    real, current catalog instead of typed blind. Ollama uses its own
    /api/tags shape; every other provider here follows the OpenAI-compatible
    GET {base}/models convention (openai, deepseek, openrouter, nanogpt all do).
    """
    base = (api_base or "").strip().rstrip("/") or settings.llm.provider_api_base_defaults.get(provider, "")
    if not base:
        raise gr.Error(f"No API base URL known for provider '{provider}'. Enter one above first.")

    key = (api_key or "").strip() or settings.llm.api_key
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    try:
        if provider == "ollama":
            resp = requests.get(f"{base}/api/tags", timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            model_ids = sorted({
                str(entry.get("model") or entry.get("name"))
                for entry in payload.get("models", [])
                if entry.get("model") or entry.get("name")
            })
        else:
            resp = requests.get(f"{base}/models", headers=headers, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            entries = payload.get("data", payload) if isinstance(payload, dict) else payload
            if not isinstance(entries, list):
                raise gr.Error(f"Unexpected response shape from {base}/models (expected a list of models).")
            model_ids = sorted({
                str(entry.get("id") or entry.get("name"))
                for entry in entries
                if isinstance(entry, dict) and (entry.get("id") or entry.get("name"))
            })
    except requests.exceptions.RequestException as exc:
        raise gr.Error(f"Failed to fetch models from {base}: {exc}")
    except ValueError as exc:  # response body wasn't valid JSON
        raise gr.Error(f"Provider returned a non-JSON response from {base}/models: {exc}")

    if not model_ids:
        raise gr.Error(f"No models returned by {base} - check the API base/key above.")

    gr.Info(f"Fetched {len(model_ids)} model(s) from {provider}.")
    return gr.update(choices=model_ids)


def do_provider_changed(provider: str):
    """Switching providers clears any previously-fetched model list, which wouldn't apply here anyway."""
    default_base = settings.llm.provider_api_base_defaults.get(provider, "")
    return (
        gr.update(choices=[], value=""),
        gr.update(placeholder=f"e.g. {default_base}" if default_base else None),
    )


# --------------------------------------------------------------------------- #
# VOD video download
# --------------------------------------------------------------------------- #


def do_download_vod(twitch_source: str, quality: str):
    """
    Downloads the full Twitch VOD video file via TwitchDownloaderCLI, so
    exports have a local source file to reference. This is deliberately NOT
    part of run_pipeline: analysis only needs subtitles/chat and is fast,
    while a VOD download can be a multi-GB, multi-hour operation - you may
    want to review candidates before committing to it.

    A generator so the UI shows an immediate "in progress" state without
    needing real progress-percentage tracking: Gradio renders each yielded
    tuple as it's produced, so the first yield shows up right away and the
    second lands whenever the (blocking) download actually finishes.
    """
    if not twitch_source:
        raise gr.Error("Please provide a Twitch VOD URL or ID first.")

    yield (
        "Downloading VOD... this can take a while for long streams (the full VOD is downloaded, "
        "not a trimmed range, since clip timings are offsets from its start). "
        "This message will update when the download finishes.",
        gr.update(interactive=False),
        gr.update(),
    )

    try:
        video_path = fetch_twitch_vod(twitch_source, quality=quality or None)
    except FetcherError as exc:
        yield f"VOD download failed: {exc}", gr.update(interactive=True), gr.update()
        return

    yield (
        f"VOD downloaded: `{video_path}`. Source video path below has been filled in automatically.",
        gr.update(interactive=True),
        gr.update(value=str(video_path)),
    )


# --------------------------------------------------------------------------- #
# Export handlers
# --------------------------------------------------------------------------- #


def do_export_files(clips: List[CandidateClip], source_video_path: str):
    accepted = [c for c in clips if c.is_clip_worthy]
    if not accepted:
        raise gr.Error(
            "No accepted clips to export yet - run an analysis first. "
            "(Clips rejected by the LLM are excluded from exports regardless of the review toggle.)"
        )
    if not source_video_path:
        raise gr.Error("Please provide the local source video path (needed by the FCPXML asset reference).")
    try:
        fcpxml_path = export_fcpxml_file(accepted, source_video_path)
        edl_path = export_edl_file(accepted, source_video_path=source_video_path)
    except ExportError as exc:
        raise gr.Error(str(exc))
    return str(fcpxml_path), str(edl_path)


def do_inject_resolve(clips: List[CandidateClip], source_video_path: str):
    accepted = [c for c in clips if c.is_clip_worthy]
    if not accepted:
        raise gr.Error(
            "No accepted clips to inject yet - run an analysis first. "
            "(Clips rejected by the LLM are excluded from injection regardless of the review toggle.)"
        )
    if not source_video_path:
        raise gr.Error("Please provide the local source video path to import into Resolve's Media Pool.")
    if not resolve_is_available():
        raise gr.Error(
            "DaVinci Resolve isn't reachable. Make sure it's running with Preferences > General > "
            "'External scripting using' set to Local, or use the FCPXML/EDL download instead."
        )
    try:
        result = inject_into_resolve(accepted, source_video_path)
    except DavinciAPIError as exc:
        raise gr.Error(str(exc))
    return (
        f"Injected {result.clips_added} clip(s) into timeline **'{result.timeline_name}'** "
        f"in project **'{result.project_name}'** ({result.clips_failed} failed)."
    )


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #


def build_app() -> gr.Blocks:
    with gr.Blocks(title="StreamCutter") as demo:
        gr.Markdown("# StreamCutter\nTurn Twitch chat hype spikes into DaVinci Resolve-ready viral clips.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Sources")
                youtube_input = gr.Textbox(
                    label="YouTube URL or local .srt/.vtt/.txt transcript path",
                    placeholder="https://youtube.com/watch?v=... or C:\\path\\to\\transcript.srt",
                )
                twitch_input = gr.Textbox(
                    label="Twitch VOD URL or ID",
                    placeholder="https://twitch.tv/videos/123456789",
                )
                offset_input = gr.Number(
                    label="Chat offset (s) - Twitch clock minus YouTube clock",
                    value=settings.fetcher.default_chat_offset_seconds,
                )
                source_video_input = gr.Textbox(
                    label="Local source video path (used by exports)",
                    placeholder=r"C:\path\to\downloaded_video.mp4",
                )
                with gr.Row():
                    vod_quality_input = gr.Textbox(
                        label="VOD download quality",
                        value=settings.fetcher.twitch_video_quality,
                        placeholder="best, worst, audio_only, or a resolution like 720p60",
                    )
                    download_vod_btn = gr.Button("Download VOD")
                download_status = gr.Markdown("")
                with gr.Accordion("Hype detection settings (advanced)", open=False):
                    with gr.Row():
                        z_threshold_input = gr.Slider(
                            label="Z-score threshold (higher = fewer, stronger-only spikes)",
                            minimum=1.0, maximum=6.0, step=0.1,
                            value=settings.hype.z_score_threshold,
                        )
                        min_gap_input = gr.Slider(
                            label="Min seconds between spikes",
                            minimum=0, maximum=300, step=5,
                            value=settings.hype.min_seconds_between_spikes,
                        )
                    with gr.Row():
                        pre_spike_input = gr.Slider(
                            label="Seconds before spike (window start)",
                            minimum=0, maximum=180, step=5,
                            value=settings.hype.pre_spike_seconds,
                        )
                        post_spike_input = gr.Slider(
                            label="Seconds after spike (window end)",
                            minimum=0, maximum=180, step=5,
                            value=settings.hype.post_spike_seconds,
                        )
                    max_merged_duration_input = gr.Slider(
                        label="Max merged candidate duration (s) - caps how many nearby spikes can chain-merge into one window",
                        minimum=30, maximum=600, step=10,
                        value=settings.hype.max_merged_duration_seconds,
                    )
                with gr.Accordion("LLM provider (advanced)", open=False):
                    llm_provider_input = gr.Dropdown(
                        label="LLM provider",
                        choices=["openai", "deepseek", "ollama", "openrouter", "nanogpt"],
                        value=settings.llm.provider,
                    )
                    with gr.Row():
                        llm_model_input = gr.Dropdown(
                            label="Model (optional - blank uses the provider's default; type to search, "
                                  "or click 'Fetch models' for a live list)",
                            choices=[settings.llm.model] if settings.llm.model else [],
                            value=settings.llm.model or "",
                            allow_custom_value=True,
                            filterable=True,
                        )
                        fetch_models_btn = gr.Button("Fetch models from API", size="sm")
                    with gr.Row():
                        llm_api_base_input = gr.Textbox(
                            label="API base URL (optional - blank uses the provider's default)",
                            value=settings.llm.api_base or "",
                            placeholder="e.g. https://nano-gpt.com/api/v1",
                        )
                        llm_api_key_input = gr.Textbox(
                            label="API key (optional - blank uses LLM_API_KEY from .env; ignored for Ollama)",
                            value="",
                            type="password",
                        )
                    gr.Markdown(
                        "_openrouter and nanogpt are third-party aggregators exposing many underlying "
                        "models - 'Fetch models from API' lists what's actually available from whichever "
                        "provider/API base/key is set above (needs a valid key for most providers). Local "
                        "Ollama note: the app can detect the model being evicted to CPU/RAM, but not GPU "
                        "**compute** contention - another GPU-heavy app (image/video generation, games) "
                        "running at the same time can still make judgment run dramatically slower with no "
                        "error, even while the model stays fully loaded on the GPU. Close other GPU-heavy "
                        "applications for best performance._"
                    )
                with gr.Accordion("LLM judgment settings (advanced)", open=False):
                    min_viral_score_input = gr.Slider(
                        label="Minimum viral score to keep (1-10) - clips the LLM itself scores below "
                              "this are rejected even if it called them worthy",
                        minimum=1, maximum=10, step=1,
                        value=settings.llm.min_viral_score,
                    )
                    content_hint_input = gr.Textbox(
                        label="Stream context hint for the LLM (optional)",
                        value="",
                        placeholder="e.g. 'podcast, lots of talking - be strict about what counts as "
                                    "notable' or 'fast-paced gaming stream'",
                    )
                with gr.Accordion("System prompt (advanced - edit with care)", open=False):
                    gr.Markdown(
                        "_This is the full instruction set the LLM judges every candidate against. "
                        "Edit it to change how judgment works; leave it as-is otherwise. If your edit "
                        "breaks the required JSON output format, calls will fail validation and those "
                        "candidates fall back to their raw chat-spike window instead of crashing._"
                    )
                    system_prompt_input = gr.Textbox(
                        value=DEFAULT_SYSTEM_PROMPT,
                        lines=20,
                        max_lines=60,
                        show_label=False,
                    )
                    reset_system_prompt_btn = gr.Button("Reset to default", size="sm")
                run_btn = gr.Button("Analyze Stream", variant="primary")
                status_box = gr.Markdown("")
            with gr.Column(scale=2):
                gr.Markdown("### 2. Chat Hype Timeline")
                hype_plot = gr.Plot()

        with gr.Accordion("3. Subtitles (verify language / scroll for context around a clip)", open=False):
            subtitles_display = gr.Textbox(
                value="(no subtitles loaded yet - run an analysis)",
                lines=15,
                max_lines=30,
                interactive=False,
                show_label=False,
            )

        gr.Markdown("### 4. Clip Candidates")
        clips_state = gr.State([])
        page_state = gr.State(0)
        session_path_state = gr.State(None)
        delete_armed_state = gr.State(None)

        with gr.Row():
            session_dropdown = gr.Dropdown(
                label="Saved sessions", choices=_session_choices(), value=None,
                filterable=True,
            )
            refresh_sessions_btn = gr.Button("Refresh", size="sm")
            load_session_btn = gr.Button("Load session", size="sm")
            save_session_btn = gr.Button("Save session", size="sm")
            delete_session_btn = gr.Button(_DELETE_SESSION_LABEL, size="sm")
        with gr.Row():
            confirm_purge_checkbox = gr.Checkbox(
                label="Confirm purge (deletes ALL saved sessions permanently)", value=False,
            )
            purge_sessions_btn = gr.Button("Purge saves", variant="stop", size="sm")

        with gr.Row():
            show_rejected_checkbox = gr.Checkbox(
                label="Show rejected candidates (statistical spikes the LLM judged not notable)",
                value=False,
            )
            unreject_all_btn = gr.Button("Un-reject all (manual only)", size="sm")
        with gr.Row():
            prev_page_btn = gr.Button("< Prev", size="sm")
            page_label = gr.Markdown("No clips yet - run an analysis.")
            next_page_btn = gr.Button("Next >", size="sm")

        card_components = []
        for _ in range(MAX_CLIP_CARDS):
            with gr.Group(visible=False) as card_group:
                # Thumbnail and video share the same slot: the thumbnail is a cheap
                # auto-generated frame shown by default; clicking it (or "Preview clip")
                # extracts and swaps in the real playable clip in its place.
                card_thumbnail = gr.Image(visible=False, interactive=False, show_label=False, height=240)
                preview_video = gr.Video(visible=False, height=240)
                card_md = gr.Markdown()
                card_transcript = gr.Textbox(
                    lines=_CARD_TRANSCRIPT_LINES, max_lines=_CARD_TRANSCRIPT_LINES,
                    interactive=False, show_label=False,
                )
                with gr.Row():
                    start_slider = gr.Slider(label="Start (seconds into VOD)", minimum=0, maximum=1, step=0.1)
                    end_slider = gr.Slider(label="End (seconds into VOD)", minimum=0, maximum=1, step=0.1)
                with gr.Row():
                    preview_btn = gr.Button("Preview clip", size="sm")
                    toggle_btn = gr.Button(_TOGGLE_LABEL_ACCEPTED, size="sm")
            card_components.append({
                "group": card_group, "md": card_md, "transcript": card_transcript, "thumbnail": card_thumbnail,
                "start": start_slider, "end": end_slider,
                "preview_btn": preview_btn, "video": preview_video, "toggle_btn": toggle_btn,
            })

        gr.Markdown("### 5. Export")
        resolve_hint = (
            "DaVinci Resolve detected and reachable."
            if resolve_is_available()
            else "DaVinci Resolve not detected - 'Inject' will fall back to an error; use the file download instead."
        )
        gr.Markdown(f"_{resolve_hint}_")
        with gr.Row():
            export_files_btn = gr.Button("Download FCPXML/EDL")
            inject_btn = gr.Button("Inject into DaVinci Resolve")
        with gr.Row():
            fcpxml_file = gr.File(label="FCPXML")
            edl_file = gr.File(label="EDL")
        inject_status = gr.Markdown("")

        # --- wiring ---

        card_outputs = []
        for c in card_components:
            card_outputs.extend([
                c["group"], c["md"], c["transcript"], c["thumbnail"],
                c["start"], c["end"], c["video"], c["toggle_btn"],
            ])

        reset_system_prompt_btn.click(
            fn=lambda: DEFAULT_SYSTEM_PROMPT,
            inputs=[],
            outputs=[system_prompt_input],
        )

        run_btn.click(
            fn=run_pipeline,
            inputs=[
                youtube_input, twitch_input, offset_input, source_video_input,
                z_threshold_input, min_gap_input, pre_spike_input, post_spike_input,
                max_merged_duration_input,
                min_viral_score_input, content_hint_input, system_prompt_input,
                llm_provider_input, llm_model_input, llm_api_base_input, llm_api_key_input,
            ],
            outputs=[
                hype_plot, clips_state, status_box, page_state,
                show_rejected_checkbox, page_label, subtitles_display,
                session_path_state, session_dropdown, *card_outputs,
            ],
        )

        download_vod_btn.click(
            fn=do_download_vod,
            inputs=[twitch_input, vod_quality_input],
            outputs=[download_status, download_vod_btn, source_video_input],
        )

        fetch_models_btn.click(
            fn=do_fetch_models,
            inputs=[llm_provider_input, llm_api_base_input, llm_api_key_input],
            outputs=[llm_model_input],
        )
        llm_provider_input.change(
            fn=do_provider_changed,
            inputs=[llm_provider_input],
            outputs=[llm_model_input, llm_api_base_input],
        )

        prev_page_btn.click(
            fn=partial(go_to_page, delta=-1),
            inputs=[clips_state, show_rejected_checkbox, page_state, source_video_input],
            outputs=[page_state, page_label, *card_outputs],
        )
        next_page_btn.click(
            fn=partial(go_to_page, delta=1),
            inputs=[clips_state, show_rejected_checkbox, page_state, source_video_input],
            outputs=[page_state, page_label, *card_outputs],
        )
        show_rejected_checkbox.change(
            fn=partial(go_to_page, delta=0),
            inputs=[clips_state, show_rejected_checkbox, page_state, source_video_input],
            outputs=[page_state, page_label, *card_outputs],
        )
        unreject_all_btn.click(
            fn=do_unreject_all_manual,
            inputs=[clips_state, show_rejected_checkbox, page_state, source_video_input],
            outputs=[clips_state, page_state, page_label, *card_outputs],
        )

        refresh_sessions_btn.click(fn=do_refresh_sessions, inputs=[], outputs=[session_dropdown])
        save_session_btn.click(
            fn=do_save_session,
            inputs=[clips_state, source_video_input, youtube_input, twitch_input, offset_input, session_path_state],
            outputs=[session_path_state, session_dropdown],
        )
        load_session_btn.click(
            fn=do_load_session,
            inputs=[session_dropdown],
            outputs=[
                clips_state, session_path_state, source_video_input, youtube_input, twitch_input, offset_input,
                show_rejected_checkbox, page_state, page_label, *card_outputs,
            ],
        )
        purge_sessions_btn.click(
            fn=do_purge_sessions,
            inputs=[confirm_purge_checkbox],
            outputs=[confirm_purge_checkbox, session_path_state, session_dropdown],
        )
        delete_session_btn.click(
            fn=do_delete_session,
            inputs=[session_dropdown, delete_armed_state, session_path_state],
            outputs=[delete_session_btn, delete_armed_state, session_dropdown, session_path_state],
        )
        session_dropdown.change(
            fn=do_reset_delete_arm,
            inputs=[],
            outputs=[delete_session_btn, delete_armed_state],
        )

        for idx, c in enumerate(card_components):
            c["start"].release(
                fn=partial(_sync_bound, idx=idx, field_name="start_time"),
                inputs=[clips_state, show_rejected_checkbox, page_state, c["start"]],
                outputs=[clips_state],
            )
            c["end"].release(
                fn=partial(_sync_bound, idx=idx, field_name="end_time"),
                inputs=[clips_state, show_rejected_checkbox, page_state, c["end"]],
                outputs=[clips_state],
            )
            c["preview_btn"].click(
                fn=do_preview_clip,
                inputs=[source_video_input, c["start"], c["end"]],
                outputs=[c["video"], c["thumbnail"]],
            )
            c["thumbnail"].select(
                fn=do_preview_clip,
                inputs=[source_video_input, c["start"], c["end"]],
                outputs=[c["video"], c["thumbnail"]],
            )
            c["toggle_btn"].click(
                fn=partial(do_toggle_worthy, idx=idx),
                inputs=[clips_state, show_rejected_checkbox, page_state, source_video_input],
                outputs=[clips_state, page_state, page_label, *card_outputs],
            )

        export_files_btn.click(
            fn=do_export_files,
            inputs=[clips_state, source_video_input],
            outputs=[fcpxml_file, edl_file],
        )
        inject_btn.click(
            fn=do_inject_resolve,
            inputs=[clips_state, source_video_input],
            outputs=[inject_status],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch(server_port=7862)
