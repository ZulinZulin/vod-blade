"""
app.py

Gradio front-end tying the whole VOD BLADE pipeline together:

    fetchers -> chat_analyzer -> llm_agent -> exporters

Layout:
    Sources & Settings - collapsible accordion at the top (Sources plus all advanced settings
                          behind their own nested accordions), full-width but collapses down to
                          a single header line once a run is underway - it doesn't need to stay
                          expanded, and unlike a side column it costs no permanent horizontal
                          space either way.
    Hype timeline      - interactive Plotly graph of chat hype score with detected spikes
    Subtitles          - collapsible full transcript preview, for verifying language/content
    Clip candidates    - one card per candidate with editable start/end sliders
    Export             - "Download FCPXML/EDL" and "Inject into DaVinci Resolve"
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Optional

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import requests

from config import DOWNLOADS_DIR, SESSIONS_DIR, settings
from core.audio_analyzer import AudioAnalysisError, AudioAnalyzer, merge_with_chat_candidates
from core.chat_analyzer import ChatAnalyzer, ClipCandidate
from core.fetchers import (
    FetcherError, fetch_subtitles, fetch_twitch_chat, fetch_twitch_vod, get_twitch_vod_title,
)
from core.llm_agent import (
    DEFAULT_SYSTEM_PROMPT, CandidateClip, LLMAgent, OllamaGpuOffloadError,
    _title_label_for_source, build_stat_only_clip, format_transcript, select_subtitle_window,
)
from core.sound_event_classifier import SoundEventClassifier, SoundEventError, merge_sound_events
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


# How often (seconds) the skyline's floor resamples rolling_mean between spikes -
# coarser than bin resolution on purpose, since rolling_mean is already smooth by
# construction and a sparse sample is enough to show real drift without redrawing
# every noisy raw bin.
_HYPE_BASELINE_SAMPLE_INTERVAL_S = 60.0
# How many bin-widths before/after each spike the sharp rise/fall happens.
_HYPE_SETTLE_BIN_MULTIPLE = 1.5


def _nearest_rolling_mean(timeline_df: pd.DataFrame, x: float) -> float:
    idx = (timeline_df["bin_start"] - x).abs().idxmin()
    return float(timeline_df.loc[idx, "rolling_mean"])


def _build_skyline_anchors(timeline_df: pd.DataFrame, candidates: List[ClipCandidate]) -> "tuple[list, list]":
    """
    Shared "skyline" anchor-point builder for both the chat and audio traces:
    tracing every raw bin reads as noisy even where nothing notable is
    happening (both chat volume and audio RMS are bursty at bin resolution).
    Instead the line is built from a sparse set of anchors - a slowly-drifting
    floor (rolling_mean, resampled every _HYPE_BASELINE_SAMPLE_INTERVAL_S
    seconds) plus a sharp rise/settle pair around each real spike, both read
    from the same rolling_mean the spike detector itself measures deviations
    against - so a spike's height above the floor stays an honest reflection
    of real chat/audio volume instead of an arbitrary flat baseline.

    `candidates` must already be filtered to ones whose peak_hype_score is on
    the SAME scale as timeline_df (chat hype score, or audio RMS) - mixing
    scales here would draw a nonsense-height vertex on the wrong curve.
    """
    bin_width = float(timeline_df["bin_start"].diff().median() or 5.0)
    step_rows = max(1, round(_HYPE_BASELINE_SAMPLE_INTERVAL_S / bin_width))
    anchors = {
        float(row.bin_start): float(row.rolling_mean)
        for row in timeline_df.iloc[::step_rows].itertuples(index=False)
    }
    last_row = timeline_df.iloc[-1]
    anchors[float(last_row["bin_start"])] = float(last_row["rolling_mean"])  # always reach the true end

    t_min = float(timeline_df["bin_start"].min())
    t_max = float(timeline_df["bin_start"].max())
    settle_offset = _HYPE_SETTLE_BIN_MULTIPLE * bin_width
    for c in candidates:
        pre_x = max(t_min, c.spike_time - settle_offset)
        post_x = min(t_max, c.spike_time + settle_offset)
        anchors[pre_x] = _nearest_rolling_mean(timeline_df, pre_x)
        anchors[post_x] = _nearest_rolling_mean(timeline_df, post_x)
        anchors[c.spike_time] = c.peak_hype_score  # the real recorded peak, not a smoothed value

    xs, ys = zip(*sorted(anchors.items()))
    return list(xs), list(ys)


def build_hype_timeline_figure(
    timeline_df: pd.DataFrame,
    candidates: List[ClipCandidate],
    audio_timeline_df: Optional[pd.DataFrame] = None,
    audio_candidates: Optional[List[ClipCandidate]] = None,
) -> go.Figure:
    """
    `candidates` is the final, post-merge list (chat/audio/sound_event tags mixed
    together in various "+"-joined combinations - see core.audio_analyzer and
    core.sound_event_classifier's merge functions) and is used for the chat
    skyline/markers. `audio_candidates`, when given, is the ORIGINAL pre-merge
    audio-detector output - used for the audio skyline/markers instead, since a
    candidate enriched with the "audio" tag keeps its ORIGINAL peak_hype_score
    (chat's scale if it started as a chat candidate), which would draw a
    nonsense-height vertex on the audio trace.

    Marker categories are decided by which tags are present, not by an exact
    match on the whole (growing) set of possible source strings - a candidate
    tagged "chat+audio+sound_event" is still fundamentally a chat candidate for
    plotting purposes, just one every detector agrees on.
    """
    fig = go.Figure()
    if timeline_df.empty:
        fig.update_layout(title="No chat data yet - run an analysis to populate this graph.", template="plotly_dark")
        return fig

    def tags(c: ClipCandidate) -> List[str]:
        return c.source.split("+")

    chat_line_candidates = [c for c in candidates if "chat" in tags(c)]
    xs, ys = _build_skyline_anchors(timeline_df, chat_line_candidates)
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines", name="Hype score", line=dict(color="#7c5cff", width=1.5),
    ))

    # Marker categories so an operator can tell at a glance which detector(s) flagged a
    # given moment: chat-only (pink circle), chat confirmed by a coincident audio peak
    # (gold circle - same shape, since it IS a chat candidate), audio-only (teal diamond,
    # plotted against the secondary axis below), and a rare sound-event-only candidate
    # neither chat nor audio-RMS caught (purple star, plotted on the primary axis using
    # the chat timeline's own value at that time - its real peak_hype_score is a [0, 1]
    # model confidence, not a chat-scale number, so it can't be plotted at face value).
    chat_only = [c for c in candidates if "chat" in tags(c) and "audio" not in tags(c)]
    chat_and_audio = [c for c in candidates if "chat" in tags(c) and "audio" in tags(c)]
    audio_only = [c for c in candidates if "audio" in tags(c) and "chat" not in tags(c)]
    sound_event_only = [c for c in candidates if tags(c) == ["sound_event"]]
    if chat_only:
        fig.add_trace(go.Scatter(
            x=[c.spike_time for c in chat_only], y=[c.peak_hype_score for c in chat_only],
            mode="markers", name="Chat spike",
            marker=dict(color="#ff4d6d", size=7, symbol="circle"),
        ))
    if chat_and_audio:
        fig.add_trace(go.Scatter(
            x=[c.spike_time for c in chat_and_audio], y=[c.peak_hype_score for c in chat_and_audio],
            mode="markers", name="Chat+audio confirmed",
            marker=dict(color="#facc15", size=8, symbol="circle"),
        ))
    if sound_event_only:
        fig.add_trace(go.Scatter(
            x=[c.spike_time for c in sound_event_only],
            y=[_nearest_rolling_mean(timeline_df, c.spike_time) for c in sound_event_only],
            mode="markers", name="Sound event (unconfirmed)",
            marker=dict(color="#c084fc", size=9, symbol="star"),
        ))
    for c in candidates:
        fig.add_vrect(x0=c.window_start, x1=c.window_end, fillcolor="#ff4d6d", opacity=0.08, line_width=0)

    # A sound event can attach to a chat or audio candidate too (not just show up on its
    # own as sound_event_only above) - a hollow star "halo" drawn over that candidate's
    # existing marker flags this without inventing a new category per combination (chat+
    # event, audio+event, chat+audio+event, ...). Split by axis for the same reason the
    # primary marker groups are: a chat-tagged candidate's peak_hype_score is only valid
    # on the primary axis, an audio-only one only on the secondary.
    primary_axis_events = [c for c in candidates if c.sound_events and "chat" in tags(c)]
    secondary_axis_events = [c for c in candidates if c.sound_events and "audio" in tags(c) and "chat" not in tags(c)]
    if primary_axis_events:
        fig.add_trace(go.Scatter(
            x=[c.spike_time for c in primary_axis_events], y=[c.peak_hype_score for c in primary_axis_events],
            mode="markers", name="Sound event detected",
            marker=dict(color="#c084fc", size=16, symbol="star-open", line=dict(width=2)),
        ))

    layout_kwargs = dict(
        title="Chat Hype Timeline",
        xaxis_title="Stream time (s)",
        yaxis_title="Hype score",
        template="plotly_dark",
        height=380,
        margin=dict(l=40, r=60, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    if audio_timeline_df is not None and not audio_timeline_df.empty:
        audio_xs, audio_ys = _build_skyline_anchors(audio_timeline_df, audio_candidates or [])
        fig.add_trace(go.Scatter(
            x=audio_xs, y=audio_ys, yaxis="y2",
            mode="lines", name="Audio energy", line=dict(color="#2dd4bf", width=1.5),
        ))
        if audio_only:
            fig.add_trace(go.Scatter(
                x=[c.spike_time for c in audio_only], y=[c.peak_hype_score for c in audio_only], yaxis="y2",
                mode="markers", name="Audio-only peak",
                marker=dict(color="#2dd4bf", size=8, symbol="diamond"),
            ))
        if secondary_axis_events:
            fig.add_trace(go.Scatter(
                x=[c.spike_time for c in secondary_axis_events],
                y=[c.peak_hype_score for c in secondary_axis_events], yaxis="y2",
                mode="markers", name="Sound event detected", showlegend=not primary_axis_events,
                marker=dict(color="#c084fc", size=16, symbol="star-open", line=dict(width=2)),
            ))
        layout_kwargs["yaxis2"] = dict(
            title="Audio RMS energy", overlaying="y", side="right", showgrid=False,
        )

    fig.update_layout(**layout_kwargs)
    return fig


# Fed to hype_plot.change(js=...) so a click on the rendered Plotly chart reaches
# a Python callback. gr.Plot has no click/select event of its own, so this listens
# for Plotly's own native 'plotly_click' directly on the chart's div and forwards
# the clicked point's x (time) through a visible="hidden" bridge textbox + button -
# visible=False would unmount them from the DOM entirely, leaving nothing for this
# script to write into. The clicked point's x is forwarded as-is rather than
# resolved to a specific candidate here, because a click landing exactly on a
# marker can get attributed to the underlying line trace instead (same x/y,
# different curveNumber) - matching by x-distance in Python, against the full
# precision spike_time list it already has in memory, sidesteps that ambiguity.
_HYPE_CLICK_BRIDGE_JS = """
() => {
    // hype_plot.change() fires the instant Gradio's value updates, which is before
    // Plotly's own async rendering has actually inserted .js-plotly-plot into the
    // DOM - so the div isn't there yet on the first attempt. Poll briefly instead
    // of assuming it already exists.
    let attempts = 0;
    const tryBind = () => {
        const plotDiv = document.querySelector('#hype_plot .js-plotly-plot');
        if (!plotDiv) {
            if (attempts++ < 40) setTimeout(tryBind, 100);
            return;
        }
        if (plotDiv._scHypeClickBound) return;
        plotDiv._scHypeClickBound = true;
        plotDiv.on('plotly_click', (data) => {
            const point = data.points && data.points[0];
            if (!point || typeof point.x !== 'number') return;
            const hiddenInput = document.querySelector('#hype_click_bridge textarea, #hype_click_bridge input');
            const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
            nativeSetter.call(hiddenInput, String(point.x));
            hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
            document.querySelector('#hype_click_bridge_button').click();
        });
    };
    tryBind();
}
"""

# Fed to the click-handling button's .then(js=...). Runs after the page/card
# updates from do_hype_plot_click have rendered, and reads the slot index that
# callback wrote into hype_highlight_signal (empty string = no match, do nothing).
_HYPE_HIGHLIGHT_SCROLL_JS = """
() => {
    const sig = document.querySelector('#hype_highlight_signal textarea, #hype_highlight_signal input');
    const slot = sig ? sig.value : '';
    if (!slot) return;
    const card = document.getElementById('candidate-card-slot-' + slot);
    if (!card) return;
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    card.classList.add('candidate-card-highlight');
    setTimeout(() => card.classList.remove('candidate-card-highlight'), 2000);
}
"""


# --------------------------------------------------------------------------- #
# Clip candidate cards
# --------------------------------------------------------------------------- #


def _format_hms(seconds: float) -> str:
    """Formats a raw seconds-into-VOD offset as an absolute HH:MM:SS (or MM:SS) timestamp."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _relative_offset_label(event_time: Optional[float], clip_start: float, clip_end: float) -> str:
    """
    " (MM:SS into this clip)" if event_time falls within [clip_start, clip_end], else "".
    The LLM's final chosen start/end is often a tighter, different range than the padded
    detection window an audio/event candidate's own timestamp was found in - deliberately
    silent rather than showing a confusing negative/out-of-range offset when it falls
    outside the clip actually being previewed.
    """
    if event_time is None or not (clip_start <= event_time <= clip_end):
        return ""
    return f" ({_format_hms(event_time - clip_start)} into this clip)"


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

    spike_label = _title_label_for_source(clip.source)
    audio_confirm_note = (
        f", audio also peaked here (z={clip.audio_peak_z_score:.2f})"
        f"{_relative_offset_label(clip.audio_peak_time, clip.start_time, clip.end_time)}"
        if clip.audio_peak_z_score is not None else ""
    )
    sound_event_note = (
        ", detected: " + ", ".join(
            f"{cls} ({conf:.2f})"
            for cls, conf in sorted(clip.sound_events.items(), key=lambda kv: kv[1], reverse=True)
        ) + _relative_offset_label(clip.sound_event_time, clip.start_time, clip.end_time)
        if clip.sound_events else ""
    )
    return (
        f"{header}\n\n"
        f"{body}\n\n"
        f"**{_format_hms(clip.start_time)} -> {_format_hms(clip.end_time)}**  "
        f"({clip.duration:.1f}s)  |  Viral score: **{clip.viral_score}/10**\n\n"
        f"{spike_label} at {_format_hms(clip.spike_time)} (z={clip.peak_z_score:.2f})"
        f"{audio_confirm_note}{sound_event_note}"
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


# How close a click needs to land (in seconds of stream time) to count as hitting a
# spike marker rather than an empty stretch of the line - generous enough to forgive
# imprecise clicks on a long timeline, tight enough not to jump to a distant spike.
_HYPE_CLICK_TOLERANCE_S = 15.0


def do_hype_plot_click(
    clips: List[CandidateClip], show_rejected: bool, page: int, source_video_path: str, clicked_x: str,
):
    """
    Handles a hype-plot click forwarded by _HYPE_CLICK_BRIDGE_JS. Finds the clip
    whose spike_time is nearest the clicked x; if it's within tolerance and not
    rejected, jumps to its page and signals which card slot to scroll to and
    highlight. Anywhere else on the graph, or a rejected clip's own spike, is a
    silent no-op - the last output (hype_highlight_signal) is left empty either way.
    """
    no_op = (gr.update(), gr.update(), *([gr.update()] * (MAX_CLIP_CARDS * 8)), "")

    x = None
    try:
        if clicked_x:
            x = float(clicked_x)
    except (TypeError, ValueError):
        x = None

    if x is None or not clips:
        return no_op

    nearest = min(clips, key=lambda c: abs(c.spike_time - x))
    if abs(nearest.spike_time - x) > _HYPE_CLICK_TOLERANCE_S or not nearest.is_clip_worthy:
        return no_op

    visible = _visible_clips(clips, show_rejected)
    target_index = next((i for i, c in enumerate(visible) if c is nearest), None)
    if target_index is None:
        return no_op  # an is_clip_worthy clip is always in `visible`; stay defensive anyway

    target_page = target_index // MAX_CLIP_CARDS
    slot = target_index % MAX_CLIP_CARDS
    return (
        target_page,
        _page_label(visible, target_page),
        *_build_card_updates(visible, target_page, source_video_path),
        str(slot),
    )


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


_AUTOSAVE_PATH = SESSIONS_DIR / "_autosave.json"

# Matches the '{slug}_{YYYYMMDD}_{HHMMSS}' stem save_session() builds for a fresh,
# manually-named file - used only to prettify the dropdown label, never to
# construct or validate an actual path.
_SESSION_FILENAME_RE = re.compile(r"^(?P<slug>.+)_(?P<date>\d{8})_(?P<time>\d{6})$")


def _prettify_session_label(path: Path, max_title_len: int = 42) -> str:
    """Turns an on-disk session filename into a short display label - real
    filenames are built from a VOD's own title and routinely run 60+ characters,
    overflowing the dropdown cell. Only the label changes; the underlying file
    and the dropdown's value (still the real path) are untouched."""
    if path == _AUTOSAVE_PATH:
        return "\U0001F504 Latest run (auto-saved)"
    match = _SESSION_FILENAME_RE.match(path.stem)
    if not match:
        return path.stem
    try:
        stamp = datetime.strptime(f"{match['date']}_{match['time']}", "%Y%m%d_%H%M%S")
    except ValueError:
        return path.stem
    title = match["slug"].replace("_", " ").strip()
    if len(title) > max_title_len:
        title = title[:max_title_len].rstrip() + "…"
    date_label = stamp.strftime("%b %d, %H:%M")
    return f"{date_label} · {title}" if title else date_label


def _session_choices():
    """(label, value) pairs for the sessions dropdown, newest first."""
    return [(_prettify_session_label(p), str(p)) for p in list_sessions()]


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
    """
    Fires automatically off the dropdown's own select event (see build_app's
    wiring), not a dedicated "Load" button - session_path can legitimately be
    empty here (e.g. the filter box gets cleared without picking anything), in
    which case this just does nothing rather than surfacing an error for what
    is normal incidental interaction with the widget.
    """
    if not session_path:
        return gr.skip()
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


def do_reset_delete_arm():
    """
    Resets the "Delete selected session" button back to its neutral label and
    clears the armed-for-delete state the instant the user picks a different
    dropdown entry. do_delete_session's own armed_path != session_path check
    already guarantees a stale confirm can never delete the wrong file, but
    without this the button would keep visually showing "Confirm delete: <old
    file>" after switching selections until the next click quietly re-arms it.
    Wired off the dropdown's select event (real user picks only) rather than
    change, so save/delete/purge silently updating the dropdown's own value
    doesn't re-trigger this.
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


_DELETE_SESSION_LABEL = "Delete current save"


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
    quickly see/hear a candidate beyond its subtitle snippet. Triggered by clicking
    the card's thumbnail, which swaps the static thumbnail out for the video player."""
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
    llm_model: str,
    llm_api_base: str,
    llm_judging_enabled: bool,
    audio_enable: bool,
    audio_z_threshold: float,
    audio_allow_new: bool,
    sound_event_enable: bool,
    sound_event_classes: List[str],
    sound_event_confidence: float,
    sound_event_allow_new: bool,
    autosave_enabled: bool,
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
    if audio_enable and not (source_video_path and Path(source_video_path).exists()):
        raise gr.Error(
            "Audio peak analysis is enabled but the local source video path above doesn't point to an "
            "existing file - download the VOD first, or disable audio peak analysis."
        )
    if sound_event_enable and not (source_video_path and Path(source_video_path).exists()):
        raise gr.Error(
            "Sound event detection is enabled but the local source video path above doesn't point to an "
            "existing file - download the VOD first, or disable sound event detection."
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

    audio_timeline_df = None
    audio_candidates: List[ClipCandidate] = []
    if audio_enable:
        progress(0.65, desc="Analyzing audio peaks...")
        audio_cfg = replace(
            settings.audio, z_score_threshold=audio_z_threshold, allow_new_candidates=audio_allow_new,
        )
        try:
            audio_candidates, audio_timeline_df = AudioAnalyzer(config=audio_cfg).analyze_with_timeline(
                source_video_path
            )
        except AudioAnalysisError as exc:
            raise gr.Error(f"Audio peak analysis failed: {exc}")
        candidates = merge_with_chat_candidates(
            candidates, audio_candidates, audio_allow_new,
            overlap_tolerance_s=settings.audio.cross_modal_overlap_tolerance_s,
        )

    if sound_event_enable:
        progress(0.68, desc="Classifying acoustic events...")
        sound_event_cfg = replace(
            settings.sound_event,
            target_classes=sound_event_classes or settings.sound_event.target_classes,
            confidence_threshold=sound_event_confidence,
            allow_new_candidates=sound_event_allow_new,
        )
        try:
            classifier = SoundEventClassifier(config=sound_event_cfg)
            sound_event_candidates, _sound_event_timeline = classifier.analyze_with_timeline(source_video_path)
        except SoundEventError as exc:
            raise gr.Error(f"Sound event classification failed: {exc}")
        candidates = merge_sound_events(
            candidates, sound_event_candidates, sound_event_allow_new,
            overlap_tolerance_s=settings.sound_event.overlap_tolerance_s,
        )

    if not llm_judging_enabled:
        # Fast path for tuning hype detection or testing the UI without waiting on real LLM
        # calls: build clips straight from the raw statistical candidates, no judgment at all.
        # All default to is_clip_worthy=True since nothing has judged them one way or the
        # other - the operator curates the list by hand with the existing accept/reject tools.
        progress(0.9, desc="Skipping LLM judging - building candidates from raw statistics...")
        refined_clips = [
            build_stat_only_clip(
                c, format_transcript(select_subtitle_window(subtitles, c.window_start, c.window_end)),
                "LLM judging skipped - raw statistical candidate, not reviewed.", used_fallback=False,
            )
            for c in candidates
        ]
    else:
        progress(0.75, desc="Asking the LLM to judge and refine clip boundaries...")
        llm_cfg = replace(
            settings.llm,
            model=llm_model.strip() or None,
            api_base=llm_api_base.strip() or None,
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
    fig = build_hype_timeline_figure(timeline_df, candidates, audio_timeline_df, audio_candidates)
    audio_note = ""
    if audio_enable:
        audio_only_count = sum(1 for c in refined_clips if "audio" in c.source.split("+") and "chat" not in c.source.split("+"))
        confirmed_count = sum(1 for c in refined_clips if "chat" in c.source.split("+") and "audio" in c.source.split("+"))
        audio_note = f" ({audio_only_count} audio-only, {confirmed_count} chat+audio-confirmed)"
    sound_event_note = ""
    if sound_event_enable:
        event_count = sum(1 for c in refined_clips if c.sound_events)
        sound_event_note = f" ({event_count} with a detected acoustic event)"
    status = (
        f"Analyzed {len(messages)} chat messages -> {len(candidates)} candidate(s){audio_note}{sound_event_note} -> "
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

    # Auto-save immediately (unless the operator opted out) - judging a real VOD can take
    # a long time (dozens of real LLM calls), so the result is checkpointed to disk before
    # the operator even starts reviewing it, rather than living only in the browser's
    # in-memory clips_state. Always overwrites one fixed slot (_AUTOSAVE_PATH) instead of
    # creating a fresh timestamped file per run, so repeated analyses don't pile up
    # disposable saves - a deliberate "Save session" click is what creates a real,
    # permanently-named file. session_path_state is reset to None (not the autosave slot)
    # below so a later manual save creates that fresh named file instead of silently
    # overwriting the autosave checkpoint.
    if autosave_enabled:
        try:
            save_session(
                refined_clips, source_video_path, youtube_source, twitch_source, chat_offset,
                session_path=str(_AUTOSAVE_PATH),
            )
            session_update = gr.update(choices=_session_choices())
        except SessionError as exc:
            logger.warning("Auto-save of the finished session failed: %s", exc)
            session_update = gr.update()
    else:
        session_update = gr.update()

    return (
        fig, refined_clips, status, 0, gr.update(value=False), _page_label(visible, 0),
        None, session_update, *page_updates,
    )


# --------------------------------------------------------------------------- #
# LLM provider helpers
# --------------------------------------------------------------------------- #


def do_fetch_models(api_base: str):
    """
    Queries the local Ollama server's /api/tags and returns the pulled model
    names as Dropdown choices, so the Model field can be picked from what's
    actually available instead of typed blind. Pulling new models is on the
    user via `ollama pull <name>` - this app only lists what's already there.
    """
    base = (api_base or "").strip().rstrip("/") or settings.llm.DEFAULT_API_BASE
    try:
        resp = requests.get(f"{base}/api/tags", timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        model_ids = sorted({
            str(entry.get("model") or entry.get("name"))
            for entry in payload.get("models", [])
            if entry.get("model") or entry.get("name")
        })
    except requests.exceptions.RequestException as exc:
        raise gr.Error(f"Failed to fetch models from {base}: {exc}")
    except ValueError as exc:  # response body wasn't valid JSON
        raise gr.Error(f"Ollama returned a non-JSON response from {base}/api/tags: {exc}")

    if not model_ids:
        raise gr.Error(f"No models returned by {base} - pull one first with 'ollama pull <name>'.")

    gr.Info(f"Fetched {len(model_ids)} model(s) from Ollama.")
    return gr.update(choices=model_ids)


# --------------------------------------------------------------------------- #
# VOD video download
# --------------------------------------------------------------------------- #


def do_download_vod(twitch_source: str, quality: str, downloads_dir: str):
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
        video_path = fetch_twitch_vod(
            twitch_source, quality=quality or None,
            downloads_dir=Path(downloads_dir) if downloads_dir and downloads_dir.strip() else None,
        )
    except FetcherError as exc:
        yield f"VOD download failed: {exc}", gr.update(interactive=True), gr.update()
        return

    yield (
        f"VOD downloaded: `{video_path}`. Source video path below has been filled in automatically.",
        gr.update(interactive=True),
        gr.update(value=str(video_path)),
    )


def do_browse_downloads_dir(current_dir: str):
    """
    Opens a native OS folder picker (tkinter, not a Gradio component - browsers have
    no API that hands a web page a real filesystem path, only file *contents*, so this
    has to run in the Python backend, which for a local app like this one shares the
    same machine/filesystem as the person clicking the button). Falls back to leaving
    the field untouched if tkinter can't display anything (e.g. headless server with no
    display) rather than crashing the request - typing the path by hand still works.
    """
    try:
        import tkinter
        from tkinter import filedialog

        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            chosen = filedialog.askdirectory(
                initialdir=current_dir if current_dir and Path(current_dir).is_dir() else str(DOWNLOADS_DIR),
                title="Choose a downloads folder",
            )
        finally:
            root.destroy()
    except Exception as exc:
        logger.warning("Folder picker unavailable (%s); type the path in manually instead.", exc)
        raise gr.Error(f"Couldn't open the folder picker ({exc}). Type the path in manually instead.")

    if not chosen:
        return gr.update()  # user cancelled the dialog
    return gr.update(value=chosen)


def do_open_downloads_folder(downloads_dir: str):
    """Opens the downloads folder in the OS's own file browser (Explorer/Finder/whatever
    the Linux file manager is), creating it first if it doesn't exist yet - e.g. a fresh
    install where no VOD has ever been downloaded there."""
    path = Path(downloads_dir) if downloads_dir and downloads_dir.strip() else DOWNLOADS_DIR
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise gr.Error(f"Could not create/access '{path}': {exc}")

    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(path)  # noqa: S606 - opening a local folder the user themselves chose
        elif system == "Darwin":
            subprocess.run(["open", str(path)], check=True)
        else:
            subprocess.run(["xdg-open", str(path)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise gr.Error(f"Could not open '{path}' in the file browser: {exc}")


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
    # Stay hidden (see fcpxml_file/edl_file's own visible=False) until there's an
    # actual file to show - an empty gr.File renders as a big dashed dropzone-looking
    # box with no indication it's an output, not something to upload into.
    return gr.update(value=str(fcpxml_path), visible=True), gr.update(value=str(edl_path), visible=True)


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

_UI_DIR = Path(__file__).resolve().parent / "ui"

# Gradio's Settings panel includes a "Screen Studio" demo-recording tool that has
# nothing to do with this app - every section in that panel (Theme, Language, PWA,
# Screen Studio, Run History) shares the same generic .banner-wrap class, so there's
# no CSS selector that targets just this one; it has to be found by its heading text
# instead. The Settings modal is only created the first time it's opened, not present
# in the initial page load, so a MutationObserver watches for it rather than trying
# to hide something that doesn't exist yet.
_HIDE_SCREEN_STUDIO_JS = """
() => {
    const hideScreenStudio = () => {
        for (const h of document.querySelectorAll('.banner-wrap h2')) {
            if (h.textContent.includes('Screen Studio')) {
                h.closest('.banner-wrap').style.display = 'none';
            }
        }
    };
    new MutationObserver(hideScreenStudio).observe(document.body, { childList: true, subtree: true });
    hideScreenStudio();
}
"""

# Retriggers the CSS-driven rainbow-spin animation on every click, not just the first -
# a CSS animation already applied to an element won't replay just because its trigger
# class gets re-added on the next click, so this removes the class, forces a reflow
# (reading offsetWidth) to flush that removal before the browser processes the next
# style change, then re-adds it. The spin class/animation live on the button's own
# wrapper (.vb-analyze-frame, a gr.Group()), not the button - the ring is the
# wrapper's ::after showing only in the padding gap around the button, with the
# button's own opaque background naturally covering the rest of it (a real DOM
# ancestor's background always paints under its children, unlike a same-element
# negative-z-index pseudo, which paints ABOVE that same element's own background -
# confirmed via live inspection: with the ring as the button's own ::after, the
# whole button face washed with rainbow instead of just a thin outline). Also
# swaps the button's own label to a random quote for the same duration, restored
# by the same animationend handler that removes the spin class - both are purely
# cosmetic and independent of the real (much longer) analysis run underneath,
# which isn't wired to this button's label at all.
# The button is part of the initial static layout (unlike the Settings modal that
# _HIDE_SCREEN_STUDIO_JS has to wait for), so a plain lookup on load is enough - no
# MutationObserver needed here.
_ANALYZE_BTN_SPIN_JS = """
() => {
    const btn = document.getElementById('analyze_stream_btn');
    if (!btn || btn.dataset.vbSpinBound) return;
    btn.dataset.vbSpinBound = '1';
    const frame = btn.closest('.vb-analyze-frame') || btn;

    // The label is a plain text node sitting between Gradio's own (currently
    // unused) icon-slot comment placeholders - mutating that node directly,
    // instead of btn.textContent, leaves those placeholders in the DOM in
    // case a later Gradio re-render of this button ever expects them there.
    // There's also a lone whitespace text node between two of the comment
    // placeholders (a template artifact) - skip it, or it (not the real
    // label) is what gets replaced, leaving "Analyze Stream" concatenated
    // right after the quote instead of swapped out.
    let labelNode = Array.from(btn.childNodes).find(
        n => n.nodeType === Node.TEXT_NODE && n.nodeValue.trim().length > 0
    );
    if (!labelNode) {
        labelNode = document.createTextNode('');
        btn.appendChild(labelNode);
    }
    const originalLabel = labelNode.nodeValue;

    const quotes = [
        "Henshin-a-go-go, baby!",
        "Heaven or Hell?",
        "Tomorrow is mine!",
        "Rip and tear until it's done",
        "Can't escape from crossing fate!",
        "Hesitation is defeat",
        "The wheel of fate is turning",
        "It's not a lake, it's an ocean",
        "The job… Killer is dead",
        "This is a story of a fighter who wanted to become...",
        "Here comes Daredevil!",
        "Eyes up, guardian",
        "In my restless dream I see this town...",
        "Optimists inbound",
        "Do not fear to commit",
        "Sheeky Breeky i v damki",
        "Humanity restored",
    ];

    btn.addEventListener('click', () => {
        labelNode.nodeValue = quotes[Math.floor(Math.random() * quotes.length)];
        frame.classList.remove('vb-rainbow-spin');
        void frame.offsetWidth;
        frame.classList.add('vb-rainbow-spin');
    });
    frame.addEventListener('animationend', (e) => {
        if (e.animationName === 'vb-rainbow-spin') {
            frame.classList.remove('vb-rainbow-spin');
            labelNode.nodeValue = originalLabel;
        }
    });
}
"""


def _logo_header_html() -> str:
    """
    Renders the top-of-page logo band. Uses logoverysmall.png (300x300) rather
    than the HQ/Small versions in the same folder - at a 72px display height,
    those would just be extra network weight (2MB/385KB vs 101KB) for zero
    visible benefit. Falls back to a text placeholder if the file is missing.
    """
    logo_path = _UI_DIR / "logos" / "logoverysmall.png"
    if logo_path.exists():
        src = str(logo_path.resolve()).replace("\\", "/")
        media = f'<img src="/gradio_api/file={src}" alt="VOD BLADE logo" class="vb-header-logo">'
    else:
        media = '<div class="vb-header-logo vb-header-logo-placeholder">VOD BLADE</div>'
    return f'<div class="vb-header"><div class="vb-header-glow"></div>{media}</div>'


def build_app() -> gr.Blocks:
    with gr.Blocks(title="VOD BLADE") as demo:
        gr.HTML(_logo_header_html())

        with gr.Accordion("Sources & Settings", open=True):
            gr.Markdown("### Sources")
            gr.Markdown("Please provide subtitles source AND Twitch VOD URL of the same stream.")
            with gr.Row():
                with gr.Column():
                    youtube_input = gr.Textbox(
                        label="YouTube URL or local .srt/.vtt/.txt transcript path",
                        placeholder="https://youtube.com/watch?v=... or C:\\path\\to\\transcript.srt",
                    )
                    twitch_input = gr.Textbox(
                        label="Twitch VOD URL or ID",
                        placeholder="https://twitch.tv/videos/123456789",
                    )
                    source_video_input = gr.Textbox(
                        label="Local source video path (Optional. Used by sound analysis and Resolve exports)",
                        placeholder=r"C:\path\to\downloaded_video.mp4",
                    )
                    offset_input = gr.Number(
                        label="Chat offset (s) - Twitch clock minus YouTube clock",
                        value=settings.fetcher.default_chat_offset_seconds,
                    )
                with gr.Column():
                    with gr.Group():
                        audio_enable_checkbox = gr.Checkbox(
                            label="Enable audio peak analysis", value=False,
                            elem_classes=["vb-toggle"],
                        )
                        sound_event_enable_checkbox = gr.Checkbox(
                            label="Enable sound event detection", value=False,
                            elem_classes=["vb-toggle"],
                        )
                        llm_judging_enabled_checkbox = gr.Checkbox(
                            label="Enable AI Arbitration", value=True,
                            elem_classes=["vb-toggle"],
                        )
                    with gr.Group():
                        download_vod_btn = gr.Button("Download VOD")
                        vod_quality_input = gr.Dropdown(
                            label="VOD download quality",
                            choices=[
                                "best", "worst", "audio_only",
                                "1080p60", "720p60", "480p30", "360p30", "160p30",
                            ],
                            value=settings.fetcher.twitch_video_quality,
                            allow_custom_value=True,
                            info="Best is recommended. 2-4GB per hour. Downloads from Twitch",
                        )
                        downloads_dir_input = gr.Textbox(
                            label="Downloads folder",
                            value=str(DOWNLOADS_DIR),
                            placeholder=str(DOWNLOADS_DIR),
                        )
                        with gr.Row():
                            browse_downloads_dir_btn = gr.Button("Browse...", size="sm")
                            open_downloads_folder_btn = gr.Button("Open Download Folder", size="sm")
            download_status = gr.Markdown("")

            clips_state = gr.State([])
            page_state = gr.State(0)
            session_path_state = gr.State(None)
            delete_armed_state = gr.State(None)

            with gr.Row():
                session_dropdown = gr.Dropdown(
                    label="Load session", choices=_session_choices(), value=None,
                    filterable=True, elem_classes=["vb-session-dropdown"],
                )
                with gr.Column():
                    save_session_btn = gr.Button("Save session", size="sm")
                    autosave_enabled_checkbox = gr.Checkbox(
                        label="Auto-save after each run", value=True,
                        elem_classes=["vb-toggle"],
                    )
            with gr.Accordion("Delete / purge saves (advanced)", open=False):
                with gr.Row():
                    delete_session_btn = gr.Button(_DELETE_SESSION_LABEL, size="sm")
                    confirm_purge_checkbox = gr.Checkbox(
                        label="Confirm purge (deletes ALL saved sessions permanently)", value=False,
                    )
                    purge_sessions_btn = gr.Button("Purge saves", variant="stop", size="sm")

            with gr.Accordion("Ollama settings", open=False):
                with gr.Row():
                    llm_model_input = gr.Dropdown(
                        label="Model (optional - blank uses the default model; type to search, "
                              "or click 'Fetch models' for what you've pulled)",
                        choices=[settings.llm.model] if settings.llm.model else [],
                        value=settings.llm.model or "",
                        allow_custom_value=True,
                        filterable=True,
                    )
                    fetch_models_btn = gr.Button("Fetch models from Ollama", size="sm")
                llm_api_base_input = gr.Textbox(
                    label="Ollama server URL (optional - blank uses the local default)",
                    value=settings.llm.api_base or "",
                    placeholder=f"e.g. {settings.llm.DEFAULT_API_BASE}",
                )
                gr.Markdown(
                    "_**Close other GPU-heavy applications for best performance.** VOD BLADE "
                    "can detect the model being evicted to CPU/RAM, but not GPU compute "
                    "contention - another GPU-heavy app (image/video generation, games) "
                    "running at the same time can make judgment run dramatically slower. New "
                    "models aren't managed here - pull them yourself via `ollama pull <name>`._"
                )

            with gr.Accordion("Chat spikes detection settings", open=False):
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
            with gr.Accordion("Audio analysis settings", open=False):
                with gr.Row(elem_classes=["vb-two-col"]):
                    with gr.Column():
                        gr.Markdown(
                            "**Audio peak analysis** - detects loud moments (shouts, sudden "
                            "outbursts) from the VOD's own audio track - requires the local "
                            "source video above to already be downloaded. Enabled via the "
                            "'Enable audio peak analysis' switch up in Sources."
                        )
                        audio_z_threshold_input = gr.Slider(
                            label="Audio Z-score threshold (higher = fewer, louder-only peaks)",
                            minimum=1.0, maximum=6.0, step=0.1,
                            value=settings.audio.z_score_threshold,
                        )
                        audio_allow_new_checkbox = gr.Checkbox(
                            label="Allow audio-only peaks (no matching chat spike) to become their own candidates",
                            value=settings.audio.allow_new_candidates,
                        )
                    with gr.Column():
                        _sound_event_problems = settings.sound_event.validate()
                        gr.Markdown(
                            "**Sound event detection** - detects specific acoustic events "
                            "(laughter, screaming, cheering, groaning) via a YAMNet model - "
                            "also requires the local source video above. Enabled via the "
                            "'Enable sound event detection' switch up in Sources.\n\n"
                            + (
                                f"_Model not ready: {' '.join(_sound_event_problems)}_"
                                if _sound_event_problems else "_YAMNet model found and ready._"
                            )
                        )
                        sound_event_classes_input = gr.CheckboxGroup(
                            label="Event types to detect",
                            choices=settings.sound_event.target_classes,
                            value=settings.sound_event.target_classes,
                        )
                        sound_event_confidence_input = gr.Slider(
                            label="Confidence threshold (higher = fewer, more certain events)",
                            minimum=0.05, maximum=0.95, step=0.05,
                            value=settings.sound_event.confidence_threshold,
                        )
                        sound_event_allow_new_checkbox = gr.Checkbox(
                            label="Allow event-only peaks (no matching chat/audio spike) to become their own candidates",
                            value=settings.sound_event.allow_new_candidates,
                        )
            with gr.Accordion("AI Arbitration settings", open=False):
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
            with gr.Accordion("System prompt [DANGER ZONE]", open=False):
                gr.Markdown(
                    "_This is the full instruction set the LLM arbitrates every candidate clip "
                    "against. Edit it to change how arbitration works. If your edit breaks the "
                    "required JSON output format, calls will fail validation and those "
                    "candidates fall back to their raw chat-spike window instead of crashing._"
                )
                system_prompt_input = gr.Textbox(
                    value=DEFAULT_SYSTEM_PROMPT,
                    lines=20,
                    max_lines=60,
                    show_label=False,
                )
                reset_system_prompt_btn = gr.Button("Reset to default", size="sm")
            with gr.Group(elem_classes=["vb-analyze-frame"]):
                run_btn = gr.Button("Analyze Stream", variant="primary", elem_id="analyze_stream_btn")
            status_box = gr.Markdown("")

        gr.Markdown(
            "### Cool Graph <small style=\"font-weight: normal;\">you can click on it</small>",
            elem_classes=["vb-section-heading"],
        )
        hype_plot = gr.Plot(elem_id="hype_plot")
        # Bridge components for _HYPE_CLICK_BRIDGE_JS / _HYPE_HIGHLIGHT_SCROLL_JS - visible="hidden"
        # (not visible=False) so they stay mounted in the DOM for the JS to reach.
        hype_click_bridge = gr.Textbox(elem_id="hype_click_bridge", visible="hidden")
        hype_click_bridge_button = gr.Button(elem_id="hype_click_bridge_button", visible="hidden")
        hype_highlight_signal = gr.Textbox(elem_id="hype_highlight_signal", visible="hidden")

        gr.Markdown("### Clip Candidates", elem_classes=["vb-section-heading"])

        with gr.Row():
            show_rejected_checkbox = gr.Checkbox(
                label="Show rejected candidates",
                value=False,
            )
            unreject_all_btn = gr.Button("Un-reject all (manual only)", size="sm")
        with gr.Row(elem_classes=["vb-pagination-row"]):
            prev_page_btn = gr.Button("< Prev", size="lg")
            page_label = gr.Markdown("No clips yet - run an analysis.", elem_classes=["vb-page-label"])
            next_page_btn = gr.Button("Next >", size="lg")

        card_components = []
        with gr.Row(elem_classes=["candidate-grid"]):
            for slot_idx in range(MAX_CLIP_CARDS):
                with gr.Group(
                    visible=False, elem_classes=["candidate-card"], elem_id=f"candidate-card-slot-{slot_idx}",
                ) as card_group:
                    # Thumbnail and video share the same slot: the thumbnail is a cheap
                    # auto-generated frame shown by default; clicking it extracts and swaps
                    # in the real playable clip in its place.
                    card_thumbnail = gr.Image(visible=False, interactive=False, show_label=False, height=240)
                    preview_video = gr.Video(visible=False, height=240)
                    card_md = gr.Markdown()
                    card_transcript = gr.Textbox(
                        lines=_CARD_TRANSCRIPT_LINES, max_lines=_CARD_TRANSCRIPT_LINES,
                        interactive=False, show_label=False,
                    )
                    with gr.Accordion("Adjust boundaries", open=False):
                        with gr.Row():
                            start_slider = gr.Slider(label="Start (seconds into VOD)", minimum=0, maximum=1, step=0.1)
                            end_slider = gr.Slider(label="End (seconds into VOD)", minimum=0, maximum=1, step=0.1)
                    toggle_btn = gr.Button(_TOGGLE_LABEL_ACCEPTED, size="sm")
                card_components.append({
                    "group": card_group, "md": card_md, "transcript": card_transcript, "thumbnail": card_thumbnail,
                    "start": start_slider, "end": end_slider,
                    "video": preview_video, "toggle_btn": toggle_btn,
                })

        gr.Markdown("### Export")
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
            fcpxml_file = gr.File(label="FCPXML", visible=False)
            edl_file = gr.File(label="EDL", visible=False)
        inject_status = gr.Markdown("")

        # --- wiring ---

        card_outputs = []
        for c in card_components:
            card_outputs.extend([
                c["group"], c["md"], c["transcript"], c["thumbnail"],
                c["start"], c["end"], c["video"], c["toggle_btn"],
            ])

        # api_visibility="private" on every real event below: this app runs ffmpeg on
        # caller-supplied paths, reads/writes/deletes real files, talks to a local
        # DaVinci Resolve instance, and spends LLM API credits from keys typed into the
        # UI - none of that should be callable via Gradio's auto-generated REST API,
        # only through the UI itself. Hiding the "Use via API" footer link (see
        # launch(footer_links=...)) only hides the docs page; it doesn't touch the
        # actual endpoints, which stay live regardless - this is the real lockdown.
        # Not needed on the fn=None bindings above/below (JS-only, no Python function
        # to call) - Gradio forces api_visibility to "private" for those automatically.
        reset_system_prompt_btn.click(
            fn=lambda: DEFAULT_SYSTEM_PROMPT,
            inputs=[],
            outputs=[system_prompt_input],
            api_visibility="private",
        )

        run_btn.click(
            fn=run_pipeline,
            inputs=[
                youtube_input, twitch_input, offset_input, source_video_input,
                z_threshold_input, min_gap_input, pre_spike_input, post_spike_input,
                max_merged_duration_input,
                min_viral_score_input, content_hint_input, system_prompt_input,
                llm_model_input, llm_api_base_input,
                llm_judging_enabled_checkbox,
                audio_enable_checkbox, audio_z_threshold_input, audio_allow_new_checkbox,
                sound_event_enable_checkbox, sound_event_classes_input,
                sound_event_confidence_input, sound_event_allow_new_checkbox,
                autosave_enabled_checkbox,
            ],
            outputs=[
                hype_plot, clips_state, status_box, page_state,
                show_rejected_checkbox, page_label,
                session_path_state, session_dropdown, *card_outputs,
            ],
            api_visibility="private",
        )

        download_vod_btn.click(
            fn=do_download_vod,
            inputs=[twitch_input, vod_quality_input, downloads_dir_input],
            outputs=[download_status, download_vod_btn, source_video_input],
            api_visibility="private",
        )
        browse_downloads_dir_btn.click(
            fn=do_browse_downloads_dir,
            inputs=[downloads_dir_input],
            outputs=[downloads_dir_input],
            api_visibility="private",
        )
        open_downloads_folder_btn.click(
            fn=do_open_downloads_folder,
            inputs=[downloads_dir_input],
            outputs=[],
            api_visibility="private",
        )

        fetch_models_btn.click(
            fn=do_fetch_models,
            inputs=[llm_api_base_input],
            outputs=[llm_model_input],
            api_visibility="private",
        )

        demo.load(fn=None, inputs=None, outputs=None, js=_HIDE_SCREEN_STUDIO_JS)
        demo.load(fn=None, inputs=None, outputs=None, js=_ANALYZE_BTN_SPIN_JS)
        hype_plot.change(fn=None, inputs=None, outputs=None, js=_HYPE_CLICK_BRIDGE_JS)
        hype_click_bridge_button.click(
            fn=do_hype_plot_click,
            inputs=[clips_state, show_rejected_checkbox, page_state, source_video_input, hype_click_bridge],
            outputs=[page_state, page_label, *card_outputs, hype_highlight_signal],
            api_visibility="private",
        ).then(fn=None, inputs=None, outputs=None, js=_HYPE_HIGHLIGHT_SCROLL_JS)

        prev_page_btn.click(
            fn=partial(go_to_page, delta=-1),
            inputs=[clips_state, show_rejected_checkbox, page_state, source_video_input],
            outputs=[page_state, page_label, *card_outputs],
            api_visibility="private",
        )
        next_page_btn.click(
            fn=partial(go_to_page, delta=1),
            inputs=[clips_state, show_rejected_checkbox, page_state, source_video_input],
            outputs=[page_state, page_label, *card_outputs],
            api_visibility="private",
        )
        show_rejected_checkbox.change(
            fn=partial(go_to_page, delta=0),
            inputs=[clips_state, show_rejected_checkbox, page_state, source_video_input],
            outputs=[page_state, page_label, *card_outputs],
            api_visibility="private",
        )
        unreject_all_btn.click(
            fn=do_unreject_all_manual,
            inputs=[clips_state, show_rejected_checkbox, page_state, source_video_input],
            outputs=[clips_state, page_state, page_label, *card_outputs],
            api_visibility="private",
        )

        save_session_btn.click(
            fn=do_save_session,
            inputs=[clips_state, source_video_input, youtube_input, twitch_input, offset_input, session_path_state],
            outputs=[session_path_state, session_dropdown],
            api_visibility="private",
        )
        purge_sessions_btn.click(
            fn=do_purge_sessions,
            inputs=[confirm_purge_checkbox],
            outputs=[confirm_purge_checkbox, session_path_state, session_dropdown],
            api_visibility="private",
        )
        delete_session_btn.click(
            fn=do_delete_session,
            inputs=[session_dropdown, delete_armed_state, session_path_state],
            outputs=[delete_session_btn, delete_armed_state, session_dropdown, session_path_state],
            api_visibility="private",
        )
        # .select (fires only on a real user pick) rather than .change (fires on ANY
        # value change, including save/delete/purge above programmatically updating
        # this same dropdown) - otherwise saving or deleting a session would
        # immediately re-trigger a reload of whatever the dropdown's new value
        # happens to be, clobbering the operator's current in-progress review state.
        session_dropdown.select(
            fn=do_reset_delete_arm,
            inputs=[],
            outputs=[delete_session_btn, delete_armed_state],
            api_visibility="private",
        ).then(
            fn=do_load_session,
            inputs=[session_dropdown],
            outputs=[
                clips_state, session_path_state, source_video_input, youtube_input, twitch_input, offset_input,
                show_rejected_checkbox, page_state, page_label, *card_outputs,
            ],
            api_visibility="private",
        )

        for idx, c in enumerate(card_components):
            c["start"].release(
                fn=partial(_sync_bound, idx=idx, field_name="start_time"),
                inputs=[clips_state, show_rejected_checkbox, page_state, c["start"]],
                outputs=[clips_state],
                api_visibility="private",
            )
            c["end"].release(
                fn=partial(_sync_bound, idx=idx, field_name="end_time"),
                inputs=[clips_state, show_rejected_checkbox, page_state, c["end"]],
                outputs=[clips_state],
                api_visibility="private",
            )
            c["thumbnail"].select(
                fn=do_preview_clip,
                inputs=[source_video_input, c["start"], c["end"]],
                outputs=[c["video"], c["thumbnail"]],
                api_visibility="private",
            )
            c["toggle_btn"].click(
                fn=partial(do_toggle_worthy, idx=idx),
                inputs=[clips_state, show_rejected_checkbox, page_state, source_video_input],
                outputs=[clips_state, page_state, page_label, *card_outputs],
                api_visibility="private",
            )

        export_files_btn.click(
            fn=do_export_files,
            inputs=[clips_state, source_video_input],
            outputs=[fcpxml_file, edl_file],
            api_visibility="private",
        )
        inject_btn.click(
            fn=do_inject_resolve,
            inputs=[clips_state, source_video_input],
            outputs=[inject_status],
            api_visibility="private",
        )

    return demo


def _load_custom_css() -> str:
    """
    custom.css references its background SVG through Gradio's /gradio_api/file=
    endpoint, which needs a real absolute filesystem path - substituted here at
    startup instead of hardcoded in the file, so the project still works after
    being moved or cloned to a different machine/drive/path.
    """
    css = (_UI_DIR / "custom.css").read_text(encoding="utf-8")
    heart_svg_path = str((_UI_DIR / "heart.svg").resolve()).replace("\\", "/")
    return css.replace("{{HEART_SVG_PATH}}", heart_svg_path)


# Loaded via launch(head=...) rather than folded into ui/custom.css's css=... string -
# two separate things need this:
#  - The Google Fonts <link>s: an @import only works if it's literally the first rule in
#    its own stylesheet, fragile to guarantee once that file's content gets composed with
#    other CSS. A real <link> in <head> has no such ordering requirement.
#  - The @property rule: confirmed via live inspection that Gradio's css=... pipeline
#    silently strips @property at-rules (they never make it into document.styleSheets),
#    but the exact same rule survives untouched when it arrives via head=... instead,
#    since that content is injected as literal HTML rather than being run through
#    whatever rewrites/scopes the css=... string. --vb-spin-angle is what lets the
#    "Analyze Stream" button's rainbow ring (ui/custom.css) actually rotate a
#    conic-gradient's angle instead of sliding a linear one sideways - see that file's
#    #analyze_stream_btn::after for why a real transform: rotate() isn't used instead.
_CUSTOM_HEAD_TAGS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Russo+One&display=swap" rel="stylesheet">
<style>@property --vb-spin-angle { syntax: "<angle>"; inherits: false; initial-value: 0deg; }</style>
"""

if __name__ == "__main__":
    app = build_app()
    app.queue().launch(
        server_port=7863,
        css_paths=[_UI_DIR / "theme.css"],
        css=_load_custom_css(),
        head=_CUSTOM_HEAD_TAGS,
        allowed_paths=[str(_UI_DIR)],
        favicon_path=str(_UI_DIR / "logos" / "logoverysmall.png"),
        footer_links=["gradio", "settings"],
    )
