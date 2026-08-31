"""
Tests for the LLM enrichment fields (category / topic / sentiment).

The dominant risk here is not "does the tag appear" - it's that adding fields to a
schema the LLM must produce could break clip generation entirely for anyone running
a custom system prompt (the UI exposes one) or a small local model that drops fields.
So most of these assert on graceful degradation rather than on the happy path.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from core.llm_agent import CLIP_CATEGORIES, CLIP_SENTIMENTS, CandidateClip, ClipSuggestion


BASE = dict(
    is_clip_worthy=True, start_time=1.0, end_time=5.0,
    title="A title", viral_score=7, summary="A summary",
)


# --------------------------------------------------------------------------- #
# Degradation - the part that must not regress
# --------------------------------------------------------------------------- #


def test_response_without_enrichment_is_still_valid():
    """A custom or older system prompt omits these fields entirely. That must cost a
    tag, never the clip."""
    s = ClipSuggestion(**BASE)
    assert s.category == "other"
    assert s.topic == ""
    assert s.sentiment == "neutral"


def test_unknown_category_falls_back_rather_than_raising():
    assert ClipSuggestion(**BASE, category="comedy").category == "other"
    assert ClipSuggestion(**BASE, category="").category == "other"


def test_unknown_sentiment_falls_back_rather_than_raising():
    assert ClipSuggestion(**BASE, sentiment="furious").sentiment == "neutral"
    assert ClipSuggestion(**BASE, sentiment="").sentiment == "neutral"


@pytest.mark.parametrize("raw,expected", [
    ("Funny", "funny"), ("FUNNY", "funny"), ("  skilled  ", "skilled"), ("Meta", "meta"),
])
def test_category_casing_and_whitespace_normalised(raw, expected):
    """A model answering 'Funny' is being useful but sloppy - coerce, don't reject."""
    assert ClipSuggestion(**BASE, category=raw).category == expected


@pytest.mark.parametrize("raw,expected", [("Positive", "positive"), (" MIXED ", "mixed")])
def test_sentiment_casing_normalised(raw, expected):
    assert ClipSuggestion(**BASE, sentiment=raw).sentiment == expected


def test_every_declared_category_is_accepted():
    for cat in CLIP_CATEGORIES:
        assert ClipSuggestion(**BASE, category=cat).category == cat


def test_every_declared_sentiment_is_accepted():
    for sen in CLIP_SENTIMENTS:
        assert ClipSuggestion(**BASE, sentiment=sen).sentiment == sen


def test_topic_is_free_text_and_preserved():
    """Unlike category/sentiment, topic is descriptive and may be non-English."""
    assert ClipSuggestion(**BASE, topic="  Elden Ring  ").topic == "Elden Ring"
    assert ClipSuggestion(**BASE, topic="кот сломал стрим").topic == "кот сломал стрим"


# --------------------------------------------------------------------------- #
# Session round-trip - old saved sessions must keep loading
# --------------------------------------------------------------------------- #


OLD_CLIP_PAYLOAD = {
    "start_time": 10.0, "end_time": 40.0, "title": "Old clip", "viral_score": 8,
    "summary": "saved before enrichment existed", "spike_time": 20.0,
    "peak_hype_score": 5.0, "peak_z_score": 3.0, "transcript_excerpt": "text",
    "used_fallback": False, "is_clip_worthy": True, "rejection_reason": None,
    "source": "chat", "audio_peak_z_score": None, "audio_peak_time": None,
    "sound_events": {}, "sound_event_time": None,
}


def test_session_saved_before_enrichment_still_loads():
    """core.session_store rebuilds clips with CandidateClip(**payload), so a session
    file written before these fields existed must not blow up on load."""
    clip = CandidateClip(**OLD_CLIP_PAYLOAD)
    assert clip.category == "other"
    assert clip.topic == ""
    assert clip.sentiment == "neutral"


def test_enriched_clip_round_trips_through_a_session():
    original = CandidateClip(**OLD_CLIP_PAYLOAD, category="funny", topic="кот", sentiment="positive")
    restored = CandidateClip(**asdict(original))
    assert (restored.category, restored.topic, restored.sentiment) == ("funny", "кот", "positive")


# --------------------------------------------------------------------------- #
# Card rendering
# --------------------------------------------------------------------------- #


def _clip(**kw):
    payload = dict(OLD_CLIP_PAYLOAD)
    payload.update(kw)
    return CandidateClip(**payload)


def test_unenriched_card_shows_no_tag_suffix():
    """Older clips must render exactly as they did before this feature."""
    import app
    assert app._format_enrichment_tags(_clip()) == ""


def test_neutral_sentiment_is_not_shown():
    """'neutral' is also the un-enriched default, so showing it would put a
    meaningless chip on every old clip."""
    import app
    assert "neutral" not in app._format_enrichment_tags(_clip(category="skilled", sentiment="neutral"))


def test_other_category_is_not_shown():
    import app
    assert app._format_enrichment_tags(_clip(category="other", topic="")) == ""


def test_full_enrichment_renders_all_three():
    import app
    out = app._format_enrichment_tags(_clip(category="dramatic", topic="Elden Ring", sentiment="mixed"))
    assert "dramatic" in out and "Elden Ring" in out and "mixed" in out
