"""
Tests for the in-memory waveform cache in core.audio_analyzer.

The risk this cache carries is memory, not correctness: a long VOD's decoded
waveform is over a gigabyte, so these tests care as much about it being
*released* and *bounded* as about it being hit.

ffmpeg is stubbed out - decoding a real multi-hour VOD in a unit test would be
absurd, and what's under test is the caching logic, not ffmpeg.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import audio_analyzer
from core.audio_analyzer import (
    AudioAnalysisError, extract_pcm_waveform, release_cached_waveform,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    """No test may leak cache state into the next one."""
    release_cached_waveform()
    yield
    release_cached_waveform()


class FakeCompleted:
    def __init__(self, payload):
        self.returncode = 0
        self.stdout = payload
        self.stderr = b""


@pytest.fixture
def counting_ffmpeg(monkeypatch):
    """Replaces the ffmpeg subprocess with a counter, so tests can assert how many
    real decodes would have happened."""
    calls = {"n": 0}
    payload = np.arange(64, dtype=np.float32).tobytes()

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        return FakeCompleted(payload)

    monkeypatch.setattr(audio_analyzer.subprocess, "run", fake_run)
    return calls


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "vod.mp4"
    p.write_bytes(b"not really a video, only stat() is read")
    return p


def test_second_call_reuses_the_decode(counting_ffmpeg, video):
    """The actual Phase 0 win: audio RMS + YAMNet decode once, not twice."""
    first = extract_pcm_waveform(str(video), sample_rate=16000)
    second = extract_pcm_waveform(str(video), sample_rate=16000)

    assert counting_ffmpeg["n"] == 1, "second call should not have re-run ffmpeg"
    assert np.array_equal(first, second)


def test_release_forces_a_fresh_decode(counting_ffmpeg, video):
    extract_pcm_waveform(str(video), sample_rate=16000)
    release_cached_waveform()
    extract_pcm_waveform(str(video), sample_rate=16000)
    assert counting_ffmpeg["n"] == 2


def test_release_actually_drops_the_reference(counting_ffmpeg, video):
    """Guards the memory-leak risk directly: the array must not stay reachable."""
    extract_pcm_waveform(str(video), sample_rate=16000)
    assert audio_analyzer._waveform_cache_value is not None
    release_cached_waveform()
    assert audio_analyzer._waveform_cache_value is None
    assert audio_analyzer._waveform_cache_key is None


def test_release_is_safe_when_nothing_cached():
    release_cached_waveform()
    release_cached_waveform()  # must not raise


def test_different_sample_rate_is_a_miss(counting_ffmpeg, video):
    """A cache hit across sample rates would return silently wrong-rate audio."""
    extract_pcm_waveform(str(video), sample_rate=16000)
    extract_pcm_waveform(str(video), sample_rate=22050)
    assert counting_ffmpeg["n"] == 2


def test_different_file_is_a_miss(counting_ffmpeg, tmp_path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"aaaa")
    b.write_bytes(b"bbbb")

    extract_pcm_waveform(str(a), sample_rate=16000)
    extract_pcm_waveform(str(b), sample_rate=16000)
    assert counting_ffmpeg["n"] == 2


def test_cache_is_single_entry_not_unbounded(counting_ffmpeg, tmp_path):
    """A second video must EVICT the first, never accumulate - an unbounded memo
    of gigabyte arrays would be a leak, not an optimisation."""
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"aaaa")
    b.write_bytes(b"bbbb")

    extract_pcm_waveform(str(a), sample_rate=16000)
    extract_pcm_waveform(str(b), sample_rate=16000)
    # a was evicted by b, so asking for a again must decode again
    extract_pcm_waveform(str(a), sample_rate=16000)
    assert counting_ffmpeg["n"] == 3


def test_modified_file_is_a_miss(counting_ffmpeg, video):
    """mtime/size are part of the identity, so re-downloading the same path must
    not serve the previous file's audio."""
    extract_pcm_waveform(str(video), sample_rate=16000)
    video.write_bytes(b"different content entirely, different size")
    extract_pcm_waveform(str(video), sample_rate=16000)
    assert counting_ffmpeg["n"] == 2


def test_missing_file_still_raises(counting_ffmpeg, tmp_path):
    with pytest.raises(AudioAnalysisError, match="Source video not found"):
        extract_pcm_waveform(str(tmp_path / "nope.mp4"))


def test_empty_path_raises(counting_ffmpeg):
    with pytest.raises(AudioAnalysisError):
        extract_pcm_waveform("")
