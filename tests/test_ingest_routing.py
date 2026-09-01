"""
Tests for multi-source video ingest routing (F6).

The interesting risk here isn't "does yt-dlp work" - it's routing: sending a Twitch
URL to yt-dlp would lose chat, and mistaking a lookalike domain for a supported host
would send a user's URL to the wrong downloader. Host parsing, not substring matching,
is what makes that safe, so it's tested directly.
"""

from __future__ import annotations

import pytest

from core.fetchers import is_ytdlp_video_source


@pytest.mark.parametrize("url", [
    "https://kick.com/streamer/videos/abc-123",
    "https://www.kick.com/streamer/videos/abc-123",
    "kick.com/streamer/videos/abc-123",          # no scheme
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ?t=90",          # timestamp query
    "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
])
def test_supported_non_twitch_hosts_route_to_ytdlp(url):
    assert is_ytdlp_video_source(url) is True


@pytest.mark.parametrize("url", [
    "https://www.twitch.tv/videos/123456789",
    "https://twitch.tv/videos/123456789",
    "2858066583",                                  # bare Twitch id
])
def test_twitch_never_routes_to_ytdlp(url):
    """Twitch must stay on TwitchDownloaderCLI - it's the only source with chat."""
    assert is_ytdlp_video_source(url) is False


@pytest.mark.parametrize("url", [
    "",
    "   ",
    None,
    r"C:\path\to\local_video.mp4",
    "/home/user/video.mp4",
    "not a url at all",
])
def test_non_urls_and_local_paths_do_not_route_to_ytdlp(url):
    assert is_ytdlp_video_source(url) is False


@pytest.mark.parametrize("url", [
    "https://evil-kick.com.attacker.net/x",   # supported host as a subdomain of another
    "https://notkick.com/x",
    "https://kick.com.evil.net/x",
    "https://youtube.com.phish.io/watch?v=x",
])
def test_lookalike_domains_are_rejected(url):
    """Substring matching would accept every one of these. Hostname parsing must not."""
    assert is_ytdlp_video_source(url) is False


# --------------------------------------------------------------------------- #
# Chat gating - the capability difference these sources carry
# --------------------------------------------------------------------------- #


def test_chat_toggle_unavailable_for_non_twitch_sources():
    """Kick/YouTube have no Twitch chat. The toggle must stay off rather than
    enabling and then failing at analysis time."""
    import app
    handler = app.do_gate_toggle("chat_enable")
    assert handler("https://kick.com/x/videos/y")["interactive"] is False
    assert handler("https://youtu.be/abc")["interactive"] is False


def test_chat_toggle_available_for_twitch():
    import app
    handler = app.do_gate_toggle("chat_enable")
    assert handler("https://twitch.tv/videos/123")["interactive"] is True
    assert handler("2858066583")["interactive"] is True


def test_other_toggles_are_not_host_gated():
    """Only chat is Twitch-only; audio/sound-event gate on the video path instead."""
    import app
    assert app.do_gate_toggle("audio_enable")("C:/x.mp4")["interactive"] is True
    assert app.do_gate_toggle("sound_event_enable")("C:/x.mp4")["interactive"] is True


# --------------------------------------------------------------------------- #
# Probe guards - stubbed, because these must hold regardless of what any live
# URL happens to return on the day the suite runs
# --------------------------------------------------------------------------- #


class _FakeYDL:
    """Stands in for yt_dlp.YoutubeDL as a context manager returning fixed info."""

    def __init__(self, info):
        self._info = info

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        return self._info


def _fetcher_with_info(monkeypatch, info, tmp_path):
    from core import fetchers
    monkeypatch.setattr(fetchers.yt_dlp, "YoutubeDL", _FakeYDL(info))
    return fetchers.GenericVideoFetcher(downloads_dir=tmp_path)


def test_playlist_is_rejected(monkeypatch, tmp_path):
    """Downloading a whole channel from one pasted link would be a wildly
    destructive reading of the request."""
    from core.fetchers import VideoFetchError
    f = _fetcher_with_info(monkeypatch, {"_type": "playlist", "id": "PL123", "entries": [{}, {}]}, tmp_path)
    with pytest.raises(VideoFetchError, match="playlist or channel"):
        f._probe("https://youtube.com/playlist?list=PL123")


def test_missing_id_is_rejected(monkeypatch, tmp_path):
    from core.fetchers import VideoFetchError
    f = _fetcher_with_info(monkeypatch, {"title": "no id here"}, tmp_path)
    with pytest.raises(VideoFetchError, match="Could not identify"):
        f._probe("https://kick.com/x")


def test_single_video_probe_returns_extractor_and_id(monkeypatch, tmp_path):
    f = _fetcher_with_info(monkeypatch, {"id": "abc123", "extractor_key": "Kick"}, tmp_path)
    assert f._probe("https://kick.com/x/videos/abc123") == {"id": "abc123", "extractor_key": "Kick"}


def test_partial_downloads_are_not_mistaken_for_finished_files(tmp_path):
    """yt-dlp leaves .part/.ytdl files behind mid-download; treating one as a
    finished video would hand a truncated file to the whole analysis pipeline."""
    from core.fetchers import GenericVideoFetcher
    (tmp_path / "kick_abc.mp4.part").write_bytes(b"incomplete")
    (tmp_path / "kick_abc.ytdl").write_bytes(b"state")
    assert GenericVideoFetcher(downloads_dir=tmp_path)._find_existing("kick_abc") is None

    (tmp_path / "kick_abc.mp4").write_bytes(b"complete")
    found = GenericVideoFetcher(downloads_dir=tmp_path)._find_existing("kick_abc")
    assert found is not None and found.suffix == ".mp4"
