"""
Tests for the export layer - the FCPXML/EDL helpers and the DaVinci Resolve
injection logic.

The Resolve tests use hand-rolled fakes rather than mocks: the real API hands
back live COM-ish objects from a running Resolve instance, which CI (and any
machine without Resolve open) can't provide. The fakes model only the three
behaviours the exporter actually depends on - CreateEmptyTimeline returning
None on a name collision, GetClipList/GetSubFolderList walking a folder tree,
and GetClipProperty returning a file path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from exporters.davinci_api import DavinciAPIError, DavinciResolveExporter
from exporters.xml_exporter import (
    _dedupe_name, _drop_frame_timecode, _frame_duration_fraction, _is_ntsc_rate,
    _safe_name, _seconds_to_timecode,
)


# --------------------------------------------------------------------------- #
# Fakes for the Resolve API
# --------------------------------------------------------------------------- #


class FakeMediaPool:
    """Rejects CreateEmptyTimeline for names already taken, exactly as Resolve does."""

    def __init__(self, existing_names=()):
        self.existing = set(existing_names)
        self.attempts = []

    def CreateEmptyTimeline(self, name):
        self.attempts.append(name)
        if name in self.existing:
            return None
        self.existing.add(name)
        return f"timeline:{name}"


class FakeClip:
    def __init__(self, path):
        self._path = path

    def GetClipProperty(self, key):
        assert key == "File Path"
        return self._path


class FakeFolder:
    def __init__(self, clips=(), subfolders=()):
        self._clips = list(clips)
        self._subfolders = list(subfolders)

    def GetClipList(self):
        return self._clips

    def GetSubFolderList(self):
        return self._subfolders


# --------------------------------------------------------------------------- #
# Timeline name collision (the "second export always failed" bug)
# --------------------------------------------------------------------------- #


def test_timeline_uses_base_name_when_free():
    pool = FakeMediaPool()
    name, timeline = DavinciResolveExporter()._create_unique_timeline(pool, "VOD BLADE Clips")
    assert name == "VOD BLADE Clips"
    assert timeline is not None


def test_timeline_increments_past_existing_name():
    """The regression that mattered: exporting twice into one project."""
    pool = FakeMediaPool(existing_names=["VOD BLADE Clips"])
    name, timeline = DavinciResolveExporter()._create_unique_timeline(pool, "VOD BLADE Clips")
    assert name == "VOD BLADE Clips 2"
    assert timeline is not None


def test_timeline_increments_past_a_run_of_existing_names():
    pool = FakeMediaPool(existing_names=[
        "VOD BLADE Clips", "VOD BLADE Clips 2", "VOD BLADE Clips 3",
    ])
    name, _ = DavinciResolveExporter()._create_unique_timeline(pool, "VOD BLADE Clips")
    assert name == "VOD BLADE Clips 4"


def test_timeline_creation_is_bounded_not_infinite():
    """A pool that rejects everything must raise, never hang the UI thread."""

    class AlwaysRejects:
        def CreateEmptyTimeline(self, name):
            return None

    with pytest.raises(DavinciAPIError, match="Could not create a timeline"):
        DavinciResolveExporter()._create_unique_timeline(AlwaysRejects(), "X")


# --------------------------------------------------------------------------- #
# Media pool search (the "re-imports a duplicate multi-GB VOD" bug)
# --------------------------------------------------------------------------- #


def test_finds_clip_in_root_folder():
    target = Path("E:/vods/stream.mp4")
    root = FakeFolder(clips=[FakeClip(str(target))])
    assert DavinciResolveExporter()._search_folder_tree(root, target) is not None


def test_finds_clip_nested_in_a_bin():
    """Editors organise into bins; a root-only scan missed these."""
    target = Path("E:/vods/stream.mp4")
    deep = FakeFolder(clips=[FakeClip(str(target))])
    root = FakeFolder(clips=[], subfolders=[FakeFolder(subfolders=[deep])])
    assert DavinciResolveExporter()._search_folder_tree(root, target) is not None


def test_returns_none_when_clip_absent():
    root = FakeFolder(clips=[FakeClip("E:/vods/other.mp4")])
    assert DavinciResolveExporter()._search_folder_tree(root, Path("E:/vods/stream.mp4")) is None


def test_search_survives_unreadable_clip_property():
    """One bad pool item must not abort the whole search."""

    class Exploding:
        def GetClipProperty(self, key):
            raise RuntimeError("Resolve returned garbage")

    target = Path("E:/vods/stream.mp4")
    root = FakeFolder(clips=[Exploding(), FakeClip(str(target))])
    assert DavinciResolveExporter()._search_folder_tree(root, target) is not None


def test_search_is_depth_bounded():
    """A pathological/cyclic tree must terminate rather than recurse forever."""
    leaf = FakeFolder(clips=[FakeClip("E:/vods/stream.mp4")])
    node = leaf
    for _ in range(40):
        node = FakeFolder(subfolders=[node])
    assert DavinciResolveExporter()._search_folder_tree(node, Path("E:/vods/stream.mp4")) is None


# --------------------------------------------------------------------------- #
# Timecode / naming helpers
# --------------------------------------------------------------------------- #


def test_ntsc_rate_detection():
    assert _is_ntsc_rate(29.97)
    assert _is_ntsc_rate(23.976)
    assert not _is_ntsc_rate(30.0)
    assert not _is_ntsc_rate(60)


def test_non_drop_timecode_uses_colon_separator():
    assert _seconds_to_timecode(0, 30) == "00:00:00:00"
    assert _seconds_to_timecode(1, 30) == "00:00:01:00"
    assert _seconds_to_timecode(3661, 30) == "01:01:01:00"


def test_drop_frame_timecode_uses_semicolon_separator():
    """Semicolon is the SMPTE convention that marks drop-frame."""
    assert ";" in _seconds_to_timecode(60, 29.97)


def test_drop_frame_timecode_start_of_timeline():
    assert _drop_frame_timecode(0, 29.97) == "00:00:00;00"


def test_frame_duration_matches_known_rates():
    assert _frame_duration_fraction(30).numerator == 1
    # NTSC 29.97 is exactly 1001/30000, not 1/30 - getting this wrong drifts
    # about 3.6 seconds per hour.
    frac = _frame_duration_fraction(29.97)
    assert (frac.numerator, frac.denominator) == (1001, 30000)


def test_safe_name_strips_illegal_characters():
    assert _safe_name('clip: "wow"/<x>', "fallback") == "clip wow x"


def test_safe_name_falls_back_when_nothing_survives():
    assert _safe_name("???", "Clip_1") == "Clip_1"
    assert _safe_name("", "Clip_1") == "Clip_1"


def test_safe_name_truncates():
    assert len(_safe_name("a" * 500, "f")) == 80


def test_dedupe_name_appends_increasing_suffixes():
    used = set()
    assert _dedupe_name("clip", used) == "clip"
    assert _dedupe_name("clip", used) == "clip_2"
    assert _dedupe_name("clip", used) == "clip_3"
