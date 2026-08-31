"""
exporters/davinci_api.py

Optional fallback exporter that talks directly to a running DaVinci Resolve
instance via the DaVinciResolveScript scripting API, bypassing FCPXML
import entirely: it imports the source media into the current project's
Media Pool and appends each refined clip straight onto a new timeline.

This module only works when:
  1. DaVinci Resolve is installed and currently running, AND
  2. Resolve Preferences > General > "External scripting using" is set to
     Local/Network, AND
  3. The Resolve scripting API/lib paths are reachable (see
     config.ExportConfig.resolved_script_paths()).

If any of that isn't true, `is_available()` returns False and callers
should fall back to exporters/xml_exporter.py instead.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import List, Optional

from config import ExportConfig, settings
from core.llm_agent import CandidateClip

logger = logging.getLogger(__name__)


class DavinciAPIError(Exception):
    """Raised for any failure talking to a running DaVinci Resolve instance."""


@dataclass(frozen=True)
class InjectionResult:
    timeline_name: str
    clips_added: int
    clips_failed: int
    project_name: str


def _load_resolve_script_module(export_config: Optional[ExportConfig] = None) -> ModuleType:
    """
    Imports DaVinciResolveScript, adding Resolve's Modules directory to
    sys.path (and its API/lib paths to the environment) first if it isn't
    already importable — mirrors the setup Blackmagic documents for
    external scripts running outside Resolve's own console.
    """
    try:
        import DaVinciResolveScript as dvr_script  # type: ignore
        return dvr_script
    except ImportError:
        pass

    cfg = export_config or settings.export
    paths = cfg.resolved_script_paths()
    modules_dir = paths["modules"]

    os.environ.setdefault("RESOLVE_SCRIPT_API", paths["api"])
    os.environ.setdefault("RESOLVE_SCRIPT_LIB", paths["lib"])
    if modules_dir not in sys.path:
        sys.path.append(modules_dir)

    try:
        import DaVinciResolveScript as dvr_script  # type: ignore
        return dvr_script
    except ImportError as exc:
        raise DavinciAPIError(
            "Could not import DaVinciResolveScript. Ensure DaVinci Resolve is installed and "
            f"its Modules directory is reachable at '{modules_dir}'. If it's installed elsewhere, "
            "set the path in VOD BLADE's 'DaVinci Resolve settings' section (or RESOLVE_SCRIPT_API / "
            "RESOLVE_SCRIPT_LIB in your .env if running from source)."
        ) from exc


def is_available(export_config: Optional[ExportConfig] = None) -> bool:
    """
    True if DaVinci Resolve is currently running and reachable via the
    scripting API. Safe to call speculatively — never raises, only logs.
    """
    try:
        dvr_script = _load_resolve_script_module(export_config)
        resolve = dvr_script.scriptapp("Resolve")
        return resolve is not None
    except Exception as exc:
        # .info(), not .debug(): this is checked on every page load (see the
        # resolve_hint markdown in app.py) so it fires constantly for anyone who
        # doesn't use Resolve - but burying the *reason* at debug level means a
        # real misconfiguration (wrong path, Resolve's "External scripting" pref
        # off, .env not actually being read) is invisible in the log file someone
        # would send us when reporting exactly this kind of "can't detect it" bug.
        logger.info("DaVinci Resolve scripting API not available: %s", exc)
        return False


class DavinciResolveExporter:
    """Injects refined clips directly into a running Resolve project's timeline."""

    def __init__(self, export_config: Optional[ExportConfig] = None):
        self.cfg = export_config or settings.export
        self._resolve = None
        self._project_manager = None
        self._project = None

    # --- connection ------------------------------------------------------------

    def connect(self) -> None:
        dvr_script = _load_resolve_script_module(self.cfg)
        resolve = dvr_script.scriptapp("Resolve")
        if resolve is None:
            raise DavinciAPIError(
                "DaVinciResolveScript imported but scriptapp('Resolve') returned None. "
                "Is DaVinci Resolve running, with Preferences > General > "
                "'External scripting using' set to Local?"
            )
        self._resolve = resolve
        self._project_manager = resolve.GetProjectManager()
        if self._project_manager is None:
            raise DavinciAPIError("Could not get Resolve's Project Manager.")
        self._project = self._project_manager.GetCurrentProject()
        if self._project is None:
            raise DavinciAPIError("No project is currently open in DaVinci Resolve.")

    def _ensure_connected(self) -> None:
        if self._project is None:
            self.connect()

    # --- injection ---------------------------------------------------------------

    def inject_clips(
        self,
        clips: List[CandidateClip],
        source_video_path: str,
        timeline_name: Optional[str] = None,
    ) -> InjectionResult:
        """
        Imports `source_video_path` into the current project's Media Pool
        (reusing an existing pool item if the same file was already
        imported) and appends each clip to a freshly created timeline.
        """
        if not clips:
            raise DavinciAPIError("No clips to inject.")

        self._ensure_connected()
        media_pool = self._project.GetMediaPool()
        if media_pool is None:
            raise DavinciAPIError("Could not get the current project's Media Pool.")

        pool_item = self._import_or_reuse_media(media_pool, source_video_path)
        # AppendToTimeline's startFrame/endFrame are frame numbers in the SOURCE CLIP's
        # own native frame rate, not the project timeline's - using the timeline's rate
        # here silently points every clip at the wrong place whenever they differ (e.g.
        # a 60fps Twitch VOD dropped onto a 24fps project: every frame number would be
        # ~2.5x too small, landing far earlier in the file than intended).
        fps = self._source_clip_frame_rate(pool_item)

        timeline_name, timeline = self._create_unique_timeline(
            media_pool, timeline_name or "VOD BLADE Clips"
        )
        # AppendToTimeline appends to whatever timeline is current; make sure that's ours.
        self._project.SetCurrentTimeline(timeline)

        clips_added = 0
        clips_failed = 0
        for clip in clips:
            clip_info = {
                "mediaPoolItem": pool_item,
                "startFrame": round(clip.start_time * fps),
                "endFrame": round(clip.end_time * fps),
            }
            appended_items = media_pool.AppendToTimeline([clip_info])
            if not appended_items:
                logger.warning(
                    "Resolve rejected clip '%s' (%.1fs-%.1fs).", clip.title, clip.start_time, clip.end_time
                )
                clips_failed += 1
                continue
            clips_added += 1
            for item in appended_items:
                self._label_timeline_item(item, clip)

        return InjectionResult(
            timeline_name=timeline_name,
            clips_added=clips_added,
            clips_failed=clips_failed,
            project_name=self._project.GetName(),
        )

    # --- internals -----------------------------------------------------------

    _MAX_TIMELINE_NAME_ATTEMPTS = 100

    def _create_unique_timeline(self, media_pool, base_name: str):
        """
        Creates a timeline, appending ' 2', ' 3', ... until Resolve accepts the name.

        CreateEmptyTimeline returns None when a timeline of that name already exists
        in the project, so without this the SECOND export into the same project
        always failed - the single most likely thing a real user does after a first
        successful export. Returns (actual_name, timeline).

        Bounded rather than looping forever: if a hundred attempts all fail, the
        cause isn't name collision (more likely no project open, or a permissions
        problem), and an endless loop would hide that.
        """
        for attempt in range(1, self._MAX_TIMELINE_NAME_ATTEMPTS + 1):
            name = base_name if attempt == 1 else f"{base_name} {attempt}"
            timeline = media_pool.CreateEmptyTimeline(name)
            if timeline is not None:
                if attempt > 1:
                    logger.info(
                        "Timeline '%s' already existed; created '%s' instead.", base_name, name
                    )
                return name, timeline
        raise DavinciAPIError(
            f"Could not create a timeline in this project - tried '{base_name}' and "
            f"{self._MAX_TIMELINE_NAME_ATTEMPTS - 1} numbered variants, and Resolve rejected "
            "all of them. Check that a project is open and writable."
        )

    def _import_or_reuse_media(self, media_pool, source_video_path: str):
        source_path = Path(source_video_path)
        if not source_path.exists():
            raise DavinciAPIError(f"Source media not found: {source_path}")

        existing = self._find_pool_item_by_path(media_pool, source_path)
        if existing is not None:
            logger.info("Reusing already-imported Media Pool item for '%s'.", source_path.name)
            return existing

        imported = media_pool.ImportMedia([str(source_path)])
        if not imported:
            raise DavinciAPIError(f"Resolve failed to import media: {source_path}")
        return imported[0]

    @classmethod
    def _find_pool_item_by_path(cls, media_pool, source_path: Path):
        """
        Searches the whole Media Pool folder tree, not just the root folder.

        Professional editors organise media into bins, so a root-only scan missed
        the already-imported clip for exactly the users most likely to have one -
        silently re-importing a duplicate of a multi-GB VOD on every export.
        """
        root_folder = media_pool.GetRootFolder()
        if root_folder is None:
            return None
        return cls._search_folder_tree(root_folder, source_path)

    @classmethod
    def _search_folder_tree(cls, folder, source_path: Path, depth: int = 0):
        # Depth-bounded: Resolve's API hands back live objects, and a malformed or
        # unexpectedly cyclic tree would otherwise recurse forever inside a UI handler.
        if depth > 16:
            logger.warning("Media Pool folder tree deeper than 16 levels; stopping search.")
            return None

        for item in folder.GetClipList() or []:
            try:
                clip_path = Path(item.GetClipProperty("File Path"))
            except Exception:
                continue
            if clip_path == source_path:
                return item

        for subfolder in folder.GetSubFolderList() or []:
            found = cls._search_folder_tree(subfolder, source_path, depth + 1)
            if found is not None:
                return found
        return None

    def _source_clip_frame_rate(self, pool_item) -> float:
        """
        The source media's OWN frame rate - what AppendToTimeline's startFrame/endFrame
        are actually counted in, regardless of what the project's timeline is set to.
        """
        try:
            return float(pool_item.GetClipProperty("FPS"))
        except (TypeError, ValueError, AttributeError):
            logger.warning(
                "Could not read the source clip's frame rate from Resolve; falling back to "
                "config default (%.2f fps). If this is wrong, injected clips will point at "
                "the wrong place in the file.", self.cfg.default_fps,
            )
            return self.cfg.default_fps

    @staticmethod
    def _label_timeline_item(item, clip: CandidateClip) -> None:
        try:
            item.SetName(clip.title[:100])
        except Exception as exc:
            logger.debug("Could not rename timeline item for clip '%s': %s", clip.title, exc)
        try:
            item.SetClipColor(clip.mark_color or ("Orange" if clip.viral_score >= 8 else "Yellow"))
        except Exception:
            pass  # clip coloring is purely cosmetic; never fail the export over it


# --------------------------------------------------------------------------- #
# Convenience module-level API
# --------------------------------------------------------------------------- #


def inject_into_resolve(
    clips: List[CandidateClip],
    source_video_path: str,
    timeline_name: Optional[str] = None,
) -> InjectionResult:
    """Convenience entry point for app.py's 'Inject into DaVinci Resolve' button."""
    return DavinciResolveExporter().inject_clips(clips, source_video_path, timeline_name)
