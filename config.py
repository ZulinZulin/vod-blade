"""
Central configuration for StreamCutter.

All tunables live here so that `core/`, `exporters/`, and `app.py` never
hardcode a path, weight, or model name. Values are sourced from environment
variables (via a local `.env` file, loaded through python-dotenv) with
sensible defaults so the app runs out of the box against a local Ollama
model and produces FCPXML-only exports until the user configures more.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Final, List, Optional

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
CACHE_DIR: Final[Path] = DATA_DIR / "cache"
THUMBNAILS_DIR: Final[Path] = CACHE_DIR / "thumbnails"
AUDIO_RMS_CACHE_DIR: Final[Path] = CACHE_DIR / "audio_rms"
SOUND_EVENT_CACHE_DIR: Final[Path] = CACHE_DIR / "sound_events"
DOWNLOADS_DIR: Final[Path] = DATA_DIR / "downloads"
EXPORTS_DIR: Final[Path] = DATA_DIR / "exports"
SESSIONS_DIR: Final[Path] = DATA_DIR / "sessions"
BIN_DIR: Final[Path] = BASE_DIR / "bin"
MODELS_DIR: Final[Path] = BIN_DIR / "models"

for _dir in (
    DATA_DIR, CACHE_DIR, THUMBNAILS_DIR, AUDIO_RMS_CACHE_DIR, SOUND_EVENT_CACHE_DIR,
    DOWNLOADS_DIR, EXPORTS_DIR, SESSIONS_DIR, BIN_DIR, MODELS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)


def _default_twitch_cli_name() -> str:
    return "TwitchDownloaderCLI.exe" if platform.system() == "Windows" else "TwitchDownloaderCLI"


# --------------------------------------------------------------------------- #
# External binaries
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BinaryConfig:
    """Locations of local CLI binaries StreamCutter shells out to."""

    twitch_downloader_cli: Path = field(
        default_factory=lambda: Path(
            os.getenv("TWITCH_DOWNLOADER_CLI_PATH", str(BIN_DIR / _default_twitch_cli_name()))
        )
    )
    ytdlp_binary: str = os.getenv("YTDLP_BINARY", "yt-dlp")

    def validate(self) -> List[str]:
        """Return a list of human-readable problems, empty if all is well."""
        problems = []
        if not self.twitch_downloader_cli.exists():
            problems.append(
                f"TwitchDownloaderCLI not found at '{self.twitch_downloader_cli}'. "
                "Set TWITCH_DOWNLOADER_CLI_PATH or place the binary in ./bin/."
            )
        return problems


# --------------------------------------------------------------------------- #
# Ingestion (core/fetchers.py)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FetcherConfig:
    """Controls how subtitles and chat logs are pulled and cached."""

    # Tried in order; first language yt-dlp actually has a track for wins (see
    # YouTubeSubtitleFetcher._select_best_subtitle_file). Defaults to preferring the
    # native-language track over YouTube's auto-translated English captions, since
    # translation loses exactly the nuance (jokes, hooks, reactions) this app cares
    # about — "-orig" is yt-dlp's marker for the untranslated source-language auto-caption.
    # Override via SUBTITLE_LANGS="en,ja" (comma-separated) for other source languages.
    subtitle_langs: List[str] = field(
        default_factory=lambda: (
            [s.strip() for s in os.getenv("SUBTITLE_LANGS", "").split(",") if s.strip()]
            or ["ru", "ru-orig", "en", "en-orig"]
        )
    )
    prefer_manual_subtitles: bool = True
    ytdlp_retries: int = 3
    ytdlp_socket_timeout_s: int = 30

    twitch_chat_format: str = "json"  # TwitchDownloaderCLI ChatDownload -O format
    twitch_download_timeout_s: int = 1800

    # Video download (core/fetchers.py TwitchVideoFetcher). "best" is TwitchDownloaderCLI's
    # generic top-quality alias; a specific rendition string (e.g. "720p60") also works if you
    # know it's available for a given VOD. VODs can be multi-GB/multi-hour, hence the long timeout.
    twitch_video_quality: str = os.getenv("TWITCH_VIDEO_QUALITY", "best")
    twitch_video_download_timeout_s: int = int(os.getenv("TWITCH_VIDEO_DOWNLOAD_TIMEOUT_S", "14400"))

    # Positive value means the Twitch VOD clock is `chat_offset_seconds`
    # AHEAD of the YouTube video clock (e.g. stream started, then YouTube
    # upload trimmed N seconds off the front). Applied as:
    #   youtube_time = twitch_time - chat_offset_seconds
    default_chat_offset_seconds: float = 0.0

    cache_enabled: bool = True


# --------------------------------------------------------------------------- #
# Chat analytics (core/chat_analyzer.py)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HypeScoreConfig:
    """Weights and thresholds for the chat hype-score pipeline."""

    bin_seconds: int = 5
    rolling_window_bins: int = 24  # 24 * 5s = 120s rolling baseline
    z_score_threshold: float = 2.5
    min_seconds_between_spikes: int = 45

    pre_spike_seconds: int = 60
    post_spike_seconds: int = 30

    # ChatAnalyzer._merge_overlapping caps how large a chain of nearby-overlapping
    # spikes can be fused into before it stops merging - without this, a wide
    # pre/post_spike_seconds can chain-merge several genuinely distinct moments
    # (e.g. two unrelated topics 10 minutes apart) into one sprawling candidate
    # that no LLM can meaningfully judge as a single self-contained moment.
    max_merged_duration_seconds: float = 300.0

    # Weighted components of the raw hype score per bin.
    weight_message_volume: float = 1.0
    weight_emote_frequency: float = 1.5
    weight_caps_exclaim: float = 0.75
    weight_unique_chatters: float = 0.5

    # Emotes/tokens (case-insensitive) that count toward "hype" emote frequency.
    # Global Twitch emotes work the same regardless of stream language, so they're
    # kept alongside Cyrillic slang common in Russian-speaking chat (кек/рофл/жиза/
    # etc.) — see also chat_analyzer._RU_LAUGHTER_RE for "ахах"/"хахаха"-style
    # laughter, which isn't a fixed token so can't live in this list.
    hype_emotes: List[str] = field(
        default_factory=lambda: [
            "kekw", "lul", "lulw", "pog", "pogchamp", "poggers",
            "wtf", "omegalul", "monkas", "pepega", "ez", "clap",
            "widepeepohappy", "sadge", "5head", "pepehands", "???",
            "lol", "lmao",
            "кек", "лол", "рофл", "жиза", "база", "огонь", "жесть",
            "капец", "имба", "красава", "вау", "ору", "умираю",
        ]
    )

    caps_min_word_len: int = 3  # ignore short all-caps tokens like "NA", "GG" noise floor
    caps_min_ratio: float = 0.6  # fraction of letters uppercase to count a message as "caps"


# --------------------------------------------------------------------------- #
# Audio peak analysis (core/audio_analyzer.py)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AudioScoreConfig:
    """
    Weights and thresholds for the audio RMS-energy pipeline. Deliberately
    mirrors HypeScoreConfig's shape (bin -> rolling mean/std -> Z-score -> spike
    windows) rather than sharing code with it - this stays a fully independent
    detector so chat-only analysis (no video required) is never affected by
    whether audio analysis is enabled.

    sample_rate is fixed at 16000 now (not a lower rate that would be "enough"
    for RMS alone) because Phase 2 (YAMNet sound-event classification, ONNX)
    requires 16kHz mono input - extracting at that rate from the start means
    the same decoded waveform can serve both, instead of a second ffmpeg pass
    once Phase 2 lands.
    """

    sample_rate: int = 16000
    bin_seconds: float = 5.0
    rolling_window_bins: int = 24
    z_score_threshold: float = 3.0
    min_seconds_between_spikes: int = 45
    pre_spike_seconds: int = 60
    post_spike_seconds: int = 30
    max_merged_duration_seconds: float = 300.0

    # Generous on purpose: audio-only decode (no video, mono, 16kHz) runs far
    # faster than realtime, so even an 8+ hour VOD finishes well inside this.
    extraction_timeout_s: int = 1800

    # How close (seconds) a chat and an audio spike's own spike_time need to be to
    # count as "the same real-world moment" during merging (core/audio_analyzer.py's
    # merge_with_chat_candidates). Deliberately NOT based on the two candidates'
    # padded windows (pre_spike_seconds/post_spike_seconds exist to give the LLM
    # transcript context, not to define real-world proximity) - with 60s/30s
    # padding on each side, comparing padded windows would let two editorially
    # distinct spikes up to ~90s apart falsely "confirm" each other.
    cross_modal_overlap_tolerance_s: float = 30.0

    # An audio spike whose window doesn't overlap any chat-detected candidate can
    # either become its own new candidate (True - full Option C) or be dropped
    # entirely once no overlap is found (False - audio only ever enriches chat
    # candidates, never creates its own). Kept as a run_pipeline UI toggle in
    # app.py, not a hardcoded choice, since which behavior is useful depends on
    # how chat-quiet the stream tends to be during its best moments.
    allow_new_candidates: bool = True


# --------------------------------------------------------------------------- #
# Sound event classification (core/sound_event_classifier.py)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SoundEventConfig:
    """
    Thresholds for YAMNet-based acoustic event detection (laughter, screaming,
    cheering, groaning). A genuinely independent third detector, same as
    AudioScoreConfig - chat-only and audio-RMS analysis both keep working
    unaffected if this is disabled or the model file isn't present.

    Detection here is a plain threshold on the model's own per-class confidence,
    NOT a rolling Z-score like HypeScoreConfig/AudioScoreConfig use: a
    classifier's confidence output is already normalized to [0, 1] by the model
    itself, so there's no stream-relative baseline to compute a spike against -
    unlike raw chat volume or audio energy, which are scale-dependent per stream
    and only meaningful relative to their own recent history.

    Model files are NOT auto-downloaded, matching BinaryConfig's own convention
    for TwitchDownloaderCLI - see validate(). sample_rate matches
    AudioScoreConfig's (16000) since both consume the exact same waveform from
    core.audio_analyzer.extract_pcm_waveform; no second decode.
    """

    model_path: Path = field(
        default_factory=lambda: Path(os.getenv("YAMNET_ONNX_PATH", str(MODELS_DIR / "yamnet.onnx")))
    )
    class_map_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("YAMNET_CLASS_MAP_PATH", str(MODELS_DIR / "yamnet_class_map.csv"))
        )
    )
    sample_rate: int = 16000

    # Event names - each is actually a SUM of several related AudioSet classes, not one
    # class thresholded alone (see core.sound_event_classifier._EVENT_CLASS_GROUPS); an
    # entry with no known grouping falls back to just that one class map display_name.
    target_classes: List[str] = field(
        default_factory=lambda: ["Laughter", "Screaming", "Cheering", "Groan"]
    )
    confidence_threshold: float = 0.5
    # Must stay comfortably under one frame's duration (the model's hop is ~0.48s) - a
    # value at or above that would reject even a single maximally-confident frame, since
    # one frame's own run-duration is only ~0.48s. Confirmed against real content where a
    # genuine laugh registered strongly for exactly one frame before dropping away again.
    min_event_duration_s: float = 0.3

    # Inference runs in chunks rather than one call over the whole waveform - confirmed
    # against a real multi-hour VOD that a single call's intermediate conv-layer memory
    # scales with total input length (an 8.5GB allocation for ~5.7 hours of audio), not
    # just the final per-frame output size. 300s keeps each call's footprint small with
    # a reasonable number of calls even for a very long stream.
    chunk_duration_s: float = 300.0

    min_seconds_between_spikes: int = 45
    pre_spike_seconds: int = 60
    post_spike_seconds: int = 30
    max_merged_duration_seconds: float = 300.0

    # See AudioScoreConfig.cross_modal_overlap_tolerance_s - same reasoning, applied
    # when merging sound-event candidates into the chat/audio-enriched list.
    overlap_tolerance_s: float = 30.0
    allow_new_candidates: bool = True

    def validate(self) -> List[str]:
        """Return a list of human-readable problems, empty if all is well."""
        problems = []
        if not self.model_path.exists():
            problems.append(
                f"YAMNet ONNX model not found at '{self.model_path}'. Set YAMNET_ONNX_PATH or place "
                "yamnet.onnx there (see the setup notes for where to get it)."
            )
        if not self.class_map_path.exists():
            problems.append(
                f"YAMNet class map not found at '{self.class_map_path}'. Set YAMNET_CLASS_MAP_PATH or "
                "place yamnet_class_map.csv there."
            )
        return problems


# --------------------------------------------------------------------------- #
# LLM agent (core/llm_agent.py)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LLMConfig:
    """
    litellm-compatible model routing. `provider` selects a preset;
    `model` follows litellm's `<provider>/<model>` convention where needed
    (e.g. "ollama/llama3.1", "deepseek/deepseek-chat", "gpt-4o-mini").

    openrouter and nanogpt are both third-party aggregators that let you pick
    from many underlying models. openrouter has native litellm support
    ("openrouter/<model>"); nanogpt doesn't, so it's routed as a generic
    OpenAI-compatible custom endpoint ("openai/<model>" + a custom api_base) -
    see provider_litellm_prefix below for the actual per-provider mapping.
    """

    provider: str = os.getenv("LLM_PROVIDER", "openai")  # openai | deepseek | ollama | openrouter | nanogpt
    # None (unset) means "use provider_model_defaults[provider]" — see resolve_model().
    model: Optional[str] = os.getenv("LLM_MODEL") or None
    api_base: Optional[str] = os.getenv("LLM_API_BASE") or None
    api_key: Optional[str] = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "800"))
    request_timeout_s: int = int(os.getenv("LLM_TIMEOUT_S", "60"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))

    # Soft bounds the LLM agent clamps clip durations into (seconds).
    min_clip_duration_s: float = float(os.getenv("LLM_MIN_CLIP_DURATION_S", "12"))
    max_clip_duration_s: float = float(os.getenv("LLM_MAX_CLIP_DURATION_S", "90"))

    # A second, independent filter on top of the LLM's own is_clip_worthy veto: even a
    # candidate the model calls "worthy" gets rejected if its own viral_score is below
    # this bar. Default of 1 preserves old behavior (no extra filtering); raise it to
    # actually make use of the confidence signal the model already produces but which
    # nothing previously consumed.
    min_viral_score: int = int(os.getenv("LLM_MIN_VIRAL_SCORE", "1"))

    # Ollama only: minimum fraction of the model that must be VRAM-resident
    # (per Ollama's /api/ps "size_vram / size") before a run is allowed to proceed.
    # When something else on the machine is competing for VRAM, Ollama silently
    # offloads part of the model to CPU/RAM instead of refusing to run - which
    # "works" but can be an order of magnitude slower with no indication why.
    # Checked once per LLMAgent instance; a value <= 0 disables the check entirely.
    min_ollama_gpu_ratio: float = float(os.getenv("LLM_MIN_OLLAMA_GPU_RATIO", "0.95"))

    # Per-provider defaults applied when the user only sets LLM_PROVIDER. Already carry
    # whatever litellm prefix that provider needs (see provider_litellm_prefix), so a
    # blank Model field resolves straight to a working default with no further work.
    provider_model_defaults: Dict[str, str] = field(
        default_factory=lambda: {
            "openai": "gpt-4o-mini",
            "deepseek": "deepseek/deepseek-chat",
            "ollama": "ollama/llama3.1",
            "openrouter": "openrouter/openai/gpt-4o-mini",
            "nanogpt": "openai/zai-org/glm-5.2",
        }
    )
    provider_api_base_defaults: Dict[str, str] = field(
        default_factory=lambda: {
            "ollama": "http://localhost:11434",
            "openai": "https://api.openai.com/v1",
            "deepseek": "https://api.deepseek.com",
            "openrouter": "https://openrouter.ai/api/v1",
            "nanogpt": "https://nano-gpt.com/api/v1",
        }
    )
    # Maps our provider name to the litellm routing prefix its model string needs.
    # Usually identical to the provider name itself ("ollama/", "deepseek/", ...) - the
    # exception is nanogpt, which litellm has no native integration for, so it's routed
    # as a generic OpenAI-compatible custom endpoint ("openai/" + a custom api_base)
    # rather than "nanogpt/", which litellm wouldn't recognize as a provider at all.
    provider_litellm_prefix: Dict[str, str] = field(
        default_factory=lambda: {
            "openai": "",
            "deepseek": "deepseek/",
            "ollama": "ollama/",
            "openrouter": "openrouter/",
            "nanogpt": "openai/",
        }
    )

    def resolve_model(self) -> str:
        """
        The explicit LLM_MODEL if set, else the provider's default model.

        litellm routes requests purely off the model string's prefix (e.g.
        "ollama/qwen2.5:14b-instruct") — without it, a non-OpenAI provider
        silently gets treated as OpenAI and every call fails. Users naturally
        set LLM_MODEL to the bare model ID as the provider itself names it
        (e.g. "zai-org/glm-5.2" as nanogpt lists it), so auto-prepend the
        litellm routing prefix if it's missing rather than requiring them to
        know litellm's convention.
        """
        model = self.model or self.provider_model_defaults.get(self.provider, "gpt-4o-mini")
        prefix = self.provider_litellm_prefix.get(self.provider, f"{self.provider}/")
        if prefix and not model.startswith(prefix):
            model = f"{prefix}{model}"
        return model

    def resolve_api_base(self) -> Optional[str]:
        """The explicit LLM_API_BASE if set, else the provider's default base URL."""
        return self.api_base or self.provider_api_base_defaults.get(self.provider)


# --------------------------------------------------------------------------- #
# Export (exporters/xml_exporter.py, exporters/davinci_api.py)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExportConfig:
    fcpxml_version: str = "1.10"  # widely compatible with Resolve Free & Studio
    # Fallback ONLY - used when the real source file's frame rate can't be read via
    # ffprobe (missing binary, unreadable file, etc). Exporters always try ffprobe
    # first, since assuming a fixed rate silently produces wrong frame math whenever
    # the actual source file differs (e.g. a 30fps VOD download with this at 60).
    default_fps: float = 60.0
    default_timeline_width: int = 1920
    default_timeline_height: int = 1080
    export_dir: Path = EXPORTS_DIR
    ffprobe_binary: str = os.getenv("FFPROBE_BINARY", "ffprobe")
    ffmpeg_binary: str = os.getenv("FFMPEG_BINARY", "ffmpeg")

    # DaVinci Resolve scripting API integration paths (per-OS defaults;
    # override via env if Resolve is installed non-standard).
    resolve_script_api: Optional[str] = os.getenv("RESOLVE_SCRIPT_API")
    resolve_script_lib: Optional[str] = os.getenv("RESOLVE_SCRIPT_LIB")

    def resolved_script_paths(self) -> Dict[str, str]:
        system = platform.system()
        if system == "Windows":
            api = self.resolve_script_api or (
                r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting"
            )
            lib = self.resolve_script_lib or (
                r"C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll"
            )
        elif system == "Darwin":
            api = self.resolve_script_api or (
                "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
            )
            lib = self.resolve_script_lib or (
                "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
            )
        else:
            api = self.resolve_script_api or (
                "/opt/resolve/Developer/Scripting"
            )
            lib = self.resolve_script_lib or (
                "/opt/resolve/libs/Fusion/fusionscript.so"
            )
        return {"api": api, "lib": lib, "modules": str(Path(api) / "Modules")}


# --------------------------------------------------------------------------- #
# Aggregate settings object
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Settings:
    binaries: BinaryConfig = field(default_factory=BinaryConfig)
    fetcher: FetcherConfig = field(default_factory=FetcherConfig)
    hype: HypeScoreConfig = field(default_factory=HypeScoreConfig)
    audio: AudioScoreConfig = field(default_factory=AudioScoreConfig)
    sound_event: SoundEventConfig = field(default_factory=SoundEventConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
