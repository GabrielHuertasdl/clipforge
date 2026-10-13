"""Configuration loading.

Defaults live here as Pydantic models; ``config.yaml`` deep-merges on top of them, so a
partial YAML file is always valid. Every knob the pipeline uses must be reachable from
``Config`` — nothing is hardcoded in the modules.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_FILENAME = "config.yaml"


class Paths(BaseModel):
    input_raw: Path = Path("input/raw")
    input_manual: Path = Path("input/manual")
    output_pending: Path = Path("output/pending")
    output_approved: Path = Path("output/approved")
    output_rejected: Path = Path("output/rejected")
    output_thumbs: Path = Path("output/thumbs")
    cache_dir: Path = Path("data/cache")
    logs_dir: Path = Path("data/logs")
    db_path: Path = Path("data/clipforge.db")

    def all_dirs(self) -> list[Path]:
        return [
            self.input_raw,
            self.input_manual,
            self.output_pending,
            self.output_approved,
            self.output_rejected,
            self.output_thumbs,
            self.cache_dir,
            self.logs_dir,
            self.db_path.parent,
        ]


class IngestCfg(BaseModel):
    extensions: list[str] = [".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm"]
    hash_chunk_bytes: int = 8 * 1024 * 1024
    hash_full: bool = False
    min_duration_s: float = 600
    title_strip_patterns: list[str] = []
    title_year_pattern: str = r"(19\d{2}|20\d{2})"


class ScenesCfg(BaseModel):
    detector: Literal["content", "adaptive"] = "content"
    threshold: float = 27.0
    min_scene_len_s: float = 0.8
    downscale_factor: int = 4
    segment_min_s: float = 150
    segment_max_s: float = 260
    segment_target_s: float = 200
    black_luma_threshold: float = 18.0
    max_dark_ratio: float = 0.45
    luma_sample_fps: float = 0.5
    skip_head_ratio: float = 0.02
    skip_tail_ratio: float = 0.06


class TranscribeCfg(BaseModel):
    model: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = True
    cache: bool = True


class AudioCfg(BaseModel):
    sample_rate: int = 16000
    frame_s: float = 1.0
    quiet_window_s: float = 1.0
    quiet_rms_percentile: float = 30
    peak_rms_percentile: float = 80
    peak_delta_db: float = 8.0
    peak_min_gap_s: float = 4.0
    silence_min_ms: int = 400
    silence_top_db: float = 35
    onset_backtrack: bool = True


class IconicScenesCfg(BaseModel):
    enabled: bool = True
    max_scenes: int = 12
    priority_window_s: float = 180
    fallback_analyze_all: bool = True


class ScoringWeights(BaseModel):
    hook_strength: float = 0.25
    emotional_peak: float = 0.15
    standalone_clarity: float = 0.20
    quotability: float = 0.10
    visual_interest: float = 0.10
    sustains_3min: float = 0.20

    def normalized(self) -> dict[str, float]:
        raw = self.model_dump()
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}


class ScoringCfg(BaseModel):
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 2000
    concurrency: int = 3
    max_retries: int = 4
    backoff_base_s: float = 2.0
    backoff_max_s: float = 30.0
    transcript_char_budget: int = 6000
    rehooks_min: int = 6
    rehooks_max: int = 8
    iconic_scenes: IconicScenesCfg = Field(default_factory=IconicScenesCfg)
    weights: ScoringWeights = Field(default_factory=ScoringWeights)


class DeadAirCfg(BaseModel):
    enabled: bool = True
    silence_min_ms: int = 400
    keep_breath_ms: int = 150
    max_cut_ratio: float = 0.35
    final_min_s: float = 170
    final_max_s: float = 200


class ColdOpenCfg(BaseModel):
    enabled: bool = True
    duration_s: float = 1.5
    hook_text_font_size: int = 78
    hook_text_max_chars_per_line: int = 22
    flash_cut: bool = True


class SpeakerSwitchCfg(BaseModel):
    enabled: bool = True
    min_hold_s: float = 1.2


class CropCfg(BaseModel):
    mode: Literal["dynamic", "center"] = "dynamic"
    sample_every_n_frames: int = 10
    detector: Literal["haar", "dnn"] = "haar"
    haar_cascade: str = "haarcascade_frontalface_default.xml"
    dnn_proto: str | None = None
    dnn_model: str | None = None
    min_face_size: int = 60
    smoothing_window: int = 25
    max_pan_px_per_s: float = 220
    speaker_switch: SpeakerSwitchCfg = Field(default_factory=SpeakerSwitchCfg)
    fallback: Literal["center"] = "center"


class SubtitlesCfg(BaseModel):
    enabled: bool = True
    max_words_per_line: int = 3
    font: str = "Montserrat ExtraBold"
    font_file: str | None = None
    font_size: int = 68
    primary_color: str = "&H00FFFFFF"
    active_color: str = "&H0000E5FF"
    outline_color: str = "&H00000000"
    outline: int = 4
    shadow: int = 2
    bold: bool = True
    uppercase: bool = True
    safe_top_px: int = 150
    safe_bottom_px: int = 300
    vertical_anchor: float = 0.55


class RehooksCfg(BaseModel):
    enabled: bool = True
    count: int = 7
    alternate: bool = True
    text_duration_s: float = 1.6
    text_font_size: int = 60
    text_slide_in: bool = True
    zoom_punch_pct: float = 4
    zoom_punch_ms: int = 300
    snap_to_audio_peak_s: float = 0.6


class CompressorCfg(BaseModel):
    enabled: bool = True
    aggressiveness: Literal["low", "medium", "high"] = "high"
    threshold_db: float = -22
    ratio: float = 6
    attack_ms: float = 5
    release_ms: float = 120
    makeup_db: float = 3


class DialogueBoostCfg(BaseModel):
    enabled: bool = True
    freq_hz: float = 2200
    width_hz: float = 1800
    gain_db: float = 3.5


class AudioChainCfg(BaseModel):
    loudnorm_i: float = -14
    loudnorm_tp: float = -1.0
    loudnorm_lra: float = 7
    compressor: CompressorCfg = Field(default_factory=CompressorCfg)
    dialogue_boost: DialogueBoostCfg = Field(default_factory=DialogueBoostCfg)
    highpass_hz: float = 80


class EndingCfg(BaseModel):
    mode: Literal["cut_before_resolution", "continuity_text", "none"] = "cut_before_resolution"
    cut_before_s: float = 1.5
    continuity_text: str = "La escena sigue…"


class ThumbnailCfg(BaseModel):
    enabled: bool = True
    strategy: Literal["peak_frame", "cold_open", "midpoint"] = "peak_frame"
    width: int = 1080


class EditCfg(BaseModel):
    target_w: int = 1080
    target_h: int = 1920
    fps: int = 30
    video_codec: str = "libx264"
    preset: str = "medium"
    crf: int = 20
    audio_codec: str = "aac"
    audio_bitrate: str = "160k"
    dead_air: DeadAirCfg = Field(default_factory=DeadAirCfg)
    cold_open: ColdOpenCfg = Field(default_factory=ColdOpenCfg)
    crop: CropCfg = Field(default_factory=CropCfg)
    subtitles: SubtitlesCfg = Field(default_factory=SubtitlesCfg)
    rehooks: RehooksCfg = Field(default_factory=RehooksCfg)
    audio_chain: AudioChainCfg = Field(default_factory=AudioChainCfg)
    ending: EndingCfg = Field(default_factory=EndingCfg)
    thumbnail: ThumbnailCfg = Field(default_factory=ThumbnailCfg)


class CleanupCfg(BaseModel):
    delete_source: bool = True
    require_all_clips_ok: bool = True
    min_output_bytes: int = 1024
    keep_source_extensions: list[str] = []


class DailyCfg(BaseModel):
    auto_queue: int = 3
    render_top: int = 6
    keep_source: bool = True
    analyze_top: int = 0


class ReviewCfg(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    autoplay: bool = True
    reject_reasons: list[str] = [
        "mal encuadre",
        "aburrido",
        "subtítulos tapados",
        "mal corte",
        "otro",
    ]


class TrendSourcesCfg(BaseModel):
    manual_csv: bool = True
    seed_llm: bool = True
    justwatch: bool = False
    pytrends: bool = False
    tiktok_cc: bool = False
    reddit: bool = False
    letterboxd: bool = False
    anniversaries: bool = False


class TrendsCfg(BaseModel):
    sources: TrendSourcesCfg = Field(default_factory=TrendSourcesCfg)
    country: str = "ES"
    language: str = "es"
    trend_window_days: int = 7
    seed_count: int = 20
    anniversary_marks: list[int] = [10, 20, 25, 30]
    anniversary_window_days: int = 30
    release_dates_csv: Path = Path("input/manual/release_dates.csv")
    manual_csv: Path = Path("input/manual/trending.csv")
    default_saturation: float = 0.5
    request_timeout_s: float = 15
    cache_ttl_hours: float = 12


class LoggingCfg(BaseModel):
    level: str = "INFO"
    rich_tracebacks: bool = True
    file: Path | None = Path("data/logs/clipforge.log")
    progress_bars: bool = True


class RuntimeCfg(BaseModel):
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    tmp_dir: Path = Path("data/cache/tmp")
    ffmpeg_threads: int = 0
    fail_fast: bool = False


class Config(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    ingest: IngestCfg = Field(default_factory=IngestCfg)
    scenes: ScenesCfg = Field(default_factory=ScenesCfg)
    transcribe: TranscribeCfg = Field(default_factory=TranscribeCfg)
    audio: AudioCfg = Field(default_factory=AudioCfg)
    scoring: ScoringCfg = Field(default_factory=ScoringCfg)
    edit: EditCfg = Field(default_factory=EditCfg)
    cleanup: CleanupCfg = Field(default_factory=CleanupCfg)
    daily: DailyCfg = Field(default_factory=DailyCfg)
    review: ReviewCfg = Field(default_factory=ReviewCfg)
    trends: TrendsCfg = Field(default_factory=TrendsCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)
    runtime: RuntimeCfg = Field(default_factory=RuntimeCfg)

    # Filled in by ``load_config`` so modules can resolve relative paths.
    project_root: Path = Path(".")

    def resolve(self, path: Path | str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.project_root / p)

    def ensure_dirs(self) -> None:
        for d in self.paths.all_dirs():
            self.resolve(d).mkdir(parents=True, exist_ok=True)
        self.resolve(self.runtime.tmp_dir).mkdir(parents=True, exist_ok=True)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def find_config(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` looking for config.yaml."""
    env = os.environ.get("CLIPFORGE_CONFIG")
    if env:
        p = Path(env).expanduser()
        return p if p.exists() else None
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        cfg = candidate / DEFAULT_CONFIG_FILENAME
        if cfg.exists():
            return cfg
    return None


def load_config(path: Path | str | None = None) -> Config:
    cfg_path = Path(path) if path else find_config()
    data: dict[str, Any] = {}
    root = Path.cwd()
    if cfg_path and cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{cfg_path}: top level must be a mapping")
        data = raw
        root = cfg_path.parent.resolve()
    merged = _deep_merge(Config().model_dump(mode="json"), data)
    merged["project_root"] = str(root)
    return Config.model_validate(merged)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide config singleton (CLI entrypoints may override with ``load_config``)."""
    return load_config()
