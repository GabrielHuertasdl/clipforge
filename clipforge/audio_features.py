"""Module 4 — audio features.

Per-second RMS, silence ratio, onset rate and energy delta, plus two derived signals the
rest of the pipeline depends on:

* **dramatic peaks** — a sharp energy rise after at least ``quiet_window_s`` of quiet audio.
  Scoring uses them as evidence of tension; the editor snaps zoom punches to them.
* **silence ranges** longer than ``silence_min_ms`` — the raw material for dead-air trimming.

Computed from the 16 kHz WAV that ``transcribe.extract_audio`` already cached, and cached
itself under ``data/cache/audio_features/``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .db import Video, session_scope
from .logging_utils import get_logger
from .transcribe import extract_audio

log = get_logger("audio")

CACHE_VERSION = 1


@dataclass(slots=True)
class DramaticPeak:
    t: float           # timestamp of the energy rise
    delta_db: float    # how much it jumped
    quiet_s: float     # how long the preceding lull lasted
    level_db: float    # level reached


@dataclass(slots=True)
class SilenceRange:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class AudioFeatures:
    frame_s: float = 1.0
    duration_s: float = 0.0
    rms: list[float] = field(default_factory=list)
    rms_db: list[float] = field(default_factory=list)
    silence_ratio: list[float] = field(default_factory=list)
    onset_rate: list[float] = field(default_factory=list)
    energy_delta: list[float] = field(default_factory=list)
    peaks: list[DramaticPeak] = field(default_factory=list)
    silences: list[SilenceRange] = field(default_factory=list)

    # ---- windowing helpers used by scoring and by the editor

    def _slice(self, values: list[float], start: float, end: float) -> np.ndarray:
        if not values or self.frame_s <= 0:
            return np.array([])
        lo = max(0, int(start / self.frame_s))
        hi = min(len(values), max(lo + 1, int(end / self.frame_s)))
        return np.asarray(values[lo:hi], dtype=float)

    def peaks_between(self, start: float, end: float) -> list[DramaticPeak]:
        return [p for p in self.peaks if start <= p.t < end]

    def silences_between(self, start: float, end: float) -> list[SilenceRange]:
        """Silence ranges clipped to the window (partial overlaps are trimmed, not dropped)."""
        out = []
        for s in self.silences:
            if s.end <= start or s.start >= end:
                continue
            out.append(SilenceRange(max(s.start, start), min(s.end, end)))
        return out

    def summary(self, start: float, end: float) -> dict[str, float | int]:
        """Compact numeric summary of a window, fed to the scoring prompt."""
        rms_db = self._slice(self.rms_db, start, end)
        sil = self._slice(self.silence_ratio, start, end)
        onset = self._slice(self.onset_rate, start, end)
        delta = self._slice(self.energy_delta, start, end)
        span = max(1e-6, end - start)
        silent_s = sum(s.duration for s in self.silences_between(start, end))
        return {
            "mean_db": round(float(rms_db.mean()), 2) if rms_db.size else 0.0,
            "peak_db": round(float(rms_db.max()), 2) if rms_db.size else 0.0,
            "dynamic_range_db": round(float(rms_db.max() - rms_db.min()), 2)
            if rms_db.size
            else 0.0,
            "silence_ratio": round(float(sil.mean()), 3) if sil.size else 0.0,
            "silent_seconds": round(silent_s, 1),
            "silence_fraction": round(silent_s / span, 3),
            "onsets_per_s": round(float(onset.mean()), 2) if onset.size else 0.0,
            "max_energy_jump_db": round(float(delta.max()), 2) if delta.size else 0.0,
            "dramatic_peaks": len(self.peaks_between(start, end)),
        }

    def to_dict(self) -> dict:
        return {
            "version": CACHE_VERSION,
            "frame_s": self.frame_s,
            "duration_s": self.duration_s,
            "rms": self.rms,
            "rms_db": self.rms_db,
            "silence_ratio": self.silence_ratio,
            "onset_rate": self.onset_rate,
            "energy_delta": self.energy_delta,
            "peaks": [asdict(p) for p in self.peaks],
            "silences": [asdict(s) for s in self.silences],
        }

    @classmethod
    def from_dict(cls, data: dict) -> AudioFeatures:
        return cls(
            frame_s=data.get("frame_s", 1.0),
            duration_s=data.get("duration_s", 0.0),
            rms=data.get("rms", []),
            rms_db=data.get("rms_db", []),
            silence_ratio=data.get("silence_ratio", []),
            onset_rate=data.get("onset_rate", []),
            energy_delta=data.get("energy_delta", []),
            peaks=[DramaticPeak(**p) for p in data.get("peaks", [])],
            silences=[SilenceRange(**s) for s in data.get("silences", [])],
        )


# --------------------------------------------------------------------------- pure helpers


def detect_dramatic_peaks(
    rms_db: list[float] | np.ndarray,
    frame_s: float,
    cfg: Config,
) -> list[DramaticPeak]:
    """Find sharp energy rises that follow a lull.

    A frame qualifies when: it sits above the ``peak_rms_percentile`` level, the preceding
    ``quiet_window_s`` averaged below the ``quiet_rms_percentile`` level, and the jump is at
    least ``peak_delta_db``. Peaks closer than ``peak_min_gap_s`` collapse to the loudest.
    """
    a = np.asarray(rms_db, dtype=float)
    if a.size < 3 or frame_s <= 0:
        return []

    quiet_level = float(np.percentile(a, cfg.audio.quiet_rms_percentile))
    loud_level = float(np.percentile(a, cfg.audio.peak_rms_percentile))
    quiet_frames = max(1, int(round(cfg.audio.quiet_window_s / frame_s)))

    found: list[DramaticPeak] = []
    for i in range(quiet_frames, a.size):
        level = a[i]
        if level < loud_level:
            continue
        before = a[i - quiet_frames : i]
        if before.size == 0 or before.mean() > quiet_level:
            continue
        delta = float(level - before.mean())
        if delta < cfg.audio.peak_delta_db:
            continue
        found.append(
            DramaticPeak(
                t=round(i * frame_s, 3),
                delta_db=round(delta, 2),
                quiet_s=round(quiet_frames * frame_s, 2),
                level_db=round(float(level), 2),
            )
        )

    # Collapse bursts: keep the strongest peak inside each min-gap window.
    merged: list[DramaticPeak] = []
    for peak in found:
        if merged and peak.t - merged[-1].t < cfg.audio.peak_min_gap_s:
            if peak.delta_db > merged[-1].delta_db:
                merged[-1] = peak
            continue
        merged.append(peak)
    return merged


def invert_intervals(
    intervals: list[tuple[float, float]], duration: float
) -> list[tuple[float, float]]:
    """Return the gaps between (and around) a sorted list of non-silent intervals."""
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in sorted(intervals):
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        gaps.append((cursor, duration))
    return gaps


def silence_ranges(
    non_silent: list[tuple[float, float]], duration: float, min_ms: int
) -> list[SilenceRange]:
    """Silences longer than ``min_ms``, derived from the non-silent intervals."""
    min_s = min_ms / 1000.0
    return [
        SilenceRange(round(a, 3), round(b, 3))
        for a, b in invert_intervals(non_silent, duration)
        if (b - a) >= min_s
    ]


# --------------------------------------------------------------------------- computation


def _cache_file(cfg: Config, file_hash: str) -> Path:
    d = cfg.resolve(cfg.paths.cache_dir) / "audio_features"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{file_hash}.json"


def compute_features(wav_path: Path | str, cfg: Config) -> AudioFeatures:
    """Compute every per-second feature from a mono WAV."""
    import librosa

    sr = cfg.audio.sample_rate
    y, sr = librosa.load(str(wav_path), sr=sr, mono=True)
    duration = float(len(y) / sr) if sr else 0.0
    frame_s = cfg.audio.frame_s
    hop = max(1, int(round(sr * frame_s)))

    rms = librosa.feature.rms(y=y, frame_length=hop, hop_length=hop, center=False)[0]
    # Floor at -80 dBFS: digital silence would otherwise report ~-160 dB and blow up every
    # delta/dynamic-range number downstream.
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-4))

    # Per-frame silence ratio: fraction of samples below a small absolute threshold.
    n_frames = len(rms)
    usable = y[: n_frames * hop].reshape(n_frames, hop) if n_frames else np.empty((0, hop))
    peak_amplitude = float(np.max(np.abs(y)) or 1.0)
    quiet_sample_threshold = 10 ** (-cfg.audio.silence_top_db / 20.0) * peak_amplitude
    silence_ratio = (
        (np.abs(usable) < quiet_sample_threshold).mean(axis=1) if n_frames else np.array([])
    )

    onset_times = librosa.onset.onset_detect(
        y=y, sr=sr, units="time", backtrack=cfg.audio.onset_backtrack
    )
    onset_rate = np.zeros(n_frames)
    for t in onset_times:
        idx = int(t / frame_s)
        if 0 <= idx < n_frames:
            onset_rate[idx] += 1.0
    onset_rate = onset_rate / frame_s

    energy_delta = np.diff(rms_db, prepend=rms_db[:1] if n_frames else np.array([0.0]))

    non_silent = librosa.effects.split(
        y, top_db=cfg.audio.silence_top_db, frame_length=2048, hop_length=512
    )
    intervals = [(float(a) / sr, float(b) / sr) for a, b in non_silent]

    features = AudioFeatures(
        frame_s=frame_s,
        duration_s=duration,
        rms=[round(float(v), 6) for v in rms],
        rms_db=[round(float(v), 3) for v in rms_db],
        silence_ratio=[round(float(v), 4) for v in silence_ratio],
        onset_rate=[round(float(v), 3) for v in onset_rate],
        energy_delta=[round(float(v), 3) for v in energy_delta],
    )
    features.peaks = detect_dramatic_peaks(features.rms_db, frame_s, cfg)
    features.silences = silence_ranges(intervals, duration, cfg.audio.silence_min_ms)
    return features


def analyze_video(video_id: int, cfg: Config, force: bool = False) -> AudioFeatures:
    """Module-4 entry point for a registered video (cached by file hash)."""
    with session_scope() as session:
        video = session.get(Video, video_id)
        if video is None:
            raise LookupError(f"video {video_id} not found")
        path, file_hash = Path(video.path), video.file_hash

    cache = _cache_file(cfg, file_hash)
    if cache.exists() and not force:
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("version") == CACHE_VERSION:
                log.info("audio features from cache (%s)", cache.name)
                return AudioFeatures.from_dict(data)
        except (json.JSONDecodeError, OSError, TypeError):
            log.warning("ignoring corrupt audio-feature cache %s", cache.name)

    if not path.exists():
        raise FileNotFoundError(f"source file gone: {path}")

    wav = extract_audio(path, cfg, file_hash)
    log.info("computing audio features for %s", path.name)
    features = compute_features(wav, cfg)
    log.info(
        "%d frames · %d dramatic peaks · %d silences >%dms",
        len(features.rms),
        len(features.peaks),
        len(features.silences),
        cfg.audio.silence_min_ms,
    )

    cache.write_text(json.dumps(features.to_dict()), encoding="utf-8")
    with session_scope() as session:
        video = session.get(Video, video_id)
        if video:
            video.audio_done = True
            session.add(video)
    return features


def load_cached(video_id: int, cfg: Config) -> AudioFeatures | None:
    with session_scope() as session:
        video = session.get(Video, video_id)
        if video is None:
            return None
        file_hash = video.file_hash
    cache = _cache_file(cfg, file_hash)
    if not cache.exists():
        return None
    try:
        return AudioFeatures.from_dict(json.loads(cache.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, TypeError):
        return None
