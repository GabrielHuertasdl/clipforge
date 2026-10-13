"""Module 2 — shot detection and candidate segmentation.

PySceneDetect gives us the shot list. Shots are then grouped into candidate segments of
``scenes.segment_min_s``-``segment_max_s`` seconds. Two hard rules:

1. A segment boundary is ALWAYS a shot boundary — we never cut mid-shot.
2. Segments that are mostly credits or black screen are dropped (mean-luminance test).

Both the shot list and the luminance samples are cached by file hash, so re-running is cheap.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .db import Segment, Video, select, session_scope
from .ffmpeg import probe
from .logging_utils import get_logger

log = get_logger("scenes")


@dataclass(slots=True)
class Shot:
    idx: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(slots=True)
class Candidate:
    idx: int
    start_s: float
    end_s: float
    shot_start: int
    shot_end: int
    shot_count: int
    dark_ratio: float = 0.0
    mean_luma: float = 0.0
    discarded: bool = False
    discard_reason: str | None = None
    priority: float = 0.0  # set by scoring's iconic-scene map; higher = analyse first

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(slots=True)
class LumaSamples:
    fps: float
    values: list[float]  # mean luminance 0-255, one per sample

    def times(self) -> np.ndarray:
        return np.arange(len(self.values)) / self.fps if self.fps else np.array([])


# --------------------------------------------------------------------------- cache


def _cache_path(cfg: Config, kind: str, file_hash: str) -> Path:
    d = cfg.resolve(cfg.paths.cache_dir) / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{file_hash}.json"


def _read_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        log.warning("ignoring corrupt cache %s", path)
        return None


def _write_cache(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------- detection


def detect_shots(
    video_path: Path | str,
    cfg: Config,
    file_hash: str | None = None,
    force: bool = False,
) -> list[Shot]:
    """Run PySceneDetect and return the shot list (cached by file hash)."""
    video_path = Path(video_path)
    cache = _cache_path(cfg, "scenes", file_hash) if file_hash else None

    if cache and not force:
        cached = _read_cache(cache)
        if cached and cached.get("threshold") == cfg.scenes.threshold:
            log.debug("shots from cache: %s", cache.name)
            return [Shot(**s) for s in cached["shots"]]

    from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video

    video = open_video(str(video_path))
    fps = video.frame_rate or 25.0
    min_scene_len = max(1, int(cfg.scenes.min_scene_len_s * fps))

    manager = SceneManager()
    if cfg.scenes.detector == "adaptive":
        manager.add_detector(
            AdaptiveDetector(adaptive_threshold=cfg.scenes.threshold, min_scene_len=min_scene_len)
        )
    else:
        manager.add_detector(
            ContentDetector(threshold=cfg.scenes.threshold, min_scene_len=min_scene_len)
        )
    manager.auto_downscale = False
    manager.downscale = max(1, cfg.scenes.downscale_factor)
    manager.detect_scenes(video, show_progress=False)

    scene_list = manager.get_scene_list()
    if not scene_list:  # single unbroken shot
        duration = probe(video_path, ffprobe=cfg.runtime.ffprobe).duration_s
        shots = [Shot(0, 0.0, duration)]
    else:
        shots = [
            Shot(i, float(start.get_seconds()), float(end.get_seconds()))
            for i, (start, end) in enumerate(scene_list)
        ]

    log.info("%d shots detected in %s", len(shots), video_path.name)
    if cache:
        _write_cache(
            cache,
            {
                "threshold": cfg.scenes.threshold,
                "detector": cfg.scenes.detector,
                "shots": [asdict(s) for s in shots],
            },
        )
    return shots


def sample_luminance(
    video_path: Path | str,
    cfg: Config,
    file_hash: str | None = None,
    force: bool = False,
) -> LumaSamples:
    """Mean frame luminance sampled at ``scenes.luma_sample_fps``.

    Uses a tiny grayscale rawvideo pipe — much cheaper than decoding full frames.
    """
    video_path = Path(video_path)
    fps = cfg.scenes.luma_sample_fps
    cache = _cache_path(cfg, "luma", file_hash) if file_hash else None

    if cache and not force:
        cached = _read_cache(cache)
        if cached and cached.get("fps") == fps:
            return LumaSamples(fps=fps, values=cached["values"])

    w, h = 64, 36
    cmd = [
        cfg.runtime.ffmpeg,
        "-v", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps},scale={w}:{h}",
        "-pix_fmt", "gray",
        "-f", "rawvideo",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 and not proc.stdout:
        err = proc.stderr.decode(errors="ignore")[-500:]
        raise RuntimeError(f"luminance sampling failed: {err}")

    frame_bytes = w * h
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = len(buf) // frame_bytes
    values = (
        buf[: n * frame_bytes].reshape(n, frame_bytes).mean(axis=1).astype(float).tolist()
        if n
        else []
    )

    if cache:
        _write_cache(cache, {"fps": fps, "values": values})
    return LumaSamples(fps=fps, values=values)


# --------------------------------------------------------------------------- grouping


def group_shots(shots: list[Shot], cfg: Config) -> list[Candidate]:
    """Group contiguous shots into candidate segments.

    Greedy, non-overlapping, left to right: keep adding whole shots while the running
    duration is below the target; stop as soon as adding the next shot would blow past
    ``segment_max_s``. Because we only ever append whole shots, every boundary produced here
    is a shot boundary.

    A shot longer than ``segment_max_s`` becomes a segment of its own and is later flagged
    oversized rather than being cut in half.
    """
    smin = cfg.scenes.segment_min_s
    smax = cfg.scenes.segment_max_s
    target = cfg.scenes.segment_target_s

    candidates: list[Candidate] = []
    i = 0
    n = len(shots)
    while i < n:
        start_shot = shots[i]
        j = i
        end_s = start_shot.end_s
        while j + 1 < n:
            nxt = shots[j + 1]
            new_dur = nxt.end_s - start_shot.start_s
            current_dur = end_s - start_shot.start_s
            if new_dur > smax:
                break
            # Past the target, stop unless we are still short of the minimum.
            if current_dur >= target:
                break
            j += 1
            end_s = nxt.end_s

        duration = end_s - start_shot.start_s
        cand = Candidate(
            idx=len(candidates),
            start_s=start_shot.start_s,
            end_s=end_s,
            shot_start=start_shot.idx,
            shot_end=shots[j].idx,
            shot_count=j - i + 1,
        )
        if duration < smin:
            cand.discarded = True
            cand.discard_reason = f"too short ({duration:.0f}s < {smin:.0f}s)"
        elif duration > smax:
            cand.discarded = True
            cand.discard_reason = f"single shot too long ({duration:.0f}s > {smax:.0f}s)"
        candidates.append(cand)
        i = j + 1

    return candidates


def dark_ratio_for(
    candidate: Candidate, luma: LumaSamples, threshold: float
) -> tuple[float, float]:
    """Return (dark_ratio, mean_luma) for the samples inside the candidate."""
    if not luma.values or not luma.fps:
        return 0.0, 0.0
    lo = int(candidate.start_s * luma.fps)
    hi = max(lo + 1, int(candidate.end_s * luma.fps))
    window = np.asarray(luma.values[lo:hi], dtype=float)
    if window.size == 0:
        return 0.0, 0.0
    return float((window < threshold).mean()), float(window.mean())


def annotate_candidates(
    candidates: list[Candidate], luma: LumaSamples, duration_s: float, cfg: Config
) -> list[Candidate]:
    """Flag credits/black segments and the head/tail zones we always skip."""
    head_limit = duration_s * cfg.scenes.skip_head_ratio
    tail_limit = duration_s * (1.0 - cfg.scenes.skip_tail_ratio)

    for cand in candidates:
        cand.dark_ratio, cand.mean_luma = dark_ratio_for(
            cand, luma, cfg.scenes.black_luma_threshold
        )
        if cand.discarded:
            continue
        if cand.end_s <= head_limit:
            cand.discarded = True
            cand.discard_reason = "inside the opening logos zone"
        elif cand.start_s >= tail_limit:
            cand.discarded = True
            cand.discard_reason = "inside the end-credits zone"
        elif cand.dark_ratio > cfg.scenes.max_dark_ratio:
            cand.discarded = True
            cand.discard_reason = f"mostly black/credits (dark_ratio={cand.dark_ratio:.2f})"
    return candidates


# --------------------------------------------------------------------------- orchestration


def build_segments(video_id: int, cfg: Config, force: bool = False) -> list[Candidate]:
    """Full module-2 pass for one video, persisting the candidates to the database."""
    with session_scope() as session:
        video = session.get(Video, video_id)
        if video is None:
            raise LookupError(f"video {video_id} not found")
        path = Path(video.path)
        file_hash = video.file_hash
        duration_s = video.duration_s
        # Materialise the rows inside the session: ORM instances detach once it closes.
        stored = [
            Candidate(
                idx=row.idx,
                start_s=row.start_s,
                end_s=row.end_s,
                shot_start=row.shot_start,
                shot_end=row.shot_end,
                shot_count=row.shot_count,
                dark_ratio=row.dark_ratio,
                mean_luma=row.mean_luma,
                discarded=row.discarded,
                discard_reason=row.discard_reason,
                priority=row.priority,
            )
            for row in session.exec(select(Segment).where(Segment.video_id == video_id)).all()
        ]

    if stored and not force:
        log.info(
            "video %d already segmented (%d segments) — use --force to redo",
            video_id,
            len(stored),
        )
        return sorted(stored, key=lambda c: c.idx)

    if not path.exists():
        raise FileNotFoundError(f"source file gone: {path}")

    shots = detect_shots(path, cfg, file_hash=file_hash, force=force)
    luma = sample_luminance(path, cfg, file_hash=file_hash, force=force)
    candidates = annotate_candidates(group_shots(shots, cfg), luma, duration_s, cfg)

    kept = [c for c in candidates if not c.discarded]
    log.info(
        "%s: %d candidate segments, %d kept (%d discarded)",
        path.name,
        len(candidates),
        len(kept),
        len(candidates) - len(kept),
    )

    with session_scope() as session:
        for old in session.exec(select(Segment).where(Segment.video_id == video_id)).all():
            session.delete(old)
        session.flush()
        for c in candidates:
            session.add(
                Segment(
                    video_id=video_id,
                    idx=c.idx,
                    start_s=c.start_s,
                    end_s=c.end_s,
                    duration_s=c.duration_s,
                    shot_start=c.shot_start,
                    shot_end=c.shot_end,
                    shot_count=c.shot_count,
                    dark_ratio=c.dark_ratio,
                    mean_luma=c.mean_luma,
                    discarded=c.discarded,
                    discard_reason=c.discard_reason,
                )
            )
        video = session.get(Video, video_id)
        if video:
            video.scenes_done = True
            session.add(video)

    return candidates
