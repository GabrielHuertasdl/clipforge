"""The daily routine: one command that does the whole round.

1. Refresh every enabled trend source.
2. Auto-queue the best titles that are neither processed nor already queued.
3. Process every new video sitting in ``input/raw/`` (analyze + render).
4. Report what to download and how many clips are waiting for review.

Designed to be run repeatedly: every stage is idempotent, and one video blowing up never
stops the others. Source deletion is off by default (``daily.keep_source``) until you decide
otherwise in config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config
from .db import Clip, Video, log_error, select, session_scope
from .logging_utils import get_logger
from .trends import RankedTitle, load_ranking, refresh, set_status

log = get_logger("daily")


@dataclass(slots=True)
class VideoOutcome:
    video_id: int
    title: str
    clips: int = 0
    errors: list[str] = field(default_factory=list)
    skipped: str = ""
    source_deleted: bool = False


@dataclass(slots=True)
class DailyReport:
    dry_run: bool = False
    refreshed: bool = False
    signals: int = 0
    queued: list[RankedTitle] = field(default_factory=list)
    already_queued: list[RankedTitle] = field(default_factory=list)
    processed: list[VideoOutcome] = field(default_factory=list)
    new_videos: int = 0
    pending_clips: int = 0
    errors: list[str] = field(default_factory=list)


def pending_clip_count() -> int:
    with session_scope() as session:
        return len(list(session.exec(select(Clip).where(Clip.status == "pending")).all()))


def videos_to_process(cfg: Config) -> list[tuple[int, str]]:
    """Registered videos that still have no clips and whose source file is present."""
    with session_scope() as session:
        rows = list(session.exec(select(Video)).all())
        clipped = {
            c.video_id for c in session.exec(select(Clip)).all()
        }
        out = [
            (v.id or -1, v.display_title, v.path)
            for v in rows
            if v.id not in clipped and v.source_deleted_at is None
        ]
    return [(vid, title) for vid, title, path in out if Path(path).exists()]


def auto_queue(
    cfg: Config, count: int, dry_run: bool
) -> tuple[list[RankedTitle], list[RankedTitle]]:
    """Mark the best untouched titles as ``to_download``.

    Skips anything already processed, already queued or explicitly ignored, so running this
    every day just tops the queue up instead of re-marking the same films.
    """
    ranking = load_ranking()
    already = [r for r in ranking if r.status == "to_download"]
    candidates = [r for r in ranking if r.status == "new"][:count]
    if not dry_run:
        for item in candidates:
            set_status(item.slug, "to_download")
    return candidates, already


def process_video(
    cfg: Config, video_id: int, title: str, top: int, keep_source: bool, dry_run: bool
) -> VideoOutcome:
    """analyze + render one video. Never raises: failures land in the outcome."""
    from .audio_features import analyze_video
    from .cleanup import cleanup_source
    from .edit import render_video
    from .scenes import build_segments
    from .scoring import score_video
    from .transcribe import transcribe_video

    outcome = VideoOutcome(video_id=video_id, title=title)
    if dry_run:
        outcome.skipped = "dry run"
        return outcome

    try:
        candidates = build_segments(video_id, cfg)
        transcript = transcribe_video(video_id, cfg)
        features = analyze_video(video_id, cfg)
        score_video(
            video_id, cfg, transcript, features, candidates, top=cfg.daily.analyze_top
        )
        results, errors = render_video(video_id, cfg, transcript, features, top=top)
        outcome.clips = len(results)
        outcome.errors = list(errors)
    except Exception as exc:
        log.error("video %d (%s) failed: %s", video_id, title, exc)
        log_error(video_id, "daily", str(exc))
        outcome.errors.append(str(exc))
        return outcome

    decision = cleanup_source(
        video_id, cfg, keep_source=keep_source, had_errors=bool(outcome.errors)
    )
    outcome.source_deleted = decision.deleted
    return outcome


def run_daily(
    cfg: Config,
    top: int | None = None,
    dry_run: bool = False,
    keep_source: bool | None = None,
    skip_refresh: bool = False,
) -> DailyReport:
    """The whole routine. Safe to run repeatedly."""
    from .ingest import scan

    report = DailyReport(dry_run=dry_run)
    queue_count = top if top is not None else cfg.daily.auto_queue
    keep = cfg.daily.keep_source if keep_source is None else keep_source

    # 1. trend signals
    if not (dry_run or skip_refresh):
        try:
            ranked = refresh(cfg)
            report.refreshed = True
            report.signals = len(ranked)
        except Exception as exc:  # a dead source must not stop the rest of the routine
            log.error("trend refresh failed: %s", exc)
            report.errors.append(f"trends: {exc}")

    # 2. queue the best untouched titles
    report.queued, report.already_queued = auto_queue(cfg, queue_count, dry_run)

    # 3. ingest and process whatever is in input/raw/
    try:
        results = scan(cfg)
        report.new_videos = sum(1 for r in results if r.status == "new")
    except Exception as exc:
        log.error("scan failed: %s", exc)
        report.errors.append(f"scan: {exc}")

    for video_id, title in videos_to_process(cfg):
        report.processed.append(
            process_video(cfg, video_id, title, cfg.daily.render_top, keep, dry_run)
        )

    # 4. what is waiting for you
    report.pending_clips = pending_clip_count()
    return report


def report_payload(report: DailyReport) -> dict[str, Any]:
    """Plain dict, handy for logs and tests."""
    return {
        "dry_run": report.dry_run,
        "refreshed": report.refreshed,
        "queued": [r.title for r in report.queued],
        "already_queued": [r.title for r in report.already_queued],
        "videos": [
            {"id": o.video_id, "title": o.title, "clips": o.clips, "errors": len(o.errors)}
            for o in report.processed
        ],
        "pending_clips": report.pending_clips,
        "errors": report.errors,
    }
