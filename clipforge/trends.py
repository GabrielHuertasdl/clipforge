"""Module 8 — trend detection.

Each source is an adapter (``clipforge/adapters/``) and **all of them are optional**. The
manual CSV is implemented first and is the one the whole flow is validated against; every
remote source ships disabled in ``config.yaml`` and fails soft.

Priority score::

    score = trend_slope × (1 / saturation) × iconic_scene_density

* ``trend_slope`` — how fast interest is climbing (0-100), not the absolute volume: a film
  going 20 -> 45 beats one sitting flat at 80.
* ``saturation`` — how crowded the clip space already is (0-1, clamped away from zero);
  either supplied by a source, set by hand in the CSV, or estimated from how many sources
  are already shouting about the title.
* ``iconic_scene_density`` — how many memorable scenes the film has per hour, reused from
  the scoring module's cached iconic-scene map when it exists (no extra API calls).

Titles already processed are flagged so they never get queued twice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, timedelta
from pathlib import Path
from typing import Any

from .adapters import ADAPTERS, RawSignal, slugify
from .config import Config
from .db import TrendSignal, TrendTitle, Video, select, session_scope, utcnow
from .logging_utils import get_logger

log = get_logger("trends")

MIN_SATURATION = 0.05


@dataclass(slots=True)
class RankedTitle:
    slug: str
    title: str
    year: int | None
    score: float
    trend_slope: float
    saturation: float
    iconic_density: float
    top_source: str
    sources: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    status: str = "new"
    processed: bool = False


# --------------------------------------------------------------------------- scoring


def priority_score(slope: float, saturation: float, iconic_density: float) -> float:
    """score = slope × (1 / saturation) × iconic density, clamped to keep it readable."""
    saturation = max(MIN_SATURATION, min(1.0, saturation))
    return round(slope * (1.0 / saturation) * max(0.1, iconic_density), 2)


def estimate_saturation(signals: list[RawSignal], default: float) -> float:
    """Use the value a source gave us; otherwise estimate it from how loud the chorus is.

    More independent sources shouting about a title means more people are already making
    clips of it, so the estimate rises with the number of distinct sources.
    """
    explicit = [s.saturation for s in signals if s.saturation is not None]
    if explicit:
        return max(MIN_SATURATION, min(1.0, sum(explicit) / len(explicit)))
    distinct_sources = len({s.source for s in signals})
    return max(MIN_SATURATION, min(1.0, default + 0.12 * (distinct_sources - 1)))


def iconic_density_for(title: str, year: int | None, cfg: Config) -> float:
    """Iconic scenes per hour, read from the scoring cache. No API call is made here."""
    from .scoring import _iconic_cache_path

    path: Path = _iconic_cache_path(cfg, title, year)
    if not path.exists():
        return 1.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 1.0
    if not data.get("recognized"):
        return 1.0
    scenes = data.get("scenes") or []
    if not scenes:
        return 1.0
    span_minutes = max(60.0, max(float(s.get("approx_minute", 0)) for s in scenes))
    per_hour = len(scenes) / (span_minutes / 60.0)
    # 4 memorable scenes an hour is a normal film; scale around that, capped.
    return round(min(2.5, max(0.5, per_hour / 4.0)), 3)


def processed_slugs() -> dict[str, int]:
    """Films already in the database (any state), so they are never queued twice."""
    out: dict[str, int] = {}
    with session_scope() as session:
        for video in session.exec(select(Video)).all():
            out[slugify(video.display_title, video.year)] = video.id or -1
            out.setdefault(slugify(video.display_title), video.id or -1)
    return out


# --------------------------------------------------------------------------- collection


def collect_signals(cfg: Config, only: list[str] | None = None) -> list[RawSignal]:
    """Run every enabled adapter. A failing source is logged and skipped."""
    signals: list[RawSignal] = []
    for name, adapter_cls in ADAPTERS.items():
        if only and name not in only:
            continue
        signals.extend(adapter_cls(cfg).safe_fetch())
    return signals


def rank(signals: list[RawSignal], cfg: Config) -> list[RankedTitle]:
    """Merge signals per film and rank them by priority score."""
    by_slug: dict[str, list[RawSignal]] = {}
    for signal in signals:
        if signal.source == "tiktok_cc":
            continue  # hashtags are context, not films
        by_slug.setdefault(signal.slug, []).append(signal)

    done = processed_slugs()
    ranked: list[RankedTitle] = []
    for slug, group in by_slug.items():
        best = max(group, key=lambda s: s.slope)
        slope = max(s.slope for s in group)
        # A title confirmed by several sources gets a modest boost.
        sources = sorted({s.source for s in group})
        slope = min(100.0, slope * (1.0 + 0.08 * (len(sources) - 1)))
        saturation = estimate_saturation(group, cfg.trends.default_saturation)
        density = iconic_density_for(best.title, best.year, cfg)
        video_id = done.get(slug) or done.get(slugify(best.title))
        ranked.append(
            RankedTitle(
                slug=slug,
                title=best.title,
                year=best.year,
                score=priority_score(slope, saturation, density),
                trend_slope=round(slope, 1),
                saturation=round(saturation, 3),
                iconic_density=density,
                top_source=best.source,
                sources=sources,
                detail={s.source: {"value": s.value, "slope": s.slope, **s.meta} for s in group},
                status="processed" if video_id else "new",
                processed=bool(video_id),
            )
        )
    ranked.sort(key=lambda r: -r.score)
    return ranked


def persist(ranked: list[RankedTitle], signals: list[RawSignal]) -> None:
    """Store the ranking and the raw readings, keeping any manual status already set."""
    with session_scope() as session:
        for signal in signals:
            session.add(
                TrendSignal(
                    slug=signal.slug,
                    source=signal.source,
                    title=signal.title,
                    year=signal.year,
                    value=signal.value,
                    slope=signal.slope,
                    saturation=signal.saturation,
                    meta=signal.meta,
                )
            )
        for item in ranked:
            row = session.exec(
                select(TrendTitle).where(TrendTitle.slug == item.slug)
            ).first()
            if row is None:
                row = TrendTitle(slug=item.slug, title=item.title, year=item.year)
            row.title = item.title
            row.year = item.year
            row.score = item.score
            row.trend_slope = item.trend_slope
            row.saturation = item.saturation
            row.iconic_density = item.iconic_density
            row.top_source = item.top_source
            row.sources = item.sources
            row.detail = item.detail
            # A manual "to_download" mark survives a refresh; "processed" always wins.
            if item.processed:
                row.status = "processed"
            elif row.status not in ("to_download", "ignored"):
                row.status = "new"
            row.updated_at = utcnow()
            session.add(row)


def recent_signals(exclude_sources: set[str], max_age_hours: float) -> list[RawSignal]:
    """Latest stored reading per (film, source), for sources not queried in this run.

    Without this a partial refresh (``--source pytrends``) would rank titles using that one
    source alone and wipe what the others had contributed.
    """
    cutoff = utcnow() - timedelta(hours=max_age_hours)
    latest: dict[tuple[str, str], RawSignal] = {}
    with session_scope() as session:
        rows = list(session.exec(select(TrendSignal)).all())
        for row in sorted(rows, key=lambda r: r.created_at):
            created = row.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            if created < cutoff or row.source in exclude_sources:
                continue
            latest[(row.slug, row.source)] = RawSignal(
                title=row.title,
                source=row.source,
                year=row.year,
                value=row.value,
                slope=row.slope,
                saturation=row.saturation,
                meta=dict(row.meta or {}),
            )
    return list(latest.values())


def refresh(
    cfg: Config, only: list[str] | None = None, merge_stored_hours: float = 72.0
) -> list[RankedTitle]:
    """Full pass: collect, rank, persist.

    When only some sources are refreshed, recent readings from the others are merged back in
    so the ranking still reflects everything the system knows.
    """
    signals = collect_signals(cfg, only=only)
    queried = set(only) if only else set(ADAPTERS)
    carried = recent_signals(queried, merge_stored_hours)
    if carried:
        log.info("carrying %d stored signal(s) from sources not queried now", len(carried))
    if not signals:
        log.warning(
            "no signals collected — enable a source in config.yaml or fill %s",
            cfg.resolve(cfg.trends.manual_csv),
        )
    ranked = rank(signals + carried, cfg)
    persist(ranked, signals)
    log.info("%d title(s) ranked from %d signal(s)", len(ranked), len(signals))
    return ranked


def load_ranking(limit: int = 0, include_processed: bool = True) -> list[RankedTitle]:
    """Read the stored ranking back — no network, no API."""
    with session_scope() as session:
        rows = list(session.exec(select(TrendTitle)).all())
        out = [
            RankedTitle(
                slug=r.slug,
                title=r.title,
                year=r.year,
                score=r.score,
                trend_slope=r.trend_slope,
                saturation=r.saturation,
                iconic_density=r.iconic_density,
                top_source=r.top_source,
                sources=list(r.sources or []),
                detail=dict(r.detail or {}),
                status=r.status,
                processed=r.status == "processed",
            )
            for r in rows
        ]
    if not include_processed:
        out = [r for r in out if not r.processed]
    out.sort(key=lambda r: -r.score)
    return out[:limit] if limit else out


def set_status(slug: str, status: str) -> str:
    """Mark a title as ``to_download`` / ``new`` / ``ignored`` from the dashboard."""
    if status not in ("new", "to_download", "processed", "ignored"):
        raise ValueError(f"unknown status: {status}")
    with session_scope() as session:
        row = session.exec(select(TrendTitle).where(TrendTitle.slug == slug)).first()
        if row is None:
            raise LookupError(f"title {slug} not found")
        row.status = status
        row.updated_at = utcnow()
        session.add(row)
        return row.status


def hashtag_signals(limit: int = 12) -> list[dict[str, Any]]:
    """Latest TikTok hashtag readings, shown as context on the dashboard."""
    with session_scope() as session:
        rows = list(
            session.exec(select(TrendSignal).where(TrendSignal.source == "tiktok_cc")).all()
        )
        latest: dict[str, dict[str, Any]] = {}
        for row in sorted(rows, key=lambda r: r.created_at):
            latest[row.title] = {"tag": row.title, "slope": row.slope, "value": row.value}
    return sorted(latest.values(), key=lambda h: -h["slope"])[:limit]
