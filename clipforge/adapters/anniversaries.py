"""Round-number anniversaries (10/20/25/30 years) computed from a CSV of release dates.

``input/manual/release_dates.csv``::

    title,year,release_date,country
    Heat,1995,1995-12-15,US

The date maths is pure and takes ``today`` as an argument so it can be tested against a
fixed calendar day instead of whatever day the suite happens to run on.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .base import RawSignal, TrendAdapter


@dataclass(slots=True)
class Release:
    title: str
    year: int | None
    release_date: date
    country: str = ""


def days_until(target: date, today: date) -> int:
    """Signed distance in days; negative means the date has already passed."""
    return (target - today).days


def anniversary_signal(
    release: date, today: date, marks: list[int], window_days: int
) -> tuple[int, int] | None:
    """Return (years, days_away) if a round anniversary falls inside the window."""
    best: tuple[int, int] | None = None
    for mark in marks:
        try:
            target = release.replace(year=release.year + mark)
        except ValueError:  # 29 February
            target = release.replace(year=release.year + mark, day=28)
        distance = days_until(target, today)
        if abs(distance) <= window_days and (best is None or abs(distance) < abs(best[1])):
            best = (mark, distance)
    return best


def anniversary_slope(years: int, distance: int, marks: list[int], window_days: int) -> float:
    """Closer dates and rounder anniversaries score higher (0-100)."""
    closeness = 1.0 - abs(distance) / max(1, window_days)
    return round(min(100.0, 40.0 + 60.0 * closeness * (years / max(marks))), 1)


def load_releases(path: Path) -> list[Release]:
    """Parse the CSV, skipping rows with a missing or unparseable date."""
    if not path.exists():
        return []
    releases: list[Release] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            keys = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            title = keys.get("title", "")
            raw_date = keys.get("release_date", "")
            if not title or not raw_date:
                continue
            try:
                parsed = datetime.strptime(raw_date, "%Y-%m-%d").date()
            except ValueError:
                continue
            year = keys.get("year", "")
            releases.append(
                Release(
                    title=title,
                    year=int(year) if year.isdigit() else parsed.year,
                    release_date=parsed,
                    country=keys.get("country", ""),
                )
            )
    return releases


def signals_for(
    releases: list[Release],
    today: date,
    marks: list[int],
    window_days: int,
    source: str = "anniversaries",
) -> list[RawSignal]:
    """Every film with a round anniversary inside the window, strongest first."""
    signals: list[RawSignal] = []
    for release in releases:
        hit = anniversary_signal(release.release_date, today, marks, window_days)
        if hit is None:
            continue
        years, distance = hit
        signals.append(
            RawSignal(
                title=release.title,
                source=source,
                year=release.year,
                value=float(years),
                slope=anniversary_slope(years, distance, marks, window_days),
                meta={
                    "anniversary_years": years,
                    "days_away": distance,
                    "release_date": release.release_date.isoformat(),
                    "country": release.country,
                },
            )
        )
    signals.sort(key=lambda s: -s.slope)
    return signals


class AnniversariesAdapter(TrendAdapter):
    name = "anniversaries"

    def releases(self) -> list[Release]:
        return load_releases(self.cfg.resolve(self.cfg.trends.release_dates_csv))

    def fetch(self, today: date | None = None) -> list[RawSignal]:
        releases = self.releases()
        if not releases:
            self.log.info(
                "no release-date CSV at %s", self.cfg.resolve(self.cfg.trends.release_dates_csv)
            )
            return []
        return signals_for(
            releases,
            today or date.today(),
            self.cfg.trends.anniversary_marks,
            self.cfg.trends.anniversary_window_days,
        )
