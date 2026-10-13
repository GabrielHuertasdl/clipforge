"""Anniversary detection, pinned to fixed calendar dates.

The date maths takes ``today`` as an argument precisely so these assertions do not depend on
the day the suite runs.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from clipforge.adapters.anniversaries import (
    AnniversariesAdapter,
    Release,
    anniversary_signal,
    anniversary_slope,
    load_releases,
    signals_for,
)
from clipforge.config import Config, load_config

MARKS = [10, 20, 25, 30]
WINDOW = 30
REPO_CSV = Path("input/manual/release_dates.csv")


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config("config.yaml")


@pytest.fixture(scope="module")
def releases() -> list[Release]:
    return load_releases(REPO_CSV)


# --------------------------------------------------------------------------- pure maths


def test_detects_each_round_anniversary():
    release = date(1995, 12, 15)
    assert anniversary_signal(release, date(2025, 12, 1), MARKS, WINDOW) == (30, 14)
    assert anniversary_signal(release, date(2020, 12, 20), MARKS, WINDOW) == (25, -5)
    assert anniversary_signal(release, date(2015, 12, 15), MARKS, WINDOW) == (20, 0)
    assert anniversary_signal(release, date(2005, 12, 10), MARKS, WINDOW) == (10, 5)


def test_non_round_anniversaries_are_ignored():
    release = date(1995, 12, 15)
    # 15, 22 and 31 years: none of them is a configured mark
    assert anniversary_signal(release, date(2010, 12, 15), MARKS, WINDOW) is None
    assert anniversary_signal(release, date(2017, 12, 15), MARKS, WINDOW) is None
    assert anniversary_signal(release, date(2026, 12, 15), MARKS, WINDOW) is None


def test_window_edges():
    release = date(2000, 6, 15)
    assert anniversary_signal(release, date(2025, 5, 16), MARKS, 30) == (25, 30)   # 30 days out
    assert anniversary_signal(release, date(2025, 5, 15), MARKS, 30) is None       # 31: outside
    assert anniversary_signal(release, date(2025, 7, 15), MARKS, 30) == (25, -30)  # 30 days past
    assert anniversary_signal(release, date(2025, 7, 16), MARKS, 30) is None


def test_leap_day_release_does_not_crash():
    assert anniversary_signal(date(1996, 2, 29), date(2026, 2, 28), MARKS, WINDOW) == (30, 0)


def test_closest_mark_wins_when_two_are_in_range():
    # A 1995 film seen in 2020 is 25 years old; the 20-year mark is nowhere near.
    assert anniversary_signal(date(1995, 6, 1), date(2020, 6, 3), MARKS, WINDOW)[0] == 25


def test_slope_rewards_closeness_and_roundness():
    assert anniversary_slope(30, 0, MARKS, WINDOW) > anniversary_slope(30, 25, MARKS, WINDOW)
    assert anniversary_slope(30, 0, MARKS, WINDOW) > anniversary_slope(10, 0, MARKS, WINDOW)
    assert 0.0 <= anniversary_slope(10, 30, MARKS, WINDOW) <= 100.0


# --------------------------------------------------------------------------- CSV parsing


def test_repo_csv_is_present_and_populated(releases):
    assert 250 <= len(releases) <= 320
    titles = {r.title for r in releases}
    assert {"Heat", "Matrix", "El laberinto del fauno", "Zodiac"} <= titles


def test_every_release_date_matches_its_year_column(releases):
    assert [r.title for r in releases if r.release_date.year != r.year] == []


def test_load_skips_unparseable_rows(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text(
        "title,year,release_date,country\n"
        "Buena,1995,1995-12-15,US\n"
        "Sin fecha,1995,,US\n"
        "Fecha mala,1995,15/12/1995,US\n"
        ",1995,1995-01-01,US\n",
        encoding="utf-8",
    )
    loaded = load_releases(path)
    assert [r.title for r in loaded] == ["Buena"]


def test_missing_csv_returns_nothing(tmp_path):
    assert load_releases(tmp_path / "nope.csv") == []


# --------------------------------------------------------------------------- fixed dates


def test_first_of_december_2026(releases):
    """On 2026-12-01 exactly these films have a 10/20/25/30 year mark within 30 days."""
    signals = signals_for(releases, date(2026, 12, 1), MARKS, WINDOW)
    got = {s.title: (s.meta["anniversary_years"], s.meta["days_away"]) for s in signals}
    assert got == {
        "Jerry Maguire": (30, 12),                                    # 1996-12-13
        "Scream": (30, 19),                                           # 1996-12-20
        "El paciente inglés": (30, -16),                              # 1996-11-15
        "Ocean's Eleven": (25, 6),                                    # 2001-12-07
        "El señor de los anillos la comunidad del anillo": (25, 18),  # 2001-12-19
        "Monstruos S.A.": (25, -29),                                  # 2001-11-02
        "Casino Royale": (20, -14),                                   # 2006-11-17
        "La La Land": (10, 8),                                        # 2016-12-09
        "Animales nocturnos": (10, -13),                              # 2016-11-18
        "La llegada": (10, -20),                                      # 2016-11-11
    }


def test_films_just_outside_the_window_are_excluded(releases):
    titles = {s.title for s in signals_for(releases, date(2026, 12, 1), MARKS, WINDOW)}
    assert "El prestigio" not in titles   # 2006-10-20: 20 years, but 42 days past
    assert "Heat" not in titles           # 1995-12-15: 31 years, not a round mark


def test_ranking_puts_the_nearest_anniversary_first(releases):
    signals = signals_for(releases, date(2027, 3, 1), MARKS, WINDOW)
    assert signals[0].title == "Zodiac"          # 2007-03-02, one day away
    assert signals[0].meta["days_away"] == 1
    assert all(a.slope >= b.slope for a, b in zip(signals, signals[1:], strict=False))


def test_a_quiet_date_produces_few_or_no_signals(releases):
    # Mid-August: Spanish and US release calendars are thin
    signals = signals_for(releases, date(2026, 8, 5), MARKS, WINDOW)
    assert len(signals) <= 6


def test_adapter_uses_the_configured_window_and_marks(cfg, releases):
    narrow = Config(project_root=cfg.project_root)
    narrow.trends.anniversary_window_days = 3
    narrow.trends.anniversary_marks = [30]
    adapter = AnniversariesAdapter(narrow)
    adapter.releases = lambda: releases  # type: ignore[method-assign]
    signals = adapter.fetch(today=date(2026, 12, 13))
    assert [s.title for s in signals] == ["Jerry Maguire"]


def test_source_is_enabled_in_the_repo_config(cfg):
    assert cfg.trends.sources.anniversaries is True
