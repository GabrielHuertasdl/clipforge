"""Trend detector tests. No network: remote adapters are exercised through fakes."""

from __future__ import annotations

from datetime import date

import pytest

from clipforge.adapters import RawSignal, slugify
from clipforge.adapters.anniversaries import AnniversariesAdapter, anniversary_signal
from clipforge.adapters.justwatch import arrival_slope, days_since
from clipforge.adapters.letterboxd import parse_films
from clipforge.adapters.manual_csv import ManualCSVAdapter
from clipforge.adapters.pytrends_source import normalised_slope
from clipforge.adapters.reddit import extract_titles
from clipforge.config import Config
from clipforge.db import TrendTitle, Video, init_engine, select, session_scope
from clipforge.trends import (
    estimate_saturation,
    load_ranking,
    priority_score,
    rank,
    refresh,
    set_status,
)


@pytest.fixture
def env(tmp_path):
    cfg = Config(project_root=tmp_path)
    cfg.ensure_dirs()
    init_engine(tmp_path / "trends.db")
    return cfg


# --------------------------------------------------------------------------- scoring


def test_priority_formula():
    # slope × (1/saturation) × iconic density
    assert priority_score(60, 0.5, 1.0) == pytest.approx(120.0)
    assert priority_score(60, 0.25, 1.0) == pytest.approx(240.0)
    assert priority_score(60, 0.5, 2.0) == pytest.approx(240.0)


def test_low_saturation_beats_raw_volume():
    climbing_unsaturated = priority_score(45, 0.2, 1.0)
    flat_saturated = priority_score(80, 0.9, 1.0)
    assert climbing_unsaturated > flat_saturated


def test_saturation_is_clamped_away_from_zero():
    assert priority_score(50, 0.0, 1.0) == priority_score(50, 0.05, 1.0)


def test_explicit_saturation_wins_over_the_estimate():
    signals = [RawSignal("Heat", "manual_csv", 1995, saturation=0.2)]
    assert estimate_saturation(signals, 0.5) == pytest.approx(0.2)


def test_more_sources_raise_the_estimated_saturation():
    one = [RawSignal("Heat", "reddit", 1995)]
    three = [
        RawSignal("Heat", "reddit", 1995),
        RawSignal("Heat", "letterboxd", 1995),
        RawSignal("Heat", "justwatch", 1995),
    ]
    assert estimate_saturation(three, 0.5) > estimate_saturation(one, 0.5)


def test_slug_is_stable_across_sources():
    assert slugify("El Padrino", 1972) == "el-padrino-1972"
    assert slugify("Amélie") == "amelie"
    assert slugify("Heat", 1995) == slugify("HEAT", 1995)


# --------------------------------------------------------------------------- ranking


def test_ranking_merges_sources_and_orders_by_score(env):
    signals = [
        RawSignal("Old Boy", "manual_csv", 2003, slope=80, saturation=0.2),
        RawSignal("Old Boy", "reddit", 2003, slope=60),
        RawSignal("El Padrino", "manual_csv", 1972, slope=35, saturation=0.85),
    ]
    ranked = rank(signals, env)
    assert [r.title for r in ranked] == ["Old Boy", "El Padrino"]
    assert ranked[0].sources == ["manual_csv", "reddit"]
    assert ranked[0].score > ranked[1].score


def test_hashtags_never_enter_the_film_ranking(env):
    signals = [
        RawSignal("#cinetok", "tiktok_cc", slope=90),
        RawSignal("Heat", "manual_csv", 1995, slope=40),
    ]
    assert [r.title for r in rank(signals, env)] == ["Heat"]


def test_already_processed_titles_are_flagged(env):
    with session_scope() as s:
        s.add(Video(file_hash="h", filename="Heat.mkv", path="/x", title="Heat", year=1995))
    ranked = rank([RawSignal("Heat", "manual_csv", 1995, slope=50)], env)
    assert ranked[0].processed is True
    assert ranked[0].status == "processed"


def test_unrecognised_film_gets_neutral_iconic_density(env):
    ranked = rank([RawSignal("Peli Desconocida", "manual_csv", 2020, slope=50)], env)
    assert ranked[0].iconic_density == 1.0


# --------------------------------------------------------------------------- manual CSV


def write_csv(cfg, text):
    path = cfg.resolve(cfg.trends.manual_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_manual_csv_is_read(env):
    write_csv(env, "title,year,slope,saturation,note\nHeat,1995,70,0.3,llega a Netflix\n")
    signals = ManualCSVAdapter(env).fetch()
    assert len(signals) == 1
    assert signals[0].title == "Heat" and signals[0].year == 1995
    assert signals[0].slope == 70 and signals[0].saturation == 0.3
    assert signals[0].meta["note"] == "llega a Netflix"


def test_manual_csv_only_needs_a_title(env):
    write_csv(env, "title\nHeat\n")
    signal = ManualCSVAdapter(env).fetch()[0]
    assert signal.year is None
    assert signal.saturation == env.trends.default_saturation


def test_manual_csv_survives_bad_numbers(env):
    write_csv(env, "title,year,slope,saturation\nHeat,mil,abc,xyz\n")
    signal = ManualCSVAdapter(env).fetch()[0]
    assert signal.slope == 50.0 and signal.year is None


def test_missing_csv_is_not_an_error(env):
    assert ManualCSVAdapter(env).fetch() == []


# --------------------------------------------------------------------------- optionality


def test_disabled_sources_return_nothing(env):
    assert env.trends.sources.reddit is False
    from clipforge.adapters.reddit import RedditAdapter

    assert RedditAdapter(env).safe_fetch() == []


def test_a_broken_source_never_raises(env):
    from clipforge.adapters.base import TrendAdapter

    class Exploding(TrendAdapter):
        name = "manual_csv"  # enabled by default

        def fetch(self):
            raise RuntimeError("endpoint changed")

    assert Exploding(env).safe_fetch() == []


def test_refresh_works_with_only_the_manual_csv(env):
    write_csv(env, "title,year,slope,saturation\nHeat,1995,70,0.3\nZodiac,2007,55,0.25\n")
    ranked = refresh(env)
    assert [r.title for r in ranked] == ["Heat", "Zodiac"]
    stored = load_ranking()
    assert len(stored) == 2 and stored[0].title == "Heat"


def test_manual_status_survives_a_refresh(env):
    write_csv(env, "title,year,slope\nHeat,1995,70\n")
    refresh(env)
    set_status("heat-1995", "to_download")
    refresh(env)
    with session_scope() as s:
        row = s.exec(select(TrendTitle).where(TrendTitle.slug == "heat-1995")).first()
        assert row.status == "to_download"


# --------------------------------------------------------------------------- adapter maths


def test_pytrends_slope_prefers_the_climb():
    climbing = normalised_slope([20, 25, 30, 35, 40, 45])
    flat_high = normalised_slope([80, 80, 80, 80, 80, 80])
    assert climbing > flat_high
    assert normalised_slope([50]) == 0.0
    assert normalised_slope([]) == 0.0


def test_pytrends_slope_is_bounded():
    assert 0.0 <= normalised_slope([0, 100, 0, 100]) <= 100.0
    assert normalised_slope([100, 80, 60, 40]) >= 0.0


def test_justwatch_window_peaks_in_the_first_two_weeks():
    assert arrival_slope(0) > arrival_slope(20)
    assert arrival_slope(7) > arrival_slope(40)
    assert arrival_slope(-3) > 0        # arriving soon still matters
    assert arrival_slope(400) == 0.0


def test_justwatch_date_parsing():
    assert days_since("2026-08-20T00:00:00Z", date(2026, 9, 3)) == 14
    assert days_since("nonsense", date(2026, 9, 3)) is None


def test_anniversary_detection():
    release = date(1995, 12, 15)
    hit = anniversary_signal(release, date(2025, 12, 1), [10, 20, 25, 30], 30)
    assert hit == (30, 14)
    assert anniversary_signal(release, date(2026, 6, 1), [10, 20, 25, 30], 30) is None


def test_anniversary_adapter_reads_the_csv(env, monkeypatch):
    env.trends.sources.anniversaries = True
    path = env.resolve(env.trends.release_dates_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    today = date.today()
    release = date(today.year - 25, today.month, min(today.day, 28))
    path.write_text(
        f"title,year,release_date\nPeli,{release.year},{release.isoformat()}\n", encoding="utf-8"
    )
    signals = AnniversariesAdapter(env).fetch()
    assert len(signals) == 1
    assert signals[0].meta["anniversary_years"] == 25


def test_reddit_title_extraction():
    assert extract_titles("Just rewatched Heat (1995) and wow") == [("Heat", 1995)]
    assert extract_titles("finally saw El Padrino (1972)") == [("El Padrino", 1972)]
    assert extract_titles('Why is "Old Boy" so good?') == [("Old Boy", None)]
    assert extract_titles("no film here") == []


def test_reddit_trim_drops_the_lead_in_words():
    from clipforge.adapters.reddit import trim_to_title

    assert trim_to_title("i watched The Godfather Part II") == "The Godfather Part II"
    assert trim_to_title("this is about Heat") == "Heat"
    assert trim_to_title("the") == ""


def test_letterboxd_parsing():
    html = (
        '<li data-film-slug="heat-1995"><img alt="Heat"></li>'
        '<li data-film-slug="zodiac-2007"><img alt="Zodiac"></li>'
    )
    assert parse_films(html) == [("Heat", 1995), ("Zodiac", 2007)]
    assert parse_films("<html></html>") == []


def test_partial_refresh_keeps_what_other_sources_contributed(env, monkeypatch):
    """`--source pytrends` must not wipe the manual CSV's contribution."""
    write_csv(env, "title,year,slope,saturation\nHeat,1995,70,0.3\n")
    refresh(env)
    before = next(r for r in load_ranking() if r.slug == "heat-1995")

    from clipforge.adapters.pytrends_source import PytrendsAdapter

    env.trends.sources.pytrends = True
    monkeypatch.setattr(
        PytrendsAdapter,
        "fetch",
        lambda self: [RawSignal("Heat", "pytrends", 1995, value=40, slope=55)],
    )
    refresh(env, only=["pytrends"])

    after = next(r for r in load_ranking() if r.slug == "heat-1995")
    assert set(after.sources) == {"manual_csv", "pytrends"}
    assert after.saturation == pytest.approx(before.saturation)   # the CSV value survived
    assert "manual_csv" in after.detail


def test_stale_stored_signals_are_not_carried_forever(env, monkeypatch):
    """Old readings expire: past the merge window they stop propping a title up."""
    write_csv(env, "title,year,slope,saturation\nHeat,1995,70,0.3\n")
    refresh(env)

    from clipforge.adapters.pytrends_source import PytrendsAdapter

    env.trends.sources.pytrends = True
    monkeypatch.setattr(
        PytrendsAdapter,
        "fetch",
        lambda self: [RawSignal("Zodiac", "pytrends", 2007, slope=60)],
    )
    ranked = refresh(env, only=["pytrends"], merge_stored_hours=0.0)
    assert [r.title for r in ranked] == ["Zodiac"]   # Heat's stored signal was too old to carry
