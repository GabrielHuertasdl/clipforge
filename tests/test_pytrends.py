"""pytrends hardening: retries, cache, candidate discovery. No network."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pytest

from clipforge.adapters.pytrends_source import PytrendsAdapter, normalised_slope
from clipforge.config import Config
from clipforge.db import TrendTitle, init_engine, session_scope


@pytest.fixture
def cfg(tmp_path):
    c = Config(project_root=tmp_path)
    c.ensure_dirs()
    c.trends.sources.pytrends = True
    init_engine(tmp_path / "pt.db")
    return c


class FakeFrame:
    """Stands in for the pandas frame pytrends returns."""

    def __init__(self, data: dict[str, list[float]]):
        self.data = data
        self.empty = not data

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        return FakeSeries(self.data[key])


class FakeSeries:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeClient:
    """Scripted pytrends client: each entry is a frame or an exception to raise."""

    def __init__(self, results):
        self.results = list(results)
        self.payloads: list[dict] = []

    def build_payload(self, keywords, timeframe, geo, cat):
        self.payloads.append(
            {"keywords": keywords, "timeframe": timeframe, "geo": geo, "cat": cat}
        )

    def interest_over_time(self):
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def patch_client(adapter, client, monkeypatch):
    monkeypatch.setattr(adapter, "_client", lambda: client)
    monkeypatch.setattr("clipforge.adapters.pytrends_source.time.sleep", lambda *_: None)


# --------------------------------------------------------------------------- slope


def test_slope_prefers_the_climb_over_the_volume():
    assert normalised_slope([20, 25, 30, 35, 40, 45]) > normalised_slope([80] * 6)


def test_slope_is_a_linear_regression_not_an_average():
    # Same mean, opposite direction: the slope must separate them.
    rising = normalised_slope([10, 20, 30, 40, 50])
    falling = normalised_slope([50, 40, 30, 20, 10])
    assert rising > 50 > falling


def test_slope_bounds():
    assert 0.0 <= normalised_slope([0, 100] * 5) <= 100.0
    assert normalised_slope([]) == 0.0 and normalised_slope([7]) == 0.0


# --------------------------------------------------------------------------- querying


def test_queries_spain_with_a_seven_day_window(cfg, monkeypatch):
    adapter = PytrendsAdapter(cfg)
    client = FakeClient([FakeFrame({"Heat": [10, 20, 30, 40]})])
    patch_client(adapter, client, monkeypatch)
    monkeypatch.setattr(adapter, "candidate_titles", lambda: [("Heat", 1995)])

    signals = adapter.fetch()
    assert client.payloads[0]["geo"] == "ES"
    assert client.payloads[0]["timeframe"] == "now 7-d"
    assert signals[0].meta["geo"] == "ES"
    assert signals[0].slope > 50


def test_titles_are_queried_in_batches_of_five(cfg, monkeypatch):
    adapter = PytrendsAdapter(cfg)
    client = FakeClient([FakeFrame({}), FakeFrame({})])
    patch_client(adapter, client, monkeypatch)
    monkeypatch.setattr(
        adapter, "candidate_titles", lambda: [(f"Peli {i}", 2000 + i) for i in range(7)]
    )
    adapter.fetch()
    assert [len(p["keywords"]) for p in client.payloads] == [5, 2]


# --------------------------------------------------------------------------- resilience


def test_retries_then_succeeds(cfg, monkeypatch):
    adapter = PytrendsAdapter(cfg)
    client = FakeClient([RuntimeError("429 too many requests"), FakeFrame({"Heat": [5, 9, 14]})])
    patch_client(adapter, client, monkeypatch)
    monkeypatch.setattr(adapter, "candidate_titles", lambda: [("Heat", 1995)])

    signals = adapter.fetch()
    assert len(signals) == 1
    assert len(client.payloads) == 2   # it really retried


def test_permanent_failure_returns_empty_without_raising(cfg, monkeypatch):
    cfg.scoring.max_retries = 2
    adapter = PytrendsAdapter(cfg)
    client = FakeClient([RuntimeError("429")] * 2)
    patch_client(adapter, client, monkeypatch)
    monkeypatch.setattr(adapter, "candidate_titles", lambda: [("Heat", 1995)])

    assert adapter.fetch() == []


def test_one_bad_batch_does_not_lose_the_good_one(cfg, monkeypatch):
    cfg.scoring.max_retries = 1
    adapter = PytrendsAdapter(cfg)
    client = FakeClient([RuntimeError("boom"), FakeFrame({"Peli 5": [1, 4, 9]})])
    patch_client(adapter, client, monkeypatch)
    monkeypatch.setattr(
        adapter, "candidate_titles", lambda: [(f"Peli {i}", None) for i in range(6)]
    )
    signals = adapter.fetch()
    assert [s.title for s in signals] == ["Peli 5"]


def test_a_crash_never_escapes_safe_fetch(cfg, monkeypatch):
    adapter = PytrendsAdapter(cfg)

    def explode():
        raise ImportError("pytrends is not installed")

    monkeypatch.setattr(adapter, "fetch", explode)
    assert adapter.safe_fetch() == []


def test_disabled_source_does_nothing(cfg):
    cfg.trends.sources.pytrends = False
    assert PytrendsAdapter(cfg).safe_fetch() == []


# --------------------------------------------------------------------------- cache


def test_results_are_cached_for_twelve_hours(cfg, monkeypatch):
    adapter = PytrendsAdapter(cfg)
    client = FakeClient([FakeFrame({"Heat": [10, 20, 30]})])
    patch_client(adapter, client, monkeypatch)
    monkeypatch.setattr(adapter, "candidate_titles", lambda: [("Heat", 1995)])

    first = adapter.fetch()
    assert len(client.payloads) == 1

    second = PytrendsAdapter(cfg)
    patch_client(second, FakeClient([]), monkeypatch)   # would explode if it queried
    cached = second.fetch()
    assert [s.title for s in cached] == [s.title for s in first]
    assert cached[0].slope == first[0].slope


def test_stale_cache_is_ignored(cfg, monkeypatch):
    adapter = PytrendsAdapter(cfg)
    stale = datetime.now(UTC) - timedelta(hours=cfg.trends.cache_ttl_hours + 1)
    adapter.cache_path().write_text(
        json.dumps({"fetched_at": stale.isoformat(), "signals": [{"title": "Vieja"}]}),
        encoding="utf-8",
    )
    client = FakeClient([FakeFrame({"Heat": [1, 2, 3]})])
    patch_client(adapter, client, monkeypatch)
    monkeypatch.setattr(adapter, "candidate_titles", lambda: [("Heat", 1995)])

    assert [s.title for s in adapter.fetch()] == ["Heat"]


def test_corrupt_cache_is_ignored(cfg, monkeypatch):
    adapter = PytrendsAdapter(cfg)
    adapter.cache_path().write_text("{not json", encoding="utf-8")
    assert adapter.read_cache() is None


def test_failed_run_writes_no_cache(cfg, monkeypatch):
    cfg.scoring.max_retries = 1
    adapter = PytrendsAdapter(cfg)
    patch_client(adapter, FakeClient([RuntimeError("429")]), monkeypatch)
    monkeypatch.setattr(adapter, "candidate_titles", lambda: [("Heat", 1995)])
    adapter.fetch()
    assert not adapter.cache_path().exists()


# --------------------------------------------------------------------------- candidates


def test_candidates_come_from_the_database(cfg):
    with session_scope() as s:
        s.add(TrendTitle(slug="heat-1995", title="Heat", year=1995, status="new"))
        s.add(TrendTitle(slug="vieja-1980", title="Vieja", year=1980, status="processed"))
        s.add(TrendTitle(slug="fuera-1990", title="Fuera", year=1990, status="ignored"))

    titles = dict(PytrendsAdapter(cfg).candidate_titles())
    assert "Heat" in titles
    assert "Vieja" not in titles   # already processed: no point spending a query on it
    assert "Fuera" not in titles


def test_candidates_include_upcoming_anniversaries(cfg):
    today = date.today()
    release = date(today.year - 25, today.month, min(today.day, 28))
    path = cfg.resolve(cfg.trends.release_dates_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"title,year,release_date,country\nAniversaria,{release.year},{release.isoformat()},ES\n",
        encoding="utf-8",
    )
    assert "Aniversaria" in dict(PytrendsAdapter(cfg).candidate_titles())


def test_candidates_include_the_manual_csv(cfg):
    path = cfg.resolve(cfg.trends.manual_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("title,year\nManualona,1999\n", encoding="utf-8")
    assert "Manualona" in dict(PytrendsAdapter(cfg).candidate_titles())


def test_no_candidates_means_no_query(cfg, monkeypatch):
    adapter = PytrendsAdapter(cfg)
    client = FakeClient([])
    patch_client(adapter, client, monkeypatch)
    monkeypatch.setattr(adapter, "candidate_titles", list)
    assert adapter.fetch() == []
    assert client.payloads == []


# --------------------------------------------------------------------------- slope scaling


def test_slope_scale_is_independent_of_sample_count():
    """A 7-day query returns 169 hourly points; the scale must not depend on that."""
    daily = [20, 25, 30, 35, 40, 45, 50]
    hourly = [20 + (30 * i / 168) for i in range(169)]   # same 20 -> 50 climb
    assert abs(normalised_slope(daily) - normalised_slope(hourly)) < 2.0


def test_flat_series_sits_at_fifty():
    assert normalised_slope([80] * 50) == 50.0


def test_dead_series_are_damped_not_neutral():
    """Nobody searching must not tie with a real climb."""
    assert normalised_slope([0] * 100) == 0.0
    assert normalised_slope([0, 1, 0, 2, 0, 1]) < 30.0
    assert normalised_slope([0, 0, 0, 0, 3]) < normalised_slope([20, 30, 40, 50, 60])
