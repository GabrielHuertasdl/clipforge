"""Re-timing tests — the most fragile part of the editor, so the most heavily covered."""

from __future__ import annotations

import pytest

from clipforge.config import Config
from clipforge.retiming import build_keeps, build_retime_map, plan_cuts


@pytest.fixture
def cfg():
    c = Config()
    c.edit.dead_air.final_min_s = 0.0     # the tiny fixtures below are not 3-minute clips
    c.edit.dead_air.final_max_s = 100000.0
    return c


def test_no_silences_keeps_everything(cfg):
    m = build_retime_map(100.0, 300.0, [], cfg)
    assert m.output_duration == pytest.approx(200.0)
    assert m.removed_s == 0.0
    assert m.map_time(150.0) == pytest.approx(50.0)


def test_short_silences_are_left_alone(cfg):
    # 300 ms < silence_min_ms (400)
    m = build_retime_map(0.0, 100.0, [(10.0, 10.3)], cfg)
    assert m.cuts == []
    assert m.output_duration == pytest.approx(100.0)


def test_long_silence_is_cut_but_breath_is_kept(cfg):
    # 3 s silence, keep_breath_ms=150 -> 75 ms at each end survive
    m = build_retime_map(0.0, 100.0, [(20.0, 23.0)], cfg)
    assert m.cuts == [(pytest.approx(20.075), pytest.approx(22.925))]
    assert m.output_duration == pytest.approx(100.0 - 2.85)


def test_time_before_and_after_a_cut(cfg):
    m = build_retime_map(0.0, 100.0, [(20.0, 23.0)], cfg)
    assert m.map_time(10.0) == pytest.approx(10.0)          # untouched
    assert m.map_time(30.0) == pytest.approx(30.0 - 2.85)   # shifted by the removed audio


def test_instant_inside_a_cut_snaps_to_the_nearest_edge(cfg):
    m = build_retime_map(0.0, 100.0, [(20.0, 30.0)], cfg)
    edge = m.map_time(20.075)
    assert m.map_time(21.0) == pytest.approx(edge)   # near the start edge
    assert m.map_time(29.0) == pytest.approx(edge)   # near the end edge, same output instant


def test_mapping_is_monotonic_across_many_cuts(cfg):
    silences = [(float(i * 10 + 3), float(i * 10 + 6)) for i in range(18)]
    m = build_retime_map(0.0, 200.0, silences, cfg)
    times = [m.map_time(t / 4) for t in range(0, 800)]
    assert all(b >= a - 1e-9 for a, b in zip(times, times[1:], strict=False))


def test_output_never_exceeds_source(cfg):
    silences = [(float(i * 10 + 2), float(i * 10 + 8)) for i in range(19)]
    m = build_retime_map(0.0, 200.0, silences, cfg)
    assert m.output_duration <= 200.0
    assert m.map_time(200.0) == pytest.approx(m.output_duration)


def test_unmap_roundtrips_outside_cuts(cfg):
    m = build_retime_map(50.0, 250.0, [(80.0, 84.0), (150.0, 156.0)], cfg)
    for src in (60.0, 90.0, 120.0, 200.0, 240.0):
        assert m.unmap_time(m.map_time(src)) == pytest.approx(src, abs=1e-6)


def test_map_span_never_goes_backwards(cfg):
    m = build_retime_map(0.0, 100.0, [(40.0, 50.0)], cfg)
    a, b = m.map_span(42.0, 48.0)  # both inside the same cut
    assert b >= a


def test_cut_budget_is_respected(cfg):
    cfg.edit.dead_air.max_cut_ratio = 0.10  # at most 10 s of a 100 s segment
    silences = [(float(i * 10), float(i * 10 + 4)) for i in range(10)]
    m = build_retime_map(0.0, 100.0, silences, cfg)
    assert m.removed_s <= 10.0 + 1e-6
    assert m.output_duration >= 90.0 - 1e-6


def test_budget_spends_on_the_longest_silences_first(cfg):
    cfg.edit.dead_air.max_cut_ratio = 0.06  # 6 s of a 100 s segment
    silences = [(10.0, 11.0), (30.0, 36.0), (60.0, 61.0)]
    m = build_retime_map(0.0, 100.0, silences, cfg)
    assert len(m.cuts) == 1
    assert m.cuts[0][0] == pytest.approx(30.075)  # the 6 s one won


def test_clip_longer_than_final_max_is_tail_trimmed():
    cfg = Config()
    cfg.edit.dead_air.final_max_s = 180.0
    cfg.edit.dead_air.final_min_s = 170.0
    m = build_retime_map(0.0, 240.0, [(100.0, 103.0)], cfg)
    assert m.output_duration == pytest.approx(180.0)
    assert m.keeps[-1].out_end == pytest.approx(180.0)


def test_disabled_dead_air_is_a_passthrough(cfg):
    cfg.edit.dead_air.enabled = False
    m = build_retime_map(10.0, 210.0, [(50.0, 60.0)], cfg)
    assert m.cuts == []
    assert m.output_duration == pytest.approx(200.0)


def test_plan_cuts_ignores_silence_outside_the_segment(cfg):
    cuts = plan_cuts(100.0, 200.0, [(10.0, 40.0), (150.0, 155.0), (400.0, 420.0)], cfg)
    assert len(cuts) == 1
    assert 150.0 < cuts[0][0] < cuts[0][1] < 155.0


def test_build_keeps_merges_overlapping_cuts():
    keeps = build_keeps(0.0, 100.0, [(10.0, 30.0), (20.0, 40.0)])
    assert [(k.src_start, k.src_end) for k in keeps] == [(0.0, 10.0), (40.0, 100.0)]
    assert keeps[1].out_start == pytest.approx(10.0)


def test_select_filter_lists_every_kept_range(cfg):
    m = build_retime_map(0.0, 100.0, [(20.0, 30.0), (60.0, 70.0)], cfg)
    expr = m.select_filter()
    assert expr.count("between(t,") == 3
    assert "+" in expr


def test_empty_map_is_safe():
    from clipforge.retiming import RetimeMap

    m = RetimeMap(segment_start=0.0, segment_end=0.0)
    assert m.map_time(5.0) == 0.0
    assert m.unmap_time(5.0) == 0.0
    assert m.select_filter() == "1"


def test_max_output_override_bounds_the_whole_clip():
    cfg = Config()
    cfg.edit.dead_air.final_max_s = 200.0
    m = build_retime_map(0.0, 260.0, [], cfg, max_output_s=198.5)
    assert m.output_duration == pytest.approx(198.5)
