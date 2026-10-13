import numpy as np
import pytest

from clipforge.config import Config
from clipforge.scenes import (
    Candidate,
    LumaSamples,
    Shot,
    annotate_candidates,
    dark_ratio_for,
    group_shots,
)


def make_shots(durations, start=0.0):
    shots, t = [], start
    for i, d in enumerate(durations):
        shots.append(Shot(i, t, t + d))
        t += d
    return shots


@pytest.fixture
def cfg():
    return Config()  # segment_min_s=150, max=260, target=200


def test_boundaries_always_fall_on_shot_edges(cfg):
    shots = make_shots([7.0] * 120)
    edges = {round(s.start_s, 6) for s in shots} | {round(s.end_s, 6) for s in shots}
    for c in group_shots(shots, cfg):
        assert round(c.start_s, 6) in edges
        assert round(c.end_s, 6) in edges


def test_kept_segments_respect_min_max(cfg):
    shots = make_shots([5.0] * 200)
    for c in group_shots(shots, cfg):
        if c.discarded:
            continue
        assert cfg.scenes.segment_min_s <= c.duration_s <= cfg.scenes.segment_max_s


def test_segments_do_not_overlap_and_are_ordered(cfg):
    shots = make_shots([3.0, 40.0, 90.0, 12.0, 70.0, 55.0, 30.0, 100.0, 20.0] * 4)
    cands = group_shots(shots, cfg)
    for a, b in zip(cands, cands[1:], strict=False):
        assert a.end_s <= b.start_s
        assert a.shot_end < b.shot_start


def test_shot_longer_than_max_is_flagged_not_split(cfg):
    shots = make_shots([400.0, 60.0, 60.0, 60.0])
    cands = group_shots(shots, cfg)
    assert cands[0].shot_count == 1
    assert cands[0].discarded and "too long" in cands[0].discard_reason


def test_trailing_leftover_is_discarded_not_padded(cfg):
    # 220s total: one 200s segment plus a 20s tail that must not be inflated into a segment.
    shots = make_shots([10.0] * 22)
    cands = group_shots(shots, cfg)
    tail = cands[-1]
    assert tail.discarded and "too short" in tail.discard_reason


def test_empty_input(cfg):
    assert group_shots([], cfg) == []


def test_dark_ratio_detects_black_window():
    luma = LumaSamples(fps=1.0, values=[5.0] * 100 + [120.0] * 100)
    dark, mean = dark_ratio_for(Candidate(0, 0, 100, 0, 0, 1), luma, threshold=18.0)
    assert dark == pytest.approx(1.0)
    assert mean == pytest.approx(5.0)
    dark2, _ = dark_ratio_for(Candidate(1, 100, 200, 0, 0, 1), luma, threshold=18.0)
    assert dark2 == pytest.approx(0.0)


def test_annotate_discards_credits_and_black(cfg):
    duration = 1000.0
    luma = LumaSamples(fps=1.0, values=list(np.full(1000, 120.0)))
    # black stretch 300-500
    for i in range(300, 500):
        luma.values[i] = 2.0
    cands = [
        Candidate(0, 10.0, 170.0, 0, 3, 4),      # fine
        Candidate(1, 300.0, 460.0, 4, 8, 5),     # black
        Candidate(2, 950.0, 1000.0, 9, 12, 4),   # tail/credits zone
        Candidate(3, 0.0, 15.0, 13, 14, 2),      # head/logos zone
    ]
    out = annotate_candidates(cands, luma, duration, cfg)
    assert not out[0].discarded
    assert out[1].discarded and "black" in out[1].discard_reason
    assert out[2].discarded and "credits" in out[2].discard_reason
    assert out[3].discarded and "logos" in out[3].discard_reason


def test_annotate_without_luma_samples_keeps_segments(cfg):
    cands = [Candidate(0, 100.0, 260.0, 0, 5, 6)]
    out = annotate_candidates(cands, LumaSamples(fps=0.0, values=[]), 1000.0, cfg)
    assert not out[0].discarded
