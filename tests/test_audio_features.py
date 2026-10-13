import json

import pytest

from clipforge.audio_features import (
    AudioFeatures,
    DramaticPeak,
    SilenceRange,
    detect_dramatic_peaks,
    invert_intervals,
    silence_ranges,
)
from clipforge.config import Config


@pytest.fixture
def cfg():
    return Config()  # quiet_window_s=1, peak_delta_db=8, peak_min_gap_s=4, frame_s=1


def test_peak_needs_a_lull_before_it(cfg):
    # Steady loud audio: no lull, so no dramatic peak.
    peaks = detect_dramatic_peaks([-10.0] * 40, 1.0, cfg)
    assert peaks == []


def test_detects_rise_after_quiet(cfg):
    rms_db = [-45.0] * 20 + [-5.0] * 10 + [-45.0] * 20
    peaks = detect_dramatic_peaks(rms_db, 1.0, cfg)
    assert len(peaks) == 1
    assert peaks[0].t == pytest.approx(20.0)
    assert peaks[0].delta_db >= cfg.audio.peak_delta_db


def test_bursts_collapse_to_the_strongest(cfg):
    # Two rises 2 s apart (below peak_min_gap_s=4): only the loudest survives.
    rms_db = [-50.0] * 15 + [-10.0] + [-50.0] + [-2.0] + [-50.0] * 15
    peaks = detect_dramatic_peaks(rms_db, 1.0, cfg)
    assert len(peaks) == 1
    assert peaks[0].level_db == pytest.approx(-2.0)


def test_small_jump_is_not_a_peak(cfg):
    cfg.audio.peak_delta_db = 20.0
    rms_db = [-30.0] * 20 + [-25.0] * 10 + [-30.0] * 20
    assert detect_dramatic_peaks(rms_db, 1.0, cfg) == []


def test_too_short_signal(cfg):
    assert detect_dramatic_peaks([], 1.0, cfg) == []
    assert detect_dramatic_peaks([-10.0, -5.0], 1.0, cfg) == []


def test_invert_intervals_covers_head_and_tail():
    assert invert_intervals([(2.0, 5.0), (7.0, 9.0)], 12.0) == [(0.0, 2.0), (5.0, 7.0), (9.0, 12.0)]
    assert invert_intervals([], 10.0) == [(0.0, 10.0)]
    assert invert_intervals([(0.0, 10.0)], 10.0) == []


def test_invert_intervals_handles_overlap():
    assert invert_intervals([(0.0, 5.0), (3.0, 8.0)], 10.0) == [(8.0, 10.0)]


def test_silence_ranges_applies_min_duration():
    non_silent = [(0.0, 1.0), (1.2, 3.0), (3.9, 6.0)]
    out = silence_ranges(non_silent, 6.0, min_ms=400)
    # the 200 ms gap is ignored, the 900 ms one is kept
    assert [(s.start, s.end) for s in out] == [(3.0, 3.9)]
    assert out[0].duration == pytest.approx(0.9)


def make_features() -> AudioFeatures:
    return AudioFeatures(
        frame_s=1.0,
        duration_s=20.0,
        rms=[0.1] * 20,
        rms_db=[-40.0] * 10 + [-6.0] * 10,
        silence_ratio=[0.9] * 10 + [0.1] * 10,
        onset_rate=[0.0] * 10 + [2.0] * 10,
        energy_delta=[0.0] * 9 + [34.0] + [0.0] * 10,
        peaks=[DramaticPeak(t=10.0, delta_db=34.0, quiet_s=1.0, level_db=-6.0)],
        silences=[SilenceRange(0.0, 4.0), SilenceRange(8.0, 9.0)],
    )


def test_summary_window():
    f = make_features()
    s = f.summary(0.0, 10.0)
    assert s["mean_db"] == pytest.approx(-40.0)
    assert s["silent_seconds"] == pytest.approx(5.0)
    assert s["silence_fraction"] == pytest.approx(0.5)
    assert s["dramatic_peaks"] == 0
    assert f.summary(10.0, 20.0)["dramatic_peaks"] == 1


def test_silences_between_clips_partial_overlap():
    f = make_features()
    got = f.silences_between(2.0, 8.5)
    assert [(s.start, s.end) for s in got] == [(2.0, 4.0), (8.0, 8.5)]


def test_roundtrip():
    f = make_features()
    back = AudioFeatures.from_dict(json.loads(json.dumps(f.to_dict())))
    assert back.peaks == f.peaks
    assert back.silences == f.silences
    assert back.rms_db == f.rms_db


def test_empty_features_summary_is_safe():
    s = AudioFeatures().summary(0.0, 100.0)
    assert s["mean_db"] == 0.0 and s["dramatic_peaks"] == 0
