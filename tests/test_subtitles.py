from __future__ import annotations

import pytest

from clipforge.config import Config
from clipforge.retiming import build_retime_map
from clipforge.subtitles import (
    SubtitleWord,
    build_ass,
    group_words,
    render_line,
    vertical_margin,
    words_to_output,
)
from clipforge.transcribe import Word


@pytest.fixture
def cfg():
    return Config()


def test_lines_never_exceed_the_word_limit(cfg):
    words = [SubtitleWord(float(i), float(i) + 0.4, f"w{i}") for i in range(20)]
    for line in group_words(words, cfg.edit.subtitles.max_words_per_line):
        assert len(line) <= 3


def test_a_long_pause_starts_a_new_line():
    words = [SubtitleWord(0.0, 0.4, "a"), SubtitleWord(0.4, 0.8, "b"), SubtitleWord(5.0, 5.4, "c")]
    lines = group_words(words, 3)
    assert len(lines) == 2


def test_active_word_is_highlighted(cfg):
    line = [SubtitleWord(0, 1, "nadie"), SubtitleWord(1, 2, "se"), SubtitleWord(2, 3, "fija")]
    text = render_line(line, 1, cfg)
    assert cfg.edit.subtitles.active_color in text
    assert text.count(cfg.edit.subtitles.active_color) == 1
    assert "SE" in text  # uppercase is on by default


def test_subtitles_stay_out_of_the_tiktok_ui_zones(cfg):
    margin = vertical_margin(cfg)
    # MarginV is measured from the bottom; it must clear the bottom safe band...
    assert margin >= cfg.edit.subtitles.safe_bottom_px
    # ...and leave the top band free too.
    assert margin <= cfg.edit.target_h - cfg.edit.subtitles.safe_top_px


def test_anchor_moves_the_block_inside_the_safe_zone(cfg):
    cfg.edit.subtitles.vertical_anchor = 0.0
    top_anchor = vertical_margin(cfg)
    cfg.edit.subtitles.vertical_anchor = 1.0
    bottom_anchor = vertical_margin(cfg)
    assert bottom_anchor == cfg.edit.subtitles.safe_bottom_px
    assert top_anchor > bottom_anchor


def test_ass_document_structure(cfg):
    words = [SubtitleWord(float(i), float(i) + 0.5, f"palabra{i}") for i in range(6)]
    doc = build_ass(words, cfg, clip_duration=10.0)
    assert "[V4+ Styles]" in doc and "[Events]" in doc
    assert doc.count("Dialogue:") == 6  # one event per word
    assert f"PlayResY: {cfg.edit.target_h}" in doc


def test_events_never_run_past_the_clip(cfg):
    words = [SubtitleWord(9.5, 12.0, "final")]
    doc = build_ass(words, cfg, clip_duration=10.0)
    assert "0:00:12" not in doc
    assert "0:00:10.00" in doc


def test_disabled_subtitles_produce_no_events(cfg):
    cfg.edit.subtitles.enabled = False
    doc = build_ass([SubtitleWord(0, 1, "x")], cfg, 10.0)
    assert "Dialogue:" not in doc


def test_words_are_remapped_after_dead_air_trimming(cfg):
    retime = build_retime_map(100.0, 300.0, [(110.0, 120.0)], cfg)
    words = [
        Word(105.0, 105.5, "antes"),
        Word(115.0, 115.4, "dentro"),
        Word(200.0, 200.6, "despues"),
    ]
    out = words_to_output(words, retime, offset_s=1.5, clip_duration=200.0)
    texts = [w.text for w in out]
    assert "dentro" not in texts             # it lived inside a removed silence
    assert out[0].start == pytest.approx(6.5)  # 105 - 100 + 1.5 cold open
    assert out[1].start == pytest.approx(200.0 - 100.0 - 9.85 + 1.5, abs=0.01)


def test_words_past_the_clip_are_dropped(cfg):
    retime = build_retime_map(0.0, 100.0, [], cfg)
    out = words_to_output([Word(95.0, 95.5, "x")], retime, 0.0, clip_duration=90.0)
    assert out == []


def test_lines_do_not_overlap_each_other(cfg):
    words = [SubtitleWord(float(i) * 0.5, float(i) * 0.5 + 0.4, f"w{i}") for i in range(6)]
    doc = build_ass(words, cfg, clip_duration=10.0)
    events = [line for line in doc.splitlines() if line.startswith("Dialogue:")]
    # the last word of line 1 must end no later than the first word of line 2 (1.5 s)
    last_of_first_line = events[2]
    assert "0:00:01.50" in last_of_first_line
