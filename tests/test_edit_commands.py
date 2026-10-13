"""ffmpeg command construction — built as pure functions so they can be asserted on."""

from __future__ import annotations

import pytest

from clipforge.audio_features import AudioFeatures, DramaticPeak
from clipforge.config import Config
from clipforge.crop import CropPlan, build_crop_expr, limit_speed, simplify, smooth
from clipforge.edit import (
    RehookEvent,
    build_audio_filters,
    build_cold_open_command,
    build_main_command,
    build_video_filters,
    build_zoom_expr,
    escape_drawtext,
    plan_rehooks,
    wrap_text,
)
from clipforge.retiming import build_retime_map
from clipforge.scoring import ScoredSegment


@pytest.fixture
def cfg():
    return Config()


# --------------------------------------------------------------------------- audio (f)


def test_audio_chain_has_loudnorm_and_aggressive_compression(cfg):
    chain = build_audio_filters(cfg, None)
    assert "loudnorm=I=-14" in chain
    assert "acompressor=" in chain
    assert "ratio=6" in chain              # high aggressiveness keeps the configured ratio
    assert "equalizer=f=2200" in chain     # dialogue lift
    assert "highpass=f=80" in chain


def test_lower_aggressiveness_softens_the_ratio(cfg):
    cfg.edit.audio_chain.compressor.aggressiveness = "low"
    assert "ratio=2" in build_audio_filters(cfg, None)


def test_audio_select_is_applied_before_the_chain(cfg):
    chain = build_audio_filters(cfg, "between(t,0,10)")
    assert chain.startswith("aselect='between(t,0,10)',asetpts=N/SR/TB")
    assert chain.index("aselect") < chain.index("loudnorm")


def test_compressor_can_be_disabled(cfg):
    cfg.edit.audio_chain.compressor.enabled = False
    assert "acompressor" not in build_audio_filters(cfg, None)


# --------------------------------------------------------------------------- video chain


def test_video_chain_order_is_trim_crop_scale_subs(cfg, tmp_path):
    ass = tmp_path / "s.ass"
    ass.write_text("x", encoding="utf-8")
    parts = build_video_filters(
        cfg, "between(t,0,10)", CropPlan([(0.0, 0.5)]), 1920, 1080, ass, []
    )
    chain = ",".join(parts)
    assert chain.index("select=") < chain.index("crop=")
    assert chain.index("crop=") < chain.index("scale=1080:1920")
    assert chain.index("scale=1080:1920") < chain.index("ass=")


def test_crop_window_is_nine_by_sixteen(cfg):
    parts = build_video_filters(cfg, None, CropPlan([(0.0, 0.5)]), 1920, 1080, None, [])
    crop = next(p for p in parts if p.startswith("crop="))
    assert "w=608" in crop      # 1080 * 1080/1920 = 607.5 -> even 608
    assert "h=1080" in crop
    assert "x='" in crop  # x is an expression, so it pans (crop evaluates it per frame)


def test_subtitles_can_be_disabled(cfg):
    parts = build_video_filters(cfg, None, CropPlan([(0.0, 0.5)]), 1920, 1080, None, [])
    assert not any(p.startswith("ass=") for p in parts)


def test_text_rehook_becomes_a_time_gated_drawtext(cfg):
    events = [RehookEvent(t=30.0, kind="text", text="fíjate en el reflejo", duration=1.6)]
    parts = build_video_filters(cfg, None, CropPlan([(0.0, 0.5)]), 1920, 1080, None, events)
    draw = next(p for p in parts if p.startswith("drawtext="))
    assert "enable='between(t,30.000,31.600)'" in draw
    assert "reflejo" in draw


def test_rehook_text_sits_below_the_top_safe_zone(cfg):
    events = [RehookEvent(t=10.0, kind="text", text="hola")]
    parts = build_video_filters(cfg, None, CropPlan([(0.0, 0.5)]), 1920, 1080, None, events)
    draw = next(p for p in parts if p.startswith("drawtext="))
    assert f"y={cfg.edit.subtitles.safe_top_px + 60}" in draw


def test_continuity_text_is_gated_to_the_end(cfg):
    parts = build_video_filters(
        cfg, None, CropPlan([(0.0, 0.5)]), 1920, 1080, None, [], continuity=(170.0, "sigue")
    )
    draw = [p for p in parts if p.startswith("drawtext=")]
    assert any("enable='gte(t,170.000)'" in p for p in draw)


# --------------------------------------------------------------------------- zoom punches (e)


def test_zoom_expression_covers_every_zoom_rehook(cfg):
    events = [RehookEvent(10.0, "zoom"), RehookEvent(40.0, "zoom"), RehookEvent(20.0, "text")]
    expr = build_zoom_expr(events, cfg)
    assert expr.count("between(in_time,") == 2
    assert "1.0400" in expr  # 4% punch


def test_no_zoom_filter_without_zoom_rehooks(cfg):
    assert build_zoom_expr([RehookEvent(10.0, "text")], cfg) is None


# --------------------------------------------------------------------------- crop maths (c)


def test_crop_expression_is_clamped_inside_the_frame():
    expr = build_crop_expr([(0.0, 0.0), (5.0, 1.0)], in_w=1920, crop_w=608)
    assert "-" not in expr.replace("(t-", "").replace("e-", "")  # no negative x values
    assert "1312" in expr  # 1920-608, the right-hand limit


def test_single_keyframe_is_a_constant():
    assert build_crop_expr([(0.0, 0.5)], 1920, 608) == "656.0"


def test_no_keyframes_falls_back_to_centre():
    assert build_crop_expr([], 1920, 608) == "656"


def test_smoothing_removes_jitter():
    noisy = [0.5, 0.9, 0.5, 0.1, 0.5, 0.9, 0.5, 0.1] * 4
    out = smooth(noisy, 9)
    assert max(out) - min(out) < max(noisy) - min(noisy)


def test_speed_limit_caps_travel():
    times = [0.0, 1.0, 2.0]
    out = limit_speed(times, [0.0, 1.0, 0.0], max_per_s=108, frame_width=1080)  # 0.1/s
    assert out[1] == pytest.approx(0.1)
    assert out[2] == pytest.approx(0.0, abs=1e-9)


def test_simplify_keeps_endpoints_and_drops_noise():
    times = [float(i) for i in range(20)]
    values = [0.5 + (0.001 if i % 2 else 0.0) for i in range(20)]
    keys = simplify(times, values)
    assert keys[0][0] == 0.0 and keys[-1][0] == 19.0
    assert len(keys) == 2  # the sub-threshold wobble is discarded


# --------------------------------------------------------------------------- commands


def test_main_command_seeks_before_the_input(cfg, tmp_path):
    cmd = build_main_command(
        cfg, tmp_path / "in.mkv", 120.0, 200.0, ["fps=30"], "loudnorm", tmp_path / "out.mp4"
    )
    assert cmd[cmd.index("-ss") + 1] == "120.000"
    assert cmd.index("-ss") < cmd.index("-i")     # input seeking, not slow output seeking
    assert cmd[cmd.index("-t") + 1] == "200.000"
    assert "-movflags" in cmd and "+faststart" in cmd
    assert cmd[-1].endswith("out.mp4")


def test_cold_open_command_is_short_and_carries_the_hook(cfg, tmp_path):
    cmd = build_cold_open_command(
        cfg, tmp_path / "in.mkv", 812.5, "nadie se fija en su mano", 1920, 1080,
        tmp_path / "cold.mp4",
    )
    assert cmd[cmd.index("-ss") + 1] == "812.500"
    assert cmd[cmd.index("-t") + 1] == "1.500"
    vf = cmd[cmd.index("-vf") + 1]
    assert "drawtext=" in vf and "nadie se fija" in vf
    assert "scale=1080:1920" in vf


def test_drawtext_escaping():
    assert escape_drawtext("a:b,c'd") == r"a\:b\,c’d"


def test_wrap_text_respects_the_line_budget():
    out = wrap_text("nadie se fija en lo que hace con la mano izquierda", 22)
    assert all(len(line) <= 22 for line in out.split("\n"))
    assert out.count("\n") >= 2


# --------------------------------------------------------------------------- re-hook planning


def make_scored(**kw) -> ScoredSegment:
    base = dict(
        segment_id=1, idx=0, start_s=100.0, end_s=300.0, total=70.0, scores={},
        cold_open_timestamp=150.0, hook_text="mira la mano",
        rehook_suggestions=[
            {"t": 100.0 + i * 20, "rel_t": i * 20, "text": f"r{i}"} for i in range(8)
        ],
        suggested_caption="c", suggested_hashtags=[], reason="r",
    )
    base.update(kw)
    return ScoredSegment(**base)


def test_rehooks_alternate_text_and_zoom(cfg):
    retime = build_retime_map(100.0, 300.0, [], cfg)
    features = AudioFeatures(frame_s=1.0, duration_s=400.0)
    events = plan_rehooks(make_scored(), retime, features, cfg, 1.5, 201.5)
    kinds = [e.kind for e in events]
    assert "text" in kinds and "zoom" in kinds
    assert kinds != ["text"] * len(kinds)


def test_zoom_rehooks_snap_to_audio_peaks(cfg):
    retime = build_retime_map(100.0, 300.0, [], cfg)
    features = AudioFeatures(
        frame_s=1.0,
        duration_s=400.0,
        peaks=[DramaticPeak(t=120.4, delta_db=20.0, quiet_s=1.0, level_db=-6.0)],
    )
    scored = make_scored(rehook_suggestions=[
        {"t": 110.0, "rel_t": 10.0, "text": "a"},
        {"t": 120.0, "rel_t": 20.0, "text": "b"},  # zoom: 0.4 s from the peak -> snaps
    ])
    events = plan_rehooks(scored, retime, features, cfg, 1.5, 201.5)
    zoom = next(e for e in events if e.kind == "zoom")
    assert zoom.t == pytest.approx(21.9)  # 120.4 - 100 + 1.5 cold open


def test_rehooks_skip_the_first_and_last_second(cfg):
    retime = build_retime_map(100.0, 300.0, [], cfg)
    features = AudioFeatures(frame_s=1.0, duration_s=400.0)
    scored = make_scored(rehook_suggestions=[
        {"t": 100.2, "rel_t": 0.2, "text": "muy pronto"},
        {"t": 299.9, "rel_t": 199.9, "text": "muy tarde"},
    ])
    assert plan_rehooks(scored, retime, features, cfg, 1.5, 201.5) == []


def test_rehooks_follow_the_retiming_map(cfg):
    # 10 s of silence removed before the re-hook -> it must move earlier by ~10 s
    retime = build_retime_map(100.0, 300.0, [(110.0, 120.0)], cfg)
    features = AudioFeatures(frame_s=1.0, duration_s=400.0)
    scored = make_scored(rehook_suggestions=[{"t": 200.0, "rel_t": 100.0, "text": "x"}])
    event = plan_rehooks(scored, retime, features, cfg, 0.0, 190.0)[0]
    assert event.t == pytest.approx(100.0 - 9.85, abs=0.01)
