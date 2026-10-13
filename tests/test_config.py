from pathlib import Path

from clipforge.config import Config, load_config


def test_defaults_are_valid():
    cfg = Config()
    assert cfg.edit.target_w == 1080 and cfg.edit.target_h == 1920
    assert cfg.transcribe.word_timestamps is True


def test_repo_config_loads():
    cfg = load_config(Path("config.yaml"))
    assert cfg.scenes.segment_min_s == 150
    assert cfg.scenes.segment_max_s == 260
    assert cfg.edit.subtitles.safe_bottom_px == 300
    assert cfg.scoring.model == "claude-sonnet-4-6"


def test_partial_yaml_deep_merges(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("scenes:\n  threshold: 40.0\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.scenes.threshold == 40.0
    assert cfg.scenes.segment_max_s == 260  # untouched default survives


def test_weights_normalize():
    w = Config().scoring.weights.normalized()
    assert abs(sum(w.values()) - 1.0) < 1e-9
