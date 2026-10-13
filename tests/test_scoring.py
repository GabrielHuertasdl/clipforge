"""Scoring tests. Every model call is mocked — the suite never touches the network."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from clipforge.audio_features import AudioFeatures, DramaticPeak
from clipforge.config import Config
from clipforge.scenes import Candidate
from clipforge.scoring import (
    IconicScene,
    IconicSceneMap,
    Scores,
    SegmentVerdict,
    _call_structured,
    _clamp_verdict,
    build_context,
    prioritize,
    score_segment,
    weighted_total,
)
from clipforge.transcribe import Transcript, Word


@pytest.fixture
def cfg():
    return Config()


def make_verdict(**overrides) -> SegmentVerdict:
    data = {
        "scores": {
            "hook_strength": 80,
            "emotional_peak": 70,
            "standalone_clarity": 60,
            "quotability": 50,
            "visual_interest": 90,
            "sustains_3min": 40,
        },
        "cold_open_at_s": 42.0,
        "hook_text": "nadie se fija en lo que hace con la mano",
        "rehook_suggestions": [
            {"at_s": 20.0, "text": "espera al plano siguiente"},
            {"at_s": 60.0, "text": "fíjate en el reflejo"},
        ],
        "suggested_caption": "La escena que nadie mira dos veces",
        "suggested_hashtags": ["cine", "#peliculas"],
        "reason": "Tensión creciente sostenida por el diálogo y un pico de audio claro.",
    }
    data.update(overrides)
    return SegmentVerdict.model_validate(data)


class FakeResponse:
    def __init__(self, parsed):
        self.parsed_output = parsed


class FakeClient:
    """Minimal stand-in for anthropic.Anthropic with a scripted messages.parse()."""

    def __init__(self, results):
        self.results = list(results)
        self.calls: list[dict] = []
        self.messages = self

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


# --------------------------------------------------------------------------- schema


def test_scores_are_bounded():
    with pytest.raises(ValidationError):
        Scores(
            hook_strength=140,
            emotional_peak=10,
            standalone_clarity=10,
            quotability=10,
            visual_interest=10,
            sustains_3min=10,
        )


def test_hashtags_get_normalised():
    v = make_verdict(suggested_hashtags=["cine", "#peliculas", "  ", "##dobles"])
    assert v.suggested_hashtags == ["#cine", "#peliculas", "#dobles"]


def test_missing_field_is_rejected():
    with pytest.raises(ValidationError):
        SegmentVerdict.model_validate({"scores": {}, "hook_text": "x"})


def test_weighted_total_uses_config_weights(cfg):
    scores = dict.fromkeys(
        (
            "hook_strength",
            "emotional_peak",
            "standalone_clarity",
            "quotability",
            "visual_interest",
            "sustains_3min",
        ),
        50,
    )
    assert weighted_total(scores, cfg) == pytest.approx(50.0)

    cfg.scoring.weights.hook_strength = 1.0
    cfg.scoring.weights.emotional_peak = 0.0
    cfg.scoring.weights.standalone_clarity = 0.0
    cfg.scoring.weights.quotability = 0.0
    cfg.scoring.weights.visual_interest = 0.0
    cfg.scoring.weights.sustains_3min = 0.0
    scores["hook_strength"] = 90
    assert weighted_total(scores, cfg) == pytest.approx(90.0)


# --------------------------------------------------------------------------- clamping


def test_clamp_keeps_timestamps_inside_the_segment(cfg):
    v = make_verdict(
        cold_open_at_s=500.0,
        rehook_suggestions=[
            {"at_s": 10.0, "text": "a"},
            {"at_s": 900.0, "text": "fuera de rango"},
            {"at_s": 5.0, "text": "b"},
        ],
    )
    out = _clamp_verdict(v, duration=200.0, cfg=cfg)
    assert out.cold_open_at_s == pytest.approx(198.5)
    assert [r.at_s for r in out.rehook_suggestions] == [5.0, 10.0]


def test_clamp_trims_to_max_rehooks(cfg):
    cfg.scoring.rehooks_max = 3
    v = make_verdict(
        rehook_suggestions=[{"at_s": float(i * 10), "text": f"t{i}"} for i in range(9)]
    )
    assert len(_clamp_verdict(v, 200.0, cfg).rehook_suggestions) == 3


# --------------------------------------------------------------------------- retries


def test_retry_then_success(cfg, monkeypatch):
    monkeypatch.setattr("clipforge.scoring._sleep_backoff", lambda *a, **k: None)
    verdict = make_verdict()
    client = FakeClient([ValueError("bad json"), verdict])
    got, _ = _call_structured(
        client, cfg, system="s", user="u", schema=SegmentVerdict, label="t"
    )
    assert got is verdict
    assert len(client.calls) == 2


def test_gives_up_after_max_retries(cfg, monkeypatch):
    monkeypatch.setattr("clipforge.scoring._sleep_backoff", lambda *a, **k: None)
    cfg.scoring.max_retries = 3
    client = FakeClient([ValueError("bad")] * 3)
    with pytest.raises(RuntimeError, match="giving up after 3"):
        _call_structured(client, cfg, system=None, user="u", schema=SegmentVerdict, label="t")
    assert len(client.calls) == 3


def test_client_error_is_not_retried(cfg, monkeypatch):
    import anthropic

    monkeypatch.setattr("clipforge.scoring._sleep_backoff", lambda *a, **k: None)

    class Boom(anthropic.APIStatusError):
        def __init__(self):
            self.status_code = 400
            Exception.__init__(self, "bad request")

    client = FakeClient([Boom()])
    with pytest.raises(anthropic.APIStatusError):
        _call_structured(client, cfg, system=None, user="u", schema=SegmentVerdict, label="t")
    assert len(client.calls) == 1


# --------------------------------------------------------------------------- priming


def test_prioritize_puts_iconic_zones_first(cfg):
    cands = [
        Candidate(0, 0.0, 200.0, 0, 5, 6),
        Candidate(1, 200.0, 400.0, 6, 11, 6),
        Candidate(2, 3400.0, 3600.0, 12, 18, 7),
    ]
    scene_map = IconicSceneMap(
        recognized=True, scenes=[IconicScene(description="el duelo", approx_minute=58.0)]
    )
    ordered = prioritize(cands, scene_map, cfg)
    assert ordered[0].idx == 2  # 58 min = 3480 s, inside segment 2
    assert ordered[0].priority > 0


def test_unrecognised_film_keeps_chronological_order(cfg):
    cands = [Candidate(i, i * 200.0, (i + 1) * 200.0, i, i, 1) for i in range(4)]
    ordered = prioritize(cands, IconicSceneMap(recognized=False), cfg)
    assert [c.idx for c in ordered] == [0, 1, 2, 3]
    assert all(c.priority == 0 for c in ordered)


# --------------------------------------------------------------------------- context


def make_inputs():
    transcript = Transcript(
        words=[Word(float(i), float(i) + 0.4, f"p{i}") for i in range(100, 160)]
    )
    features = AudioFeatures(
        frame_s=1.0,
        duration_s=600.0,
        rms_db=[-30.0] * 600,
        silence_ratio=[0.2] * 600,
        onset_rate=[1.0] * 600,
        energy_delta=[0.0] * 600,
        peaks=[DramaticPeak(t=150.0, delta_db=20.0, quiet_s=1.0, level_db=-8.0)],
    )
    return transcript, features


def test_context_includes_the_signals_the_prompt_relies_on(cfg):
    transcript, features = make_inputs()
    cand = Candidate(3, 100.0, 300.0, 5, 20, 16)
    ctx = build_context(cand, transcript, features, 6000.0, "Heat", 1995, cfg)
    assert "Heat" in ctx and "1995" in ctx
    assert "palabras_por_segundo" in ctx
    assert "picos_dramaticos_s" in ctx
    assert "50.0" in ctx  # the peak at 150 s is 50 s into the segment
    assert "posicion_relativa_en_la_pelicula" in ctx


def test_context_respects_the_transcript_budget(cfg):
    cfg.scoring.transcript_char_budget = 200
    transcript = Transcript(words=[Word(float(i), float(i) + 0.5, "palabra") for i in range(400)])
    features = AudioFeatures(frame_s=1.0, duration_s=500.0)
    cand = Candidate(0, 0.0, 200.0, 0, 1, 2)
    ctx = build_context(cand, transcript, features, 500.0, "X", None, cfg)
    assert "[...corte...]" in ctx
    assert len(ctx) < 2000


# --------------------------------------------------------------------------- end to end (mocked)


def test_score_segment_maps_timestamps_to_absolute(cfg, monkeypatch):
    monkeypatch.setattr("clipforge.scoring._sleep_backoff", lambda *a, **k: None)
    transcript, features = make_inputs()
    cand = Candidate(3, 100.0, 300.0, 5, 20, 16)
    client = FakeClient([make_verdict()])

    scored = score_segment(cand, 77, transcript, features, 6000.0, "Heat", 1995, cfg, client)

    assert scored.segment_id == 77
    assert scored.cold_open_timestamp == pytest.approx(142.0)  # 100 + 42
    assert [r["t"] for r in scored.rehook_suggestions] == [120.0, 160.0]
    assert [r["rel_t"] for r in scored.rehook_suggestions] == [20.0, 60.0]
    assert scored.total == pytest.approx(weighted_total(scored.scores, cfg))
    assert scored.hook_text.startswith("nadie se fija")


def test_score_segment_sends_the_hook_rules_in_the_system_prompt(cfg, monkeypatch):
    monkeypatch.setattr("clipforge.scoring._sleep_backoff", lambda *a, **k: None)
    transcript, features = make_inputs()
    client = FakeClient([make_verdict()])
    score_segment(
        Candidate(0, 0.0, 200.0, 0, 5, 6), 1, transcript, features, 600.0, "X", None, cfg, client
    )
    system = client.calls[0]["system"]
    assert "hook_text" in system
    assert "nadie se fija" in system  # the good example
    assert client.calls[0]["model"] == cfg.scoring.model
    assert client.calls[0]["output_format"] is SegmentVerdict


def test_request_carries_no_sampling_parameters(cfg, monkeypatch):
    """Models 4.6+ reject temperature/top_p: the call must not send them."""
    monkeypatch.setattr("clipforge.scoring._sleep_backoff", lambda *a, **k: None)
    client = FakeClient([make_verdict()])
    _call_structured(client, cfg, system="s", user="u", schema=SegmentVerdict, label="t")
    sent = client.calls[0]
    assert "temperature" not in sent and "top_p" not in sent and "top_k" not in sent
    assert set(sent) <= {"model", "max_tokens", "messages", "output_format", "system"}


def test_a_bad_call_signature_is_not_retried(cfg, monkeypatch):
    """A TypeError means the code is wrong; burning the backoff ladder helps nobody."""
    monkeypatch.setattr("clipforge.scoring._sleep_backoff", lambda *a, **k: None)
    client = FakeClient([TypeError("parse() got an unexpected keyword argument")])
    with pytest.raises(TypeError):
        _call_structured(client, cfg, system=None, user="u", schema=SegmentVerdict, label="t")
    assert len(client.calls) == 1


def test_workspace_header_is_sent_when_configured(monkeypatch):
    import clipforge.scoring as scoring

    captured = {}

    class FakeAnthropic:
        def __init__(self, api_key, default_headers=None):
            captured["key"] = api_key
            captured["headers"] = default_headers

    import sys

    fake_module = type("M", (), {"Anthropic": FakeAnthropic})
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_123")
    scoring.get_client()
    assert captured["headers"] == {"anthropic-workspace-id": "wrkspc_123"}

    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID")
    scoring.get_client()
    assert captured["headers"] is None


def test_missing_key_raises_a_readable_error(monkeypatch):
    import clipforge.scoring as scoring

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        scoring.get_client()
