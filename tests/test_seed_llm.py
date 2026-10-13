"""seed_llm: schema validation, monthly cache, and the "don't trust its score" rule."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from clipforge.adapters.seed_llm import SEED_SLOPE, SeedLLMAdapter, SeedResponse, load_seed_reasons
from clipforge.config import Config


@pytest.fixture
def cfg(tmp_path):
    c = Config(project_root=tmp_path)
    c.ensure_dirs()
    c.trends.sources.seed_llm = True
    return c


def sample_response(n: int = 3) -> SeedResponse:
    return SeedResponse.model_validate(
        {
            "titles": [
                {
                    "title": f"Peli {i}",
                    "year": 2000 + i,
                    "reason": f"estreno de secuela numero {i} este mes",
                    "estimated_saturation": 0.1 * (i + 1),
                }
                for i in range(n)
            ]
        }
    )


# --------------------------------------------------------------------------- schema


def test_schema_requires_every_field():
    with pytest.raises(ValidationError):
        SeedResponse.model_validate({"titles": [{"title": "Heat"}]})


def test_saturation_is_bounded():
    with pytest.raises(ValidationError):
        SeedResponse.model_validate(
            {"titles": [{"title": "Heat", "reason": "porque si", "estimated_saturation": 1.7}]}
        )


def test_absurd_years_are_rejected():
    with pytest.raises(ValidationError):
        SeedResponse.model_validate(
            {
                "titles": [
                    {
                        "title": "Heat",
                        "year": 3200,
                        "reason": "porque si claro",
                        "estimated_saturation": 0.3,
                    }
                ]
            }
        )


def test_titles_are_trimmed():
    parsed = SeedResponse.model_validate(
        {
            "titles": [
                {"title": "  Heat  ", "reason": "aniversario redondo", "estimated_saturation": 0.2}
            ]
        }
    )
    assert parsed.titles[0].title == "Heat"


# --------------------------------------------------------------------------- behaviour


def test_seed_score_is_not_the_models_own_confidence(cfg, monkeypatch):
    adapter = SeedLLMAdapter(cfg)
    monkeypatch.setattr(adapter, "ask_model", lambda today: sample_response())
    signals = adapter.fetch(today=date(2026, 9, 15))
    # Every seed enters with the same fixed slope: the measuring sources rank them later.
    assert {s.slope for s in signals} == {SEED_SLOPE}
    # Its saturation estimate IS used, since no other source provides one for a new title.
    assert signals[0].saturation == pytest.approx(0.1)
    assert signals[0].meta["reason"].startswith("estreno")


def test_seed_count_is_respected(cfg, monkeypatch):
    cfg.trends.seed_count = 2
    adapter = SeedLLMAdapter(cfg)
    monkeypatch.setattr(adapter, "ask_model", lambda today: sample_response(5))
    assert len(adapter.fetch(today=date(2026, 9, 15))) == 2


def test_the_model_is_called_once_per_month(cfg, monkeypatch):
    calls: list[date] = []

    def fake_ask(today):
        calls.append(today)
        return sample_response()

    adapter = SeedLLMAdapter(cfg)
    monkeypatch.setattr(adapter, "ask_model", fake_ask)

    adapter.fetch(today=date(2026, 9, 15))
    adapter.fetch(today=date(2026, 9, 28))     # same month -> cache
    assert len(calls) == 1

    adapter.fetch(today=date(2026, 10, 2))     # new month -> one more call
    assert len(calls) == 2


def test_a_second_adapter_instance_reuses_the_cache(cfg, monkeypatch):
    first = SeedLLMAdapter(cfg)
    monkeypatch.setattr(first, "ask_model", lambda today: sample_response())
    first.fetch(today=date(2026, 9, 15))

    second = SeedLLMAdapter(cfg)

    def explode(today):
        raise AssertionError("should have used the cache")

    monkeypatch.setattr(second, "ask_model", explode)
    assert len(second.fetch(today=date(2026, 9, 15))) == 3


def test_corrupt_cache_is_ignored(cfg, monkeypatch):
    adapter = SeedLLMAdapter(cfg)
    adapter.cache_path(date(2026, 9, 15)).write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(adapter, "ask_model", lambda today: sample_response())
    assert len(adapter.fetch(today=date(2026, 9, 15))) == 3


def test_missing_api_key_does_not_break_the_run(cfg, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adapter = SeedLLMAdapter(cfg)
    assert adapter.safe_fetch() == []      # logged and skipped, never raised


def test_reasons_are_readable_from_cache_without_an_api_call(cfg, monkeypatch):
    adapter = SeedLLMAdapter(cfg)
    monkeypatch.setattr(adapter, "ask_model", lambda today: sample_response())
    adapter.fetch(today=date(2026, 9, 15))
    reasons = load_seed_reasons(cfg, date(2026, 9, 15))
    assert reasons["Peli 0"].startswith("estreno")


def test_disabled_source_does_nothing(cfg):
    cfg.trends.sources.seed_llm = False
    assert SeedLLMAdapter(cfg).safe_fetch() == []
