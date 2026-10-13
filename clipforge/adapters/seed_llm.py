"""LLM seed — where new candidate titles come from.

Without this adapter nothing new ever enters the system: pytrends needs a list to query and
anniversaries only knows the films in your CSV. This one asks Claude for films likely to have
TikTok demand in Spain this month, with the reason (sequel out, anniversary, actor in the
news, announced remake, rediscovery).

Two deliberate limits:

* **Its numbers are not trusted.** The model's ``estimated_saturation`` is used, but the
  slope it gets is a fixed, deliberately mid-range seed value: these titles are *candidates*,
  and pytrends and anniversaries are what actually score them on the next refresh.
* **Cached per month.** One call a month, not one per refresh.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .base import RawSignal, TrendAdapter

# Seeds enter mid-table: high enough to be queried by pytrends next run, low enough that a
# real measured climb overtakes them.
SEED_SLOPE = 45.0

PROMPT = """\
Estamos en {month_name} de {year}, en España.

Dame {count} películas que probablemente tengan demanda de clips cortos en TikTok/Reels en \
España ESTE MES. Para cada una, la razón concreta por la que ahora y no en otro momento:
estreno de secuela o precuela, serie derivada, aniversario redondo, un actor o director en \
las noticias, remake anunciado, llegada a un catálogo de streaming en España, o \
redescubrimiento (una película antigua que ha vuelto a circular).

Criterios:
- Cine comercial con escenas reconocibles, no cine de autor oscuro sin momentos memorables.
- Que un espectador español de 16-35 años reconozca el título o el momento.
- Evita las obviedades permanentes (El Padrino, Pulp Fiction) salvo que haya un motivo real \
  ESTE MES.
- Mezcla décadas: no todo estrenos recientes.

Para cada película indica estimated_saturation entre 0 y 1: cuántos clips crees que ya \
circulan de ella (0 = nadie ha hecho clips, 1 = saturadísima). Una película con motivo fuerte \
y saturación baja es lo más valioso.
"""


class SeedTitle(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    year: int | None = Field(default=None, ge=1900, le=2100)
    reason: str = Field(min_length=5, max_length=400)
    estimated_saturation: float = Field(ge=0.0, le=1.0)

    @field_validator("title")
    @classmethod
    def _clean_title(cls, value: str) -> str:
        return value.strip()


class SeedResponse(BaseModel):
    titles: list[SeedTitle]


class SeedLLMAdapter(TrendAdapter):
    name = "seed_llm"

    # ---- monthly cache

    def cache_path(self, today: date | None = None) -> Path:
        today = today or date.today()
        directory = self.cfg.resolve(self.cfg.paths.cache_dir) / "trends"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"seed_llm.{today:%Y-%m}.json"

    def read_cache(self, today: date | None = None) -> SeedResponse | None:
        path = self.cache_path(today)
        if not path.exists():
            return None
        try:
            return SeedResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            self.log.warning("ignoring corrupt seed cache %s", path.name)
            return None

    # ---- querying

    def ask_model(self, today: date) -> SeedResponse:
        from ..scoring import _call_structured, get_client

        month_names = [
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
        ]
        prompt = PROMPT.format(
            month_name=month_names[today.month - 1],
            year=today.year,
            count=self.cfg.trends.seed_count,
        )
        response, _ = _call_structured(
            get_client(),
            self.cfg,
            system=None,
            user=prompt,
            schema=SeedResponse,
            label="seed_llm",
        )
        return response

    def fetch(self, today: date | None = None) -> list[RawSignal]:
        today = today or date.today()
        cached = self.read_cache(today)
        if cached is not None:
            self.log.info("using this month's cached seed (%d titles)", len(cached.titles))
            response = cached
        else:
            response = self.ask_model(today)
            self.cache_path(today).write_text(response.model_dump_json(), encoding="utf-8")
            self.log.info("seeded %d candidate title(s) for %s", len(response.titles), today)

        return [
            RawSignal(
                title=item.title,
                source=self.name,
                year=item.year,
                value=0.0,
                # Deliberately NOT derived from the model's own confidence: seeds are
                # candidates, and the measuring sources rank them on the next pass.
                slope=SEED_SLOPE,
                saturation=item.estimated_saturation,
                meta={"reason": item.reason, "seeded_for": f"{today:%Y-%m}"},
            )
            for item in response.titles[: self.cfg.trends.seed_count]
        ]


def load_seed_reasons(cfg, today: date | None = None) -> dict[str, str]:
    """Reason text per title, for the dashboard. Reads the cache only — no API call."""
    adapter = SeedLLMAdapter(cfg)
    cached = adapter.read_cache(today)
    if cached is None:
        return {}
    return {item.title: item.reason for item in cached.titles}
