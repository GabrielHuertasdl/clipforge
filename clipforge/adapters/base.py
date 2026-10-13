"""Adapter interface shared by every trend source."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..logging_utils import get_logger


def slugify(title: str, year: int | None = None) -> str:
    """Stable key for a film across sources ('El Padrino', 1972 -> 'el-padrino-1972')."""
    text = unicodedata.normalize("NFKD", title.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return f"{text}-{year}" if year else text


@dataclass(slots=True)
class RawSignal:
    """One reading about one film from one source.

    ``slope`` is what actually matters: a film climbing from 20 to 45 beats one flat at 80.
    Adapters normalise it to 0-100. ``value`` keeps the absolute level for context, and
    ``saturation`` (0-1) is how crowded the clip space already is, when the source knows.
    """

    title: str
    source: str
    year: int | None = None
    value: float = 0.0
    slope: float = 0.0
    saturation: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return slugify(self.title, self.year)


class TrendAdapter:
    """Base class. Subclasses only implement ``fetch``."""

    name: str = "base"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log = get_logger(f"trends.{self.name}")

    def enabled(self) -> bool:
        return bool(getattr(self.cfg.trends.sources, self.name, False))

    def fetch(self) -> list[RawSignal]:  # pragma: no cover - interface
        raise NotImplementedError

    def safe_fetch(self) -> list[RawSignal]:
        """Never raises: a broken source must not take the run down."""
        if not self.enabled():
            self.log.debug("disabled")
            return []
        try:
            signals = self.fetch()
        except Exception as exc:
            self.log.warning("source unavailable (%s) — skipping", exc)
            return []
        self.log.info("%d signal(s)", len(signals))
        return signals
