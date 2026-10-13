"""Manual CSV — the fallback that always works, and the first one implemented.

``input/manual/trending.csv``::

    title,year,slope,saturation,note
    Heat,1995,70,0.3,llega a Netflix el 12
    El Padrino,1972,40,0.8,

Only ``title`` is required. ``slope`` is 0-100 (how fast interest is climbing),
``saturation`` is 0-1 (how many clips already exist); both fall back to config defaults.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .base import RawSignal, TrendAdapter


def _to_float(value: str | None, default: float) -> float:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return default


def _to_int(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None


class ManualCSVAdapter(TrendAdapter):
    name = "manual_csv"

    def csv_path(self) -> Path:
        return self.cfg.resolve(self.cfg.trends.manual_csv)

    def fetch(self) -> list[RawSignal]:
        path = self.csv_path()
        if not path.exists():
            self.log.info("no manual CSV at %s", path)
            return []

        signals: list[RawSignal] = []
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                keys = {(k or "").strip().lower(): (v or "") for k, v in row.items()}
                title = keys.get("title", "").strip()
                if not title:
                    continue
                signals.append(
                    RawSignal(
                        title=title,
                        source=self.name,
                        year=_to_int(keys.get("year")),
                        value=_to_float(keys.get("value"), 0.0),
                        slope=_to_float(keys.get("slope"), 50.0),
                        saturation=_to_float(
                            keys.get("saturation"), self.cfg.trends.default_saturation
                        ),
                        meta={"note": keys.get("note", "").strip()},
                    )
                )
        return signals
