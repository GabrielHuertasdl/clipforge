"""Google Trends via pytrends.

Two things make this adapter awkward and both are handled here:

* **It breaks often.** Google rate-limits hard (429), pytrends changes shape between
  releases, and the endpoint occasionally just returns nothing. Every call is retried with
  exponential backoff and, when it still fails, the adapter logs it and returns an empty
  list — the other sources are unaffected.
* **It needs a list of titles to ask about.** Google Trends has no "what is hot in cinema"
  endpoint we can query blind. The candidates come from what the system already knows:
  every title in ``trend_title`` (whatever source put it there) plus films from the
  release-date CSV with an anniversary coming up. That is what makes the sources feed each
  other — a title seeded once gets a real trend reading on the next refresh.

Results are cached on disk for ``trends.cache_ttl_hours`` so a refresh does not hit Google
again within the window.
"""

from __future__ import annotations

import json
import random
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from .anniversaries import anniversary_signal, load_releases
from .base import RawSignal, TrendAdapter, slugify

BATCH_SIZE = 5          # Google accepts at most 5 keywords per payload
CATEGORY_MOVIES = 34    # Google Trends category: Movies
MIN_INTEREST_PEAK = 5.0  # below this peak the series is noise, not a trend


def normalised_slope(values: list[float]) -> float:
    """Least-squares slope of the series, rescaled to 0-100 (50 = flat).

    The regression runs over the sample index, then the per-sample slope is multiplied by
    the number of intervals to get the **total change across the whole window** in Google
    interest points. That keeps the scale independent of how many samples the API returns —
    a 7-day query comes back as 169 hourly points, and scaling per sample would squash every
    series to "flat".

    A title going 20 -> 45 across the window scores ~62; one flat at 80 scores 50. Series
    with almost no search volume at all are damped towards zero so they cannot tie with a
    real climb.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2
    mean_y = sum(values) / n
    denominator = sum((i - mean_x) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    per_sample = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values)) / denominator
    total_change = per_sample * (n - 1)          # points gained/lost across the window
    score = max(0.0, min(100.0, 50.0 + total_change * 0.5))

    peak = max(values)
    if peak < MIN_INTEREST_PEAK:                 # essentially no one is searching for it
        score *= peak / MIN_INTEREST_PEAK
    return round(score, 2)


class PytrendsAdapter(TrendAdapter):
    name = "pytrends"

    # ---- cache

    def cache_path(self) -> Path:
        directory = self.cfg.resolve(self.cfg.paths.cache_dir) / "trends"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "pytrends.json"

    def read_cache(self) -> list[RawSignal] | None:
        path = self.cache_path()
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            fetched_at = datetime.fromisoformat(payload["fetched_at"])
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            return None
        age = datetime.now(UTC) - fetched_at
        if age > timedelta(hours=self.cfg.trends.cache_ttl_hours):
            return None
        self.log.info(
            "using cached Google Trends data (%.1f h old)", age.total_seconds() / 3600
        )
        return [
            RawSignal(
                title=item["title"],
                source=self.name,
                year=item.get("year"),
                value=item.get("value", 0.0),
                slope=item.get("slope", 0.0),
                meta=item.get("meta", {}),
            )
            for item in payload.get("signals", [])
        ]

    def write_cache(self, signals: list[RawSignal]) -> None:
        self.cache_path().write_text(
            json.dumps(
                {
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "signals": [
                        {
                            "title": s.title,
                            "year": s.year,
                            "value": s.value,
                            "slope": s.slope,
                            "meta": s.meta,
                        }
                        for s in signals
                    ],
                }
            ),
            encoding="utf-8",
        )

    # ---- candidates

    def candidate_titles(self) -> list[tuple[str, int | None]]:
        """Titles already known to the system, plus films with an anniversary coming up."""
        from ..db import TrendTitle, select, session_scope
        from .manual_csv import ManualCSVAdapter

        candidates: dict[str, tuple[str, int | None]] = {}

        for signal in ManualCSVAdapter(self.cfg).fetch():
            candidates[signal.slug] = (signal.title, signal.year)

        try:
            with session_scope() as session:
                for row in session.exec(select(TrendTitle)).all():
                    if row.status in ("ignored", "processed"):
                        continue
                    candidates.setdefault(row.slug, (row.title, row.year))
        except Exception as exc:  # database not initialised yet
            self.log.debug("no database candidates (%s)", exc)

        releases = load_releases(self.cfg.resolve(self.cfg.trends.release_dates_csv))
        today = date.today()
        for release in releases:
            hit = anniversary_signal(
                release.release_date,
                today,
                self.cfg.trends.anniversary_marks,
                self.cfg.trends.anniversary_window_days,
            )
            if hit is not None:
                candidates.setdefault(
                    slugify(release.title, release.year), (release.title, release.year)
                )

        return list(candidates.values())

    # ---- querying

    def _client(self):
        from pytrends.request import TrendReq

        timeout = self.cfg.trends.request_timeout_s
        return TrendReq(
            hl=f"{self.cfg.trends.language}-{self.cfg.trends.country}",
            tz=60,
            timeout=(timeout, timeout * 2),
            retries=0,          # we do our own retrying so the backoff is ours to control
        )

    def _query_batch(self, client, keywords: list[str], window: str):
        """One payload + interest_over_time, retried with exponential backoff."""
        max_retries = self.cfg.scoring.max_retries
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                client.build_payload(
                    keywords,
                    timeframe=window,
                    geo=self.cfg.trends.country,   # 'ES'
                    cat=CATEGORY_MOVIES,
                )
                frame = client.interest_over_time()
                return frame
            except Exception as exc:
                last_error = exc
                delay = min(2.0 * (2**attempt), 30.0) + random.uniform(0, 1.0)
                self.log.warning(
                    "batch %s failed (attempt %d/%d): %s — retrying in %.1fs",
                    keywords,
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
                if attempt < max_retries - 1:
                    time.sleep(delay)
        self.log.warning("giving up on batch %s: %s", keywords, last_error)
        return None

    def fetch(self) -> list[RawSignal]:
        cached = self.read_cache()
        if cached is not None:
            return cached

        candidates = self.candidate_titles()
        if not candidates:
            self.log.info("no titles to query — seed some first (manual CSV or seed_llm)")
            return []

        window = f"now {self.cfg.trends.trend_window_days}-d"
        client = self._client()
        signals: list[RawSignal] = []
        failed_batches = 0

        for start in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[start : start + BATCH_SIZE]
            frame = self._query_batch(client, [title for title, _ in batch], window)
            if frame is None:
                failed_batches += 1
                continue
            if frame.empty:
                continue
            for title, year in batch:
                if title not in frame:
                    continue
                series = [float(v) for v in frame[title].tolist()]
                signals.append(
                    RawSignal(
                        title=title,
                        source=self.name,
                        year=year,
                        value=series[-1] if series else 0.0,
                        slope=normalised_slope(series),
                        meta={"samples": len(series), "window": window, "geo": "ES"},
                    )
                )
            time.sleep(random.uniform(0.5, 1.5))  # be polite between batches

        if failed_batches:
            self.log.warning(
                "%d/%d batches failed — the ranking will use whatever came back",
                failed_batches,
                (len(candidates) + BATCH_SIZE - 1) // BATCH_SIZE,
            )
        if signals:
            self.write_cache(signals)
        return signals
