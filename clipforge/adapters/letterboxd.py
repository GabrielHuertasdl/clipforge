"""Letterboxd 'popular this week' — the other rediscovery detector.

Scrapes the public listing page. If Letterboxd changes its markup the adapter finds nothing
and says so; it never raises into the run.
"""

from __future__ import annotations

import re

from .base import RawSignal, TrendAdapter

URL = "https://letterboxd.com/films/ajax/popular/this/week/"
FILM_SLUG = re.compile(r'data-film-slug="([^"]+)"')
FILM_NAME = re.compile(r'alt="([^"]+)"')
YEAR_IN_SLUG = re.compile(r"-(19\d{2}|20\d{2})$")


def parse_films(html: str) -> list[tuple[str, int | None]]:
    """Extract (title, year) pairs in ranking order, de-duplicated."""
    slugs = FILM_SLUG.findall(html)
    names = FILM_NAME.findall(html)
    films: list[tuple[str, int | None]] = []
    seen: set[str] = set()
    for i, slug in enumerate(slugs):
        year_match = YEAR_IN_SLUG.search(slug)
        year = int(year_match.group(1)) if year_match else None
        name = names[i] if i < len(names) else slug.replace("-", " ").title()
        key = f"{name}-{year}"
        if key in seen:
            continue
        seen.add(key)
        films.append((name, year))
    return films


class LetterboxdAdapter(TrendAdapter):
    name = "letterboxd"

    def fetch(self) -> list[RawSignal]:
        import httpx

        headers = {"User-Agent": "clipforge/0.1 (local trend detector)"}
        with httpx.Client(
            timeout=self.cfg.trends.request_timeout_s, headers=headers, follow_redirects=True
        ) as client:
            response = client.get(URL)
            response.raise_for_status()
            films = parse_films(response.text)

        if not films:
            self.log.info("nothing parsed — the page markup may have changed")
            return []

        total = len(films)
        return [
            RawSignal(
                title=title,
                source=self.name,
                year=year,
                value=float(total - rank),
                # Higher up the weekly list = stronger signal.
                slope=round(100.0 * (total - rank) / total, 1),
                meta={"rank": rank + 1},
            )
            for rank, (title, year) in enumerate(films)
        ]
