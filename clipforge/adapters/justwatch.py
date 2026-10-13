"""JustWatch — streaming arrival dates in Spain.

The best signal there is: a film landing on a catalogue creates a 7-14 day demand spike.
JustWatch has no public documented API, so this adapter talks to the GraphQL endpoint the
site itself uses. That is exactly why it ships **disabled by default** — if the endpoint or
the schema moves, the adapter logs it and returns nothing.
"""

from __future__ import annotations

from datetime import date, datetime

from .base import RawSignal, TrendAdapter

ENDPOINT = "https://apis.justwatch.com/graphql"

QUERY = """
query NewTitles($country: Country!, $language: Language!, $first: Int!) {
  newTitles(country: $country, first: $first, filter: {objectTypes: [MOVIE]}) {
    edges {
      node {
        content(country: $country, language: $language) {
          title
          originalReleaseYear
        }
        offers(country: $country, platform: WEB) {
          availableFromTime
          package { clearName }
        }
      }
    }
  }
}
"""


def days_since(iso_timestamp: str, today: date) -> int | None:
    try:
        arrival = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00")).date()
    except (ValueError, AttributeError):
        return None
    return (today - arrival).days


def arrival_slope(days_ago: int) -> float:
    """Demand peaks for 7-14 days after a title lands, then decays."""
    if days_ago < 0:
        return 70.0                     # arriving soon: worth preparing
    if days_ago <= 14:
        return 100.0 - days_ago * 2.0   # inside the window
    return max(0.0, 70.0 - (days_ago - 14) * 3.0)


class JustWatchAdapter(TrendAdapter):
    name = "justwatch"

    def fetch(self) -> list[RawSignal]:
        import httpx

        variables = {
            "country": self.cfg.trends.country,
            "language": self.cfg.trends.language,
            "first": 60,
        }
        with httpx.Client(timeout=self.cfg.trends.request_timeout_s) as client:
            response = client.post(
                ENDPOINT,
                json={"query": QUERY, "variables": variables},
                headers={"User-Agent": "clipforge/0.1", "Content-Type": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()

        if payload.get("errors"):
            self.log.warning("GraphQL errors: %s", payload["errors"][:1])
            return []

        today = date.today()
        signals: list[RawSignal] = []
        edges = (payload.get("data") or {}).get("newTitles", {}).get("edges", [])
        for edge in edges:
            node = edge.get("node") or {}
            content = node.get("content") or {}
            title = content.get("title")
            if not title:
                continue
            offers = node.get("offers") or []
            best_days: int | None = None
            provider = ""
            for offer in offers:
                delta = days_since(offer.get("availableFromTime", ""), today)
                if delta is None:
                    continue
                if best_days is None or abs(delta) < abs(best_days):
                    best_days = delta
                    provider = (offer.get("package") or {}).get("clearName", "")
            if best_days is None:
                continue
            signals.append(
                RawSignal(
                    title=title,
                    source=self.name,
                    year=content.get("originalReleaseYear"),
                    value=float(best_days),
                    slope=round(arrival_slope(best_days), 1),
                    meta={"days_since_arrival": best_days, "provider": provider},
                )
            )
        return signals
