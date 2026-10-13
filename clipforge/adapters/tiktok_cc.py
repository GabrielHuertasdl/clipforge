"""TikTok Creative Center — trending hashtags and sounds (Spain, entertainment).

Context rather than a per-film signal: it tells you which hashtags and sounds are hot right
now, which the dashboard shows next to the ranking. Undocumented endpoint, so it is disabled
by default and fails soft.
"""

from __future__ import annotations

from .base import RawSignal, TrendAdapter

HASHTAG_URL = "https://ads.tiktok.com/creative_radar_api/v1/popular_trend/hashtag/list"


class TikTokCreativeCenterAdapter(TrendAdapter):
    name = "tiktok_cc"

    def fetch(self) -> list[RawSignal]:
        import httpx

        params = {
            "period": "7",
            "page": "1",
            "limit": "50",
            "country_code": self.cfg.trends.country,
            "industry_id": "22000000000",  # entertainment
            "sort_by": "popular",
        }
        headers = {
            "User-Agent": "clipforge/0.1",
            "Accept": "application/json",
            "Referer": "https://ads.tiktok.com/business/creativecenter/",
        }
        with httpx.Client(timeout=self.cfg.trends.request_timeout_s, headers=headers) as client:
            response = client.get(HASHTAG_URL, params=params)
            response.raise_for_status()
            payload = response.json()

        items = ((payload.get("data") or {}).get("list")) or []
        if not items:
            self.log.info("no hashtags returned — endpoint shape may have changed")
            return []

        signals: list[RawSignal] = []
        for rank, item in enumerate(items):
            name = item.get("hashtag_name") or item.get("hashtag") or ""
            if not name:
                continue
            signals.append(
                RawSignal(
                    title=f"#{name.lstrip('#')}",
                    source=self.name,
                    value=float(item.get("video_views", 0) or 0),
                    slope=round(max(0.0, 100.0 - rank * 2.0), 1),
                    meta={"kind": "hashtag", "rank": rank + 1, "raw": name},
                )
            )
        return signals
