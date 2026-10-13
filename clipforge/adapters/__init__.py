"""Trend signal adapters.

Every source lives behind the same tiny interface and is **optional**: if an adapter is
disabled in config, its dependency is missing, or the remote endpoint changes shape, it
returns nothing and the rest of the system carries on. The manual CSV adapter is the one
that always works and the one the whole flow was validated against.
"""

from __future__ import annotations

from .anniversaries import AnniversariesAdapter
from .base import RawSignal, TrendAdapter, slugify
from .justwatch import JustWatchAdapter
from .letterboxd import LetterboxdAdapter
from .manual_csv import ManualCSVAdapter
from .pytrends_source import PytrendsAdapter
from .reddit import RedditAdapter
from .seed_llm import SeedLLMAdapter
from .tiktok_cc import TikTokCreativeCenterAdapter

ADAPTERS: dict[str, type[TrendAdapter]] = {
    "manual_csv": ManualCSVAdapter,
    "seed_llm": SeedLLMAdapter,
    "justwatch": JustWatchAdapter,
    "pytrends": PytrendsAdapter,
    "tiktok_cc": TikTokCreativeCenterAdapter,
    "reddit": RedditAdapter,
    "letterboxd": LetterboxdAdapter,
    "anniversaries": AnniversariesAdapter,
}

__all__ = ["ADAPTERS", "RawSignal", "TrendAdapter", "slugify"]
