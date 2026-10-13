"""Reddit (r/movies, r/NetflixBestOf) — good at catching rediscoveries.

A film resurfacing on Reddit usually has no fresh clip competition, which is exactly the
gap worth filling.
"""

from __future__ import annotations

import re

from .base import RawSignal, TrendAdapter

SUBREDDITS = ("movies", "NetflixBestOf")
QUOTED = re.compile(r"[\"“']([^\"”']{2,60})[\"”']")
TITLE_YEAR = re.compile(r"([\w'’:.\- ]{2,80}?)\s*\((19\d{2}|20\d{2})\)")

# Particles that legitimately sit inside a title without being capitalised.
PARTICLES = {"of", "the", "a", "an", "and", "in", "on", "to", "de", "del", "la",
             "el", "los", "las", "y", "un", "una"}


def trim_to_title(phrase: str) -> str:
    """Keep only the title-looking tail of a Reddit post title.

    'Just rewatched Heat (1995)' -> 'Heat'. Walking right to left, we keep capitalised
    words and in-title particles and stop at the first ordinary lowercase word. It is a
    heuristic: a post that capitalises its verbs will keep them.
    """
    words = phrase.strip(" -:–—").split()
    kept: list[str] = []
    for word in reversed(words):
        clean = word.strip(",.;:!?")
        if not clean:
            break
        if clean[0].isupper() or clean.lower() in PARTICLES:
            kept.append(clean)
            continue
        break
    # A leading article belongs to the title ("The Godfather", "El Padrino"), so it stays —
    # unless the whole thing is particles, which means we caught no title at all.
    if all(word.lower() in PARTICLES for word in kept):
        return ""
    return " ".join(reversed(kept))


def extract_titles(text: str) -> list[tuple[str, int | None]]:
    """Pull film titles out of a post title: 'Heat (1995)' or quoted names."""
    found: list[tuple[str, int | None]] = []
    for match in TITLE_YEAR.finditer(text):
        title = trim_to_title(match.group(1))
        if len(title) >= 2:
            found.append((title, int(match.group(2))))
    if not found:
        for match in QUOTED.finditer(text):
            candidate = match.group(1).strip()
            if len(candidate.split()) <= 8:
                found.append((candidate, None))
    return found


class RedditAdapter(TrendAdapter):
    name = "reddit"

    def fetch(self) -> list[RawSignal]:
        import httpx

        headers = {"User-Agent": "clipforge/0.1 (local trend detector)"}
        counts: dict[tuple[str, int | None], dict] = {}

        with httpx.Client(
            timeout=self.cfg.trends.request_timeout_s, headers=headers, follow_redirects=True
        ) as client:
            for sub in SUBREDDITS:
                response = client.get(
                    f"https://www.reddit.com/r/{sub}/top.json", params={"t": "week", "limit": 100}
                )
                response.raise_for_status()
                for child in response.json().get("data", {}).get("children", []):
                    post = child.get("data", {})
                    score = float(post.get("score", 0))
                    for title, year in extract_titles(post.get("title", "")):
                        key = (title, year)
                        entry = counts.setdefault(
                            key, {"score": 0.0, "mentions": 0, "subs": set()}
                        )
                        entry["score"] += score
                        entry["mentions"] += 1
                        entry["subs"].add(sub)

        if not counts:
            return []
        top_score = max(entry["score"] for entry in counts.values()) or 1.0
        return [
            RawSignal(
                title=title,
                source=self.name,
                year=year,
                value=entry["score"],
                slope=round(100.0 * entry["score"] / top_score, 1),
                meta={"mentions": entry["mentions"], "subreddits": sorted(entry["subs"])},
            )
            for (title, year), entry in counts.items()
            if entry["mentions"] >= 2
        ]
