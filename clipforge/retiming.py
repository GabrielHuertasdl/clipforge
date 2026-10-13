"""Dead-air trimming and the re-timing map.

This is the most fragile piece of the editor: once silences are cut out of a segment, every
downstream timestamp — subtitle words, re-hook beats, the cold-open frame, audio peaks — has
to move with it. ``RetimeMap`` is the single source of truth for that translation.

Vocabulary:
* **source time** — seconds in the original video file.
* **output time** — seconds in the trimmed clip, where 0 is the start of the segment.

A source instant that falls inside a removed silence has no exact output time; ``map_time``
snaps it to the nearest surviving edge, which is what subtitles and re-hooks want.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

from .config import Config
from .logging_utils import get_logger

log = get_logger("retiming")


@dataclass(slots=True)
class KeepInterval:
    """A stretch of source video that survives into the clip."""

    src_start: float
    src_end: float
    out_start: float

    @property
    def duration(self) -> float:
        return self.src_end - self.src_start

    @property
    def out_end(self) -> float:
        return self.out_start + self.duration


@dataclass(slots=True)
class RetimeMap:
    """Piecewise-linear source -> output mapping built from the kept intervals."""

    segment_start: float
    segment_end: float
    keeps: list[KeepInterval] = field(default_factory=list)
    cuts: list[tuple[float, float]] = field(default_factory=list)

    @property
    def source_duration(self) -> float:
        return self.segment_end - self.segment_start

    @property
    def output_duration(self) -> float:
        return sum(k.duration for k in self.keeps)

    @property
    def removed_s(self) -> float:
        return self.source_duration - self.output_duration

    def map_time(self, src_t: float) -> float:
        """Source instant -> output instant, snapping cut instants to the nearest edge."""
        if not self.keeps:
            return 0.0
        if src_t <= self.keeps[0].src_start:
            return 0.0
        if src_t >= self.keeps[-1].src_end:
            return self.output_duration

        starts = [k.src_start for k in self.keeps]
        i = bisect_right(starts, src_t) - 1
        keep = self.keeps[i]
        if src_t <= keep.src_end:
            return keep.out_start + (src_t - keep.src_start)
        # Inside a removed gap: snap to whichever edge is closer.
        nxt = self.keeps[i + 1]
        gap_before = src_t - keep.src_end
        gap_after = nxt.src_start - src_t
        return keep.out_end if gap_before <= gap_after else nxt.out_start

    def unmap_time(self, out_t: float) -> float:
        """Output instant -> source instant."""
        if not self.keeps:
            return self.segment_start
        out_t = max(0.0, min(out_t, self.output_duration))
        for keep in self.keeps:
            if out_t <= keep.out_end:
                return keep.src_start + (out_t - keep.out_start)
        return self.keeps[-1].src_end

    def is_cut(self, src_t: float) -> bool:
        return any(a <= src_t < b for a, b in self.cuts)

    def map_span(self, src_start: float, src_end: float) -> tuple[float, float]:
        """Map an interval, guaranteeing a non-negative duration."""
        a = self.map_time(src_start)
        b = self.map_time(src_end)
        return a, max(a, b)

    def select_filter(self) -> str:
        """ffmpeg ``select`` expression keeping exactly the surviving source ranges."""
        parts = [f"between(t,{k.src_start:.3f},{k.src_end:.3f})" for k in self.keeps]
        return "+".join(parts) if parts else "1"


def build_keeps(
    segment_start: float,
    segment_end: float,
    cuts: list[tuple[float, float]],
) -> list[KeepInterval]:
    """Turn a list of removed ranges into the kept intervals, with output offsets."""
    keeps: list[KeepInterval] = []
    cursor = segment_start
    out = 0.0
    for a, b in sorted(cuts):
        a = max(a, segment_start, cursor)  # overlapping cuts merge instead of being skipped
        b = min(b, segment_end)
        if b <= a:
            continue
        if a > cursor:
            keeps.append(KeepInterval(cursor, a, out))
            out += a - cursor
        cursor = b
    if cursor < segment_end:
        keeps.append(KeepInterval(cursor, segment_end, out))
    return keeps


def plan_cuts(
    segment_start: float,
    segment_end: float,
    silences: list[tuple[float, float]],
    cfg: Config,
) -> list[tuple[float, float]]:
    """Choose which parts of the silences to remove.

    Each silence longer than ``silence_min_ms`` is cut except for ``keep_breath_ms`` of
    breathing room, split evenly across both ends. Cuts are then accepted longest-first until
    the ``max_cut_ratio`` budget runs out, so we always spend the budget on the deadest air.
    """
    da = cfg.edit.dead_air
    if not da.enabled:
        return []

    min_s = da.silence_min_ms / 1000.0
    breath = da.keep_breath_ms / 1000.0
    duration = segment_end - segment_start
    budget = duration * da.max_cut_ratio

    proposed: list[tuple[float, float]] = []
    for raw_a, raw_b in silences:
        a = max(raw_a, segment_start)
        b = min(raw_b, segment_end)
        if (b - a) < min_s:
            continue
        cut_a = a + breath / 2
        cut_b = b - breath / 2
        if cut_b - cut_a <= 0.05:  # nothing worth cutting once breath is preserved
            continue
        proposed.append((cut_a, cut_b))

    accepted: list[tuple[float, float]] = []
    spent = 0.0
    for cut in sorted(proposed, key=lambda c: c[1] - c[0], reverse=True):
        length = cut[1] - cut[0]
        if spent + length > budget:
            continue
        accepted.append(cut)
        spent += length
    return sorted(accepted)


def build_retime_map(
    segment_start: float,
    segment_end: float,
    silences: list[tuple[float, float]],
    cfg: Config,
    max_output_s: float | None = None,
) -> RetimeMap:
    """Full dead-air pass: plan the cuts, enforce the target length, build the map.

    If the trimmed clip still exceeds ``final_max_s`` the tail is dropped as well — the
    editor's ending rule cuts before the resolution anyway, so losing the tail is preferable
    to shipping a clip that runs long.
    """
    cuts = plan_cuts(segment_start, segment_end, silences, cfg)
    keeps = build_keeps(segment_start, segment_end, cuts)
    out_duration = sum(k.duration for k in keeps)

    da = cfg.edit.dead_air
    limit = da.final_max_s if max_output_s is None else max_output_s
    if out_duration > limit:
        overflow = out_duration - limit
        tail_cut_start = None
        # Walk backwards removing output time until we fit.
        remaining = overflow
        trimmed: list[KeepInterval] = []
        for keep in reversed(keeps):
            if remaining <= 0:
                trimmed.append(keep)
                continue
            if keep.duration <= remaining:
                remaining -= keep.duration
                tail_cut_start = keep.src_start
                continue
            new_end = keep.src_end - remaining
            tail_cut_start = new_end
            remaining = 0
            trimmed.append(KeepInterval(keep.src_start, new_end, keep.out_start))
        keeps = list(reversed(trimmed))
        if tail_cut_start is not None:
            cuts = cuts + [(tail_cut_start, segment_end)]
        out_duration = sum(k.duration for k in keeps)

    if out_duration < da.final_min_s:
        log.debug(
            "trimmed clip is %.1fs, below final_min_s=%.0fs — keeping it as is",
            out_duration,
            da.final_min_s,
        )

    return RetimeMap(
        segment_start=segment_start,
        segment_end=segment_end,
        keeps=keeps,
        cuts=sorted(cuts),
    )
