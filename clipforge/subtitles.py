"""Burned-in subtitles as ASS, built from word-level timestamps.

Rules that matter for retention:

* at most ``max_words_per_line`` words on screen at once (3 by default);
* the active word is highlighted in a different colour — one ASS event per word, each
  redrawing the whole line with that word coloured;
* **position**: the block sits inside the safe zone, never in the bottom ``safe_bottom_px``
  (TikTok's caption/buttons) nor the top ``safe_top_px``.

Timestamps arriving here are already OUTPUT times (post dead-air trim, post cold open).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .transcribe import Word


@dataclass(slots=True)
class SubtitleWord:
    start: float
    end: float
    text: str


def _fmt_time(t: float) -> str:
    """ASS timestamp: h:mm:ss.cc"""
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def group_words(words: list[SubtitleWord], max_per_line: int) -> list[list[SubtitleWord]]:
    """Chop the word stream into lines of at most ``max_per_line`` words.

    A gap longer than 0.8 s starts a new line — it usually means a new sentence.
    """
    lines: list[list[SubtitleWord]] = []
    current: list[SubtitleWord] = []
    for w in words:
        if current and (len(current) >= max_per_line or w.start - current[-1].end > 0.8):
            lines.append(current)
            current = []
        current.append(w)
    if current:
        lines.append(current)
    return lines


def vertical_margin(cfg: Config) -> int:
    """Bottom margin (ASS MarginV) that lands the text inside the safe zone.

    ``vertical_anchor`` 0 pins the block to the top of the safe zone, 1 to the bottom.
    """
    sub = cfg.edit.subtitles
    height = cfg.edit.target_h
    safe_top = sub.safe_top_px
    safe_bottom = sub.safe_bottom_px
    usable = max(1, height - safe_top - safe_bottom)
    # Distance from the bottom of the frame to the anchor point inside the safe band.
    offset_from_top = safe_top + usable * min(1.0, max(0.0, sub.vertical_anchor))
    return int(round(height - offset_from_top))


def build_style(cfg: Config) -> str:
    sub = cfg.edit.subtitles
    margin_v = vertical_margin(cfg)
    side_margin = int(cfg.edit.target_w * 0.08)
    bold = -1 if sub.bold else 0
    # Alignment 2 = bottom-centre; combined with MarginV this positions the block.
    return (
        "Style: Main,"
        f"{sub.font},{sub.font_size},"
        f"{sub.primary_color},{sub.primary_color},{sub.outline_color},&H64000000,"
        f"{bold},0,0,0,100,100,0,0,1,{sub.outline},{sub.shadow},2,"
        f"{side_margin},{side_margin},{margin_v},1"
    )


def ass_header(cfg: Config) -> str:
    return "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {cfg.edit.target_w}",
            f"PlayResY: {cfg.edit.target_h}",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
            "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
            "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
            build_style(cfg),
            "",
            "[Events]",
            "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
        ]
    )


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def render_line(line: list[SubtitleWord], active_index: int, cfg: Config) -> str:
    """One line of text with the active word coloured differently."""
    sub = cfg.edit.subtitles
    parts = []
    for i, w in enumerate(line):
        text = _escape(w.text.strip())
        if sub.uppercase:
            text = text.upper()
        if i == active_index:
            parts.append(f"{{\\c{sub.active_color}}}{text}{{\\c{sub.primary_color}}}")
        else:
            parts.append(text)
    return " ".join(parts)


def build_ass(words: list[SubtitleWord], cfg: Config, clip_duration: float) -> str:
    """Full ASS document: one event per word, each showing its whole line."""
    sub = cfg.edit.subtitles
    events: list[str] = []
    if sub.enabled:
        lines = group_words(words, sub.max_words_per_line)
        for line_idx, line in enumerate(lines):
            # Never let a line linger into the next one: two stacked lines on screen is
            # exactly the noise the 3-word rule exists to avoid.
            next_start = (
                lines[line_idx + 1][0].start if line_idx + 1 < len(lines) else clip_duration
            )
            for i, word in enumerate(line):
                start = max(0.0, word.start)
                if i < len(line) - 1:
                    end = word.end
                else:
                    end = min(word.end + 0.25, next_start, clip_duration)
                end = max(end, start + 0.08)
                if start >= clip_duration:
                    continue
                events.append(
                    f"Dialogue: 0,{_fmt_time(start)},{_fmt_time(min(end, clip_duration))},"
                    f"Main,,0,0,0,,{render_line(line, i, cfg)}"
                )
    return ass_header(cfg) + "\n" + "\n".join(events) + "\n"


def words_to_output(
    words: list[Word], retime, offset_s: float, clip_duration: float
) -> list[SubtitleWord]:
    """Map source-time words into clip time, dropping anything trimmed away.

    ``offset_s`` is the cold-open length, since subtitles start after it.
    """
    out: list[SubtitleWord] = []
    for w in words:
        if retime.is_cut((w.start + w.end) / 2):
            continue
        a, b = retime.map_span(w.start, w.end)
        a += offset_s
        b += offset_s
        if b <= a:
            b = a + 0.08
        if a >= clip_duration:
            continue
        out.append(SubtitleWord(a, min(b, clip_duration), w.text))
    return out
