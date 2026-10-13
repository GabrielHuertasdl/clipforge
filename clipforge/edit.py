"""Module 6 — the editor. ffmpeg through subprocess, no heavy libraries.

Order of operations (a-h from the spec):

a) dead-air trim (``retiming``) — silences over 400 ms removed, 150 ms of breath kept,
   and a re-timing map so subtitles and re-hooks keep landing on the right frames;
b) cold open — 1.5 s taken from the peak-tension frame with the hook text over it,
   concatenated in front of the clip;
c) dynamic 9:16 crop (``crop``) with smoothing and speaker alternation;
d) burned-in subtitles (``subtitles``) inside the safe zone;
e) re-hooks — alternating incoming text and 3-5% zoom punches snapped to audio peaks;
f) audio — loudnorm to -14 LUFS, aggressive compression, dialogue lift;
g) ending — cut before the resolution, or a continuity line;
h) thumbnail.

Output: ``output/pending/<clip_uid>.mp4`` plus a ``.json`` with the metadata and the exact
variants applied, so real performance can be correlated with them later.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .audio_features import AudioFeatures
from .config import Config
from .crop import CropPlan, build_crop_expr, plan_crop
from .db import Clip, Segment, SegmentScore, Video, log_error, select, session_scope, utcnow
from .ffmpeg import probe, run
from .logging_utils import get_logger
from .retiming import RetimeMap, build_retime_map
from .scoring import ScoredSegment
from .subtitles import build_ass, words_to_output
from .transcribe import Transcript

log = get_logger("edit")


@dataclass(slots=True)
class RehookEvent:
    t: float           # output time, after the cold open
    kind: str          # "text" | "zoom"
    text: str = ""
    duration: float = 1.6


@dataclass(slots=True)
class RenderResult:
    clip_uid: str
    video_path: Path
    json_path: Path
    thumb_path: Path | None
    duration_s: float
    variants: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- escaping


def escape_filter_path(path: Path | str) -> str:
    """Escape a path for use inside an ffmpeg filter argument (ass=, drawtext=...)."""
    text = str(path)
    return text.replace("\\", "/").replace(":", r"\:").replace("'", r"\'").replace(",", r"\,")


def escape_drawtext(text: str) -> str:
    """Escape literal text for drawtext."""
    return (
        text.replace("\\", r"\\")
        .replace(":", r"\:")
        .replace("'", "’")
        .replace("%", r"\%")
        .replace(",", r"\,")
    )


def wrap_text(text: str, max_chars: int) -> str:
    """Hard-wrap a hook line so it never runs off a 1080-wide frame."""
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


# --------------------------------------------------------------------------- filter builders


def build_audio_filters(cfg: Config, select_expr: str | None) -> str:
    """f) loudnorm + aggressive dynamic-range compression + dialogue lift."""
    chain = cfg.edit.audio_chain
    parts: list[str] = []
    if select_expr:
        parts += [f"aselect='{select_expr}'", "asetpts=N/SR/TB"]
    if chain.highpass_hz:
        parts.append(f"highpass=f={chain.highpass_hz:g}")
    if chain.dialogue_boost.enabled:
        db = chain.dialogue_boost
        parts.append(f"equalizer=f={db.freq_hz:g}:t=h:w={db.width_hz:g}:g={db.gain_db:g}")
    if chain.compressor.enabled:
        comp = chain.compressor
        ratio = {"low": max(2.0, comp.ratio / 3), "medium": max(3.0, comp.ratio / 1.5)}.get(
            comp.aggressiveness, comp.ratio
        )
        parts.append(
            "acompressor="
            f"threshold={comp.threshold_db}dB:ratio={ratio:g}:"
            f"attack={comp.attack_ms:g}:release={comp.release_ms:g}:makeup={comp.makeup_db:g}dB"
        )
    parts.append(
        f"loudnorm=I={chain.loudnorm_i:g}:TP={chain.loudnorm_tp:g}:LRA={chain.loudnorm_lra:g}"
    )
    parts.append("aresample=async=1:first_pts=0")
    return ",".join(parts)


def build_zoom_expr(events: list[RehookEvent], cfg: Config) -> str | None:
    """e) zoom punches: a 3-5% push for ``zoom_punch_ms`` at each zoom re-hook."""
    zooms = [e for e in events if e.kind == "zoom"]
    if not zooms:
        return None
    amount = 1.0 + cfg.edit.rehooks.zoom_punch_pct / 100.0
    dur = cfg.edit.rehooks.zoom_punch_ms / 1000.0
    conditions = "+".join(f"between(in_time,{e.t:.3f},{e.t + dur:.3f})" for e in zooms)
    return f"if(gt({conditions},0),{amount:.4f},1)"


def build_video_filters(
    cfg: Config,
    select_expr: str | None,
    crop_plan: CropPlan,
    source_w: int,
    source_h: int,
    ass_path: Path | None,
    rehooks: list[RehookEvent],
    continuity: tuple[float, str] | None = None,
    font_file: str | None = None,
) -> list[str]:
    """Video filter chain in the order the spec requires."""
    edit = cfg.edit
    parts: list[str] = []

    # a) dead-air trim
    if select_expr:
        parts += [f"select='{select_expr}'", "setpts=N/FRAME_RATE/TB"]
    parts.append(f"fps={edit.fps}")

    # c) 9:16 crop, tracking the subject
    crop_w = min(source_w, int(round(source_h * edit.target_w / edit.target_h)))
    crop_w -= crop_w % 2
    x_expr = build_crop_expr(crop_plan.keyframes, source_w, crop_w)
    # crop re-evaluates x/y per frame by default (there is no `eval` option on crop),
    # which is what lets the window pan.
    parts.append(f"crop=w={crop_w}:h={source_h}:x='{x_expr}':y=0")
    parts.append(f"scale={edit.target_w}:{edit.target_h}:flags=bicubic")
    parts.append("setsar=1")

    # e) zoom punches (before text so the text does not wobble)
    zoom = build_zoom_expr(rehooks, cfg)
    if zoom:
        parts.append(
            f"zoompan=z='{zoom}':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"s={edit.target_w}x{edit.target_h}:fps={edit.fps}"
        )

    # d) burned-in subtitles inside the safe zone
    if ass_path is not None:
        parts.append(f"ass='{escape_filter_path(ass_path)}'")

    # e) text re-hooks, sliding in from the right
    font_arg = f":fontfile='{escape_filter_path(font_file)}'" if font_file else ""
    safe_top = edit.subtitles.safe_top_px
    for ev in [e for e in rehooks if e.kind == "text"]:
        end = ev.t + ev.duration
        x = (
            f"'if(lt(t,{ev.t + 0.35:.3f}),w-(w-(w-text_w)/2)*(t-{ev.t:.3f})/0.35,(w-text_w)/2)'"
            if edit.rehooks.text_slide_in
            else "(w-text_w)/2"
        )
        parts.append(
            "drawtext="
            f"text='{escape_drawtext(ev.text)}'{font_arg}:"
            f"fontsize={edit.rehooks.text_font_size}:fontcolor=white:"
            f"borderw=5:bordercolor=black@0.9:"
            f"x={x}:y={safe_top + 60}:"
            f"enable='between(t,{ev.t:.3f},{end:.3f})'"
        )

    # g) continuity line at the end
    if continuity:
        at, text = continuity
        parts.append(
            "drawtext="
            f"text='{escape_drawtext(text)}'{font_arg}:"
            f"fontsize={edit.cold_open.hook_text_font_size - 10}:fontcolor=white:"
            f"borderw=6:bordercolor=black@0.9:x=(w-text_w)/2:y=(h-text_h)/2:"
            f"enable='gte(t,{at:.3f})'"
        )
    return parts


def build_main_command(
    cfg: Config,
    source: Path,
    start_s: float,
    duration_s: float,
    video_filters: list[str],
    audio_filter: str,
    out_path: Path,
) -> list[str]:
    """ffmpeg invocation for the body of the clip."""
    edit = cfg.edit
    return [
        cfg.runtime.ffmpeg,
        "-y",
        "-v", "error",
        "-ss", f"{start_s:.3f}",
        "-t", f"{duration_s:.3f}",
        "-i", str(source),
        "-vf", ",".join(video_filters),
        "-af", audio_filter,
        "-c:v", edit.video_codec,
        "-preset", edit.preset,
        "-crf", str(edit.crf),
        "-pix_fmt", "yuv420p",
        "-r", str(edit.fps),
        "-c:a", edit.audio_codec,
        "-b:a", edit.audio_bitrate,
        "-ar", "48000",
        "-ac", "2",
        "-movflags", "+faststart",
        "-threads", str(cfg.runtime.ffmpeg_threads),
        str(out_path),
    ]


def build_cold_open_command(
    cfg: Config,
    source: Path,
    at_s: float,
    hook_text: str,
    source_w: int,
    source_h: int,
    out_path: Path,
    font_file: str | None = None,
) -> list[str]:
    """b) 1.5 s from the peak-tension frame with the hook text over it."""
    edit = cfg.edit
    co = edit.cold_open
    crop_w = min(source_w, int(round(source_h * edit.target_w / edit.target_h)))
    crop_w -= crop_w % 2
    x = max(0, (source_w - crop_w) // 2)
    font_arg = f":fontfile='{escape_filter_path(font_file)}'" if font_file else ""
    text = wrap_text(hook_text, co.hook_text_max_chars_per_line)
    vf = ",".join(
        [
            f"fps={edit.fps}",
            f"crop=w={crop_w}:h={source_h}:x={x}:y=0",
            f"scale={edit.target_w}:{edit.target_h}:flags=bicubic",
            "setsar=1",
            "drawtext="
            f"text='{escape_drawtext(text)}'{font_arg}:"
            f"fontsize={co.hook_text_font_size}:fontcolor=white:line_spacing=12:"
            "borderw=7:bordercolor=black@0.92:"
            "x=(w-text_w)/2:y=(h-text_h)/2",
        ]
    )
    return [
        cfg.runtime.ffmpeg,
        "-y",
        "-v", "error",
        "-ss", f"{max(0.0, at_s):.3f}",
        "-t", f"{co.duration_s:.3f}",
        "-i", str(source),
        "-vf", vf,
        "-af", "loudnorm=I=-14:TP=-1.0:LRA=7,aresample=async=1:first_pts=0",
        "-c:v", edit.video_codec,
        "-preset", edit.preset,
        "-crf", str(edit.crf),
        "-pix_fmt", "yuv420p",
        "-r", str(edit.fps),
        "-c:a", edit.audio_codec,
        "-b:a", edit.audio_bitrate,
        "-ar", "48000",
        "-ac", "2",
        str(out_path),
    ]


def build_concat_command(cfg: Config, list_file: Path, out_path: Path) -> list[str]:
    return [
        cfg.runtime.ffmpeg,
        "-y",
        "-v", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]


def build_thumbnail_command(cfg: Config, source: Path, at_s: float, out_path: Path) -> list[str]:
    return [
        cfg.runtime.ffmpeg,
        "-y",
        "-v", "error",
        "-ss", f"{max(0.0, at_s):.3f}",
        "-i", str(source),
        "-frames:v", "1",
        "-vf", f"scale={cfg.edit.thumbnail.width}:-2",
        str(out_path),
    ]


# --------------------------------------------------------------------------- re-hook planning


def plan_rehooks(
    scored: ScoredSegment,
    retime: RetimeMap,
    features: AudioFeatures,
    cfg: Config,
    offset_s: float,
    clip_duration: float,
) -> list[RehookEvent]:
    """Turn the model's suggestions into alternating text / zoom-punch events.

    Zoom punches are magnetised to the nearest dramatic audio peak within
    ``snap_to_audio_peak_s`` so the push lands on the hit rather than next to it.
    """
    rh = cfg.edit.rehooks
    if not rh.enabled or not scored.rehook_suggestions:
        return []

    peaks_out = [
        retime.map_time(p.t) + offset_s
        for p in features.peaks_between(retime.segment_start, retime.segment_end)
    ]

    events: list[RehookEvent] = []
    for i, suggestion in enumerate(scored.rehook_suggestions[: rh.count]):
        src_t = float(suggestion.get("t", 0.0))
        t = retime.map_time(src_t) + offset_s
        if t <= offset_s + 1.0 or t >= clip_duration - 1.0:
            continue
        kind = "zoom" if (rh.alternate and i % 2 == 1) else "text"
        if kind == "zoom" and peaks_out:
            nearest = min(peaks_out, key=lambda p: abs(p - t))
            if abs(nearest - t) <= rh.snap_to_audio_peak_s:
                t = nearest
        events.append(
            RehookEvent(
                t=round(t, 3),
                kind=kind,
                text=str(suggestion.get("text", "")),
                duration=rh.text_duration_s,
            )
        )
    events.sort(key=lambda e: e.t)
    return events


# --------------------------------------------------------------------------- rendering


def _segment_bounds(scored: ScoredSegment, cfg: Config) -> tuple[float, float]:
    """g) optionally end the clip before the scene resolves."""
    end = scored.end_s
    if cfg.edit.ending.mode == "cut_before_resolution":
        end = max(scored.start_s + 30.0, end - cfg.edit.ending.cut_before_s)
    return scored.start_s, end


def render_clip(
    video_id: int,
    scored: ScoredSegment,
    cfg: Config,
    transcript: Transcript,
    features: AudioFeatures,
    source_path: Path,
    source_w: int,
    source_h: int,
) -> RenderResult:
    """Render one scored segment into a finished vertical clip."""
    edit = cfg.edit
    tmp_dir = cfg.resolve(cfg.runtime.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_dir = cfg.resolve(cfg.paths.output_pending)
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_uid = uuid.uuid4().hex[:12]

    seg_start, seg_end = _segment_bounds(scored, cfg)

    # a) dead-air trim + re-timing map
    silences = [
        (s.start, s.end) for s in features.silences_between(seg_start, seg_end)
    ]
    offset_s = edit.cold_open.duration_s if edit.cold_open.enabled else 0.0
    # final_max_s bounds the CLIP, so the body budget is what is left after the cold open.
    retime = build_retime_map(
        seg_start, seg_end, silences, cfg, max_output_s=edit.dead_air.final_max_s - offset_s
    )
    body_duration = retime.output_duration
    if body_duration < 5.0:
        raise RuntimeError(f"segment {scored.segment_id} collapsed to {body_duration:.1f}s")

    clip_duration = body_duration + offset_s

    # c) crop plan
    crop_plan = plan_crop(str(source_path), seg_start, seg_end, cfg, transcript, retime)

    # e) re-hooks
    rehooks = plan_rehooks(scored, retime, features, cfg, offset_s, clip_duration)

    # d) subtitles, mapped into clip time (0 = start of the body, before the cold open)
    ass_path: Path | None = None
    if edit.subtitles.enabled:
        words = words_to_output(
            transcript.words_between(seg_start, seg_end), retime, 0.0, body_duration
        )
        ass_path = tmp_dir / f"{clip_uid}.ass"
        ass_path.write_text(build_ass(words, cfg, body_duration), encoding="utf-8")

    # Re-hook / continuity timings inside the body are offset by the cold open, which is
    # concatenated separately, so shift them back here.
    body_rehooks = [
        RehookEvent(max(0.0, e.t - offset_s), e.kind, e.text, e.duration) for e in rehooks
    ]
    continuity = None
    if edit.ending.mode == "continuity_text":
        continuity = (max(0.0, body_duration - 2.0), edit.ending.continuity_text)

    select_expr = None
    if retime.cuts:
        select_expr = "+".join(
            f"between(t,{k.src_start - seg_start:.3f},{k.src_end - seg_start:.3f})"
            for k in retime.keeps
        )

    video_filters = build_video_filters(
        cfg,
        select_expr,
        crop_plan,
        source_w,
        source_h,
        ass_path,
        body_rehooks,
        continuity,
        font_file=edit.subtitles.font_file,
    )
    audio_filter = build_audio_filters(cfg, select_expr)

    body_path = tmp_dir / f"{clip_uid}_body.mp4"
    run(
        build_main_command(
            cfg, source_path, seg_start, seg_end - seg_start, video_filters, audio_filter, body_path
        )
    )

    final_path = out_dir / f"{clip_uid}.mp4"
    cold_open_used = False
    if edit.cold_open.enabled and scored.hook_text:
        cold_path = tmp_dir / f"{clip_uid}_cold.mp4"
        run(
            build_cold_open_command(
                cfg,
                source_path,
                scored.cold_open_timestamp,
                scored.hook_text,
                source_w,
                source_h,
                cold_path,
                font_file=edit.subtitles.font_file,
            )
        )
        list_file = tmp_dir / f"{clip_uid}_concat.txt"
        list_file.write_text(
            f"file '{cold_path.resolve()}'\nfile '{body_path.resolve()}'\n", encoding="utf-8"
        )
        run(build_concat_command(cfg, list_file, final_path))
        cold_open_used = True
        for leftover in (cold_path, list_file):
            leftover.unlink(missing_ok=True)
    else:
        shutil.move(str(body_path), str(final_path))

    body_path.unlink(missing_ok=True)
    if ass_path:
        ass_path.unlink(missing_ok=True)

    info = probe(final_path, ffprobe=cfg.runtime.ffprobe)

    # h) thumbnail
    thumb_path = None
    if edit.thumbnail.enabled:
        thumb_dir = cfg.resolve(cfg.paths.output_thumbs)
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = thumb_dir / f"{clip_uid}.jpg"
        at = {
            "peak_frame": min(info.duration_s - 0.1, offset_s + body_duration * 0.35),
            "cold_open": 0.5,
            "midpoint": info.duration_s / 2,
        }.get(edit.thumbnail.strategy, info.duration_s / 2)
        try:
            run(build_thumbnail_command(cfg, final_path, at, thumb_path))
        except Exception as exc:
            log.warning("thumbnail failed: %s", exc)
            thumb_path = None

    variants = {
        "cold_open": cold_open_used,
        "cold_open_s": offset_s if cold_open_used else 0.0,
        "cold_open_source_s": scored.cold_open_timestamp,
        "hook_text": scored.hook_text,
        "rehooks_total": len(rehooks),
        "rehooks_text": sum(1 for e in rehooks if e.kind == "text"),
        "rehooks_zoom": sum(1 for e in rehooks if e.kind == "zoom"),
        "rehook_events": [asdict(e) for e in rehooks],
        "dead_air_removed_s": round(retime.removed_s, 2),
        "dead_air_cuts": len(retime.cuts),
        "segment_source_s": round(seg_end - seg_start, 2),
        "final_duration_s": round(info.duration_s, 2),
        "crop_dynamic": crop_plan.dynamic,
        "crop_reason": crop_plan.reason,
        "crop_keyframes": len(crop_plan.keyframes),
        "subtitles": edit.subtitles.enabled,
        "subtitle_safe_zone": [edit.subtitles.safe_top_px, edit.subtitles.safe_bottom_px],
        "ending_mode": edit.ending.mode,
        "audio_compression": edit.audio_chain.compressor.aggressiveness,
        "loudnorm_i": edit.audio_chain.loudnorm_i,
        "whisper_model": cfg.transcribe.model,
        "scoring_model": cfg.scoring.model,
    }

    metadata = {
        "clip_uid": clip_uid,
        "video_id": video_id,
        "segment_id": scored.segment_id,
        "source_start_s": scored.start_s,
        "source_end_s": scored.end_s,
        "total": scored.total,
        "scores": scored.scores,
        "hook_text": scored.hook_text,
        "reason": scored.reason,
        "caption": scored.suggested_caption,
        "hashtags": scored.suggested_hashtags,
        "rehook_suggestions": scored.rehook_suggestions,
        "variants": variants,
        "created_at": utcnow().isoformat(),
    }
    json_path = out_dir / f"{clip_uid}.json"
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    with session_scope() as session:
        session.add(
            Clip(
                clip_uid=clip_uid,
                video_id=video_id,
                segment_id=scored.segment_id,
                path=str(final_path),
                json_path=str(json_path),
                thumb_path=str(thumb_path) if thumb_path else "",
                duration_s=info.duration_s,
                size_bytes=final_path.stat().st_size,
                total_score=scored.total,
                caption=scored.suggested_caption,
                hashtags=scored.suggested_hashtags,
                variants=variants,
                status="pending",
            )
        )

    log.info(
        "rendered %s — %.1fs (trimmed %.1fs, %d re-hooks, crop=%s)",
        clip_uid,
        info.duration_s,
        retime.removed_s,
        len(rehooks),
        "dynamic" if crop_plan.dynamic else "center",
    )
    return RenderResult(
        clip_uid=clip_uid,
        video_path=final_path,
        json_path=json_path,
        thumb_path=thumb_path,
        duration_s=info.duration_s,
        variants=variants,
    )


def render_video(
    video_id: int,
    cfg: Config,
    transcript: Transcript,
    features: AudioFeatures,
    top: int = 6,
) -> tuple[list[RenderResult], list[str]]:
    """Render the ``top`` best-scored segments of a video. Returns (results, errors)."""
    from .scoring import load_scores

    with session_scope() as session:
        video = session.get(Video, video_id)
        if video is None:
            raise LookupError(f"video {video_id} not found")
        source_path, width, height = Path(video.path), video.width, video.height
        segment_index = {
            row.id: row.idx
            for row in session.exec(select(Segment).where(Segment.video_id == video_id)).all()
        }
        rendered = {
            c.segment_id
            for c in session.exec(select(Clip).where(Clip.video_id == video_id)).all()
        }

    if not source_path.exists():
        raise FileNotFoundError(f"source file gone: {source_path}")

    scores = [s for s in load_scores(video_id) if s.segment_id not in rendered]
    if top > 0:
        scores = scores[:top]
    if not scores:
        log.info("nothing left to render for video %d", video_id)
        return [], []

    results: list[RenderResult] = []
    errors: list[str] = []
    for scored in scores:
        scored.idx = segment_index.get(scored.segment_id, -1)
        try:
            results.append(
                render_clip(
                    video_id, scored, cfg, transcript, features, source_path, width, height
                )
            )
        except Exception as exc:
            message = f"segment {scored.segment_id}: {exc}"
            log.error("render failed — %s", message)
            log_error(video_id, "edit", message)
            errors.append(message)
            if cfg.runtime.fail_fast:
                raise
    return results, errors


def load_score_rows(video_id: int) -> list[SegmentScore]:
    with session_scope() as session:
        return list(
            session.exec(select(SegmentScore).where(SegmentScore.video_id == video_id)).all()
        )
