"""Typer entrypoint.

Commands are wired here as the modules land; anything not built yet exits with a clear
message instead of a traceback.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from . import __version__
from .config import Config, load_config
from .db import init_engine
from .logging_utils import console, setup_logging

app = typer.Typer(
    name="clipforge",
    help="Turn long movie files into scored vertical clips, ready for manual upload.",
    no_args_is_help=True,
    add_completion=False,
)

_state: dict[str, Config] = {}


def get_ctx_config() -> Config:
    return _state["config"]


def load_dotenv(root: Path) -> None:
    """Minimal .env loader — enough for ANTHROPIC_API_KEY, no extra dependency."""
    env_file = root / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def bootstrap(config_path: Path | None = None) -> Config:
    cfg = load_config(config_path)
    load_dotenv(cfg.project_root)
    setup_logging(
        level=cfg.logging.level,
        file=cfg.resolve(cfg.logging.file) if cfg.logging.file else None,
        rich_tracebacks=cfg.logging.rich_tracebacks,
    )
    cfg.ensure_dirs()
    init_engine(cfg.resolve(cfg.paths.db_path))
    _state["config"] = cfg
    return cfg


@app.callback()
def main(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to config.yaml")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="DEBUG logging")] = False,
) -> None:
    cfg = bootstrap(config)
    if verbose:
        log_file = cfg.resolve(cfg.logging.file) if cfg.logging.file else None
        setup_logging(level="DEBUG", file=log_file)


@app.command()
def version() -> None:
    """Print the clipforge version."""
    console.print(f"clipforge {__version__}")


def _hms(seconds: float) -> str:
    seconds = int(seconds or 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _todo(module: str, phase: str) -> None:
    console.print(f"[yellow]{module} not built yet ({phase}).[/yellow]")
    raise typer.Exit(code=2)


@app.command()
def scan(
    force: Annotated[
        bool, typer.Option("--force", help="Re-register already known videos")
    ] = False,
) -> None:
    """Detect new videos in input/raw/ and register them in the database."""
    from rich.table import Table

    from .ingest import scan as run_scan

    cfg = get_ctx_config()
    results = run_scan(cfg, force=force)
    if not results:
        console.print(f"[yellow]No videos in {cfg.resolve(cfg.paths.input_raw)}[/yellow]")
        raise typer.Exit()

    table = Table(title="input/raw", header_style="bold")
    for col in ("id", "title", "year", "duration", "resolution", "status", "note"):
        table.add_column(col)
    for r in results:
        style = {"new": "green", "error": "red", "skipped": "dim"}.get(r.status, "")
        table.add_row(
            str(r.video_id) if r.video_id > 0 else "-",
            r.title or r.path.name,
            str(r.year or ""),
            _hms(r.duration_s),
            f"{r.width}x{r.height}" if r.width else "",
            r.status,
            r.reason,
            style=style,
        )
    console.print(table)
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    console.print(" · ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


@app.command()
def title(
    video_id: Annotated[int, typer.Argument()],
    new_title: Annotated[str, typer.Argument(help="Manual title override")],
) -> None:
    """Override the auto-detected movie title for a video."""
    from .ingest import set_title

    console.print(f"video {video_id} -> [bold]{set_title(video_id, new_title)}[/bold]")


@app.command()
def scenes(
    video_id: Annotated[int, typer.Argument()],
    force: Annotated[bool, typer.Option("--force", help="Re-detect, ignoring the cache")] = False,
    show_discarded: Annotated[
        bool, typer.Option("--all", help="Also list discarded segments")
    ] = False,
) -> None:
    """Detect shots and build candidate segments for one video."""
    from rich.table import Table

    from .scenes import build_segments

    cfg = get_ctx_config()
    candidates = build_segments(video_id, cfg, force=force)

    table = Table(title=f"video {video_id} — candidate segments", header_style="bold")
    for col in ("#", "start", "end", "dur", "shots", "dark", "luma", "state"):
        table.add_column(col)
    for c in candidates:
        if c.discarded and not show_discarded:
            continue
        table.add_row(
            str(c.idx),
            _hms(c.start_s),
            _hms(c.end_s),
            f"{c.duration_s:.0f}s",
            str(c.shot_count),
            f"{c.dark_ratio:.2f}",
            f"{c.mean_luma:.0f}",
            c.discard_reason if c.discarded else "ok",
            style="dim" if c.discarded else "green",
        )
    console.print(table)
    kept = sum(1 for c in candidates if not c.discarded)
    console.print(f"kept: {kept} · discarded: {len(candidates) - kept}")


@app.command()
def transcribe(
    video_id: Annotated[int, typer.Argument()],
    force: Annotated[bool, typer.Option("--force", help="Ignore the transcript cache")] = False,
    preview: Annotated[int, typer.Option("--preview", help="Print the first N words")] = 25,
) -> None:
    """Transcribe one video with word-level timestamps."""
    from .transcribe import transcribe_video

    cfg = get_ctx_config()
    transcript = transcribe_video(video_id, cfg, force=force)
    console.print(
        f"language=[bold]{transcript.language}[/bold] words={len(transcript.words)} "
        f"utterances={len(transcript.utterances)}"
    )
    for w in transcript.words[:preview]:
        console.print(f"  {w.start:7.2f}-{w.end:7.2f}  {w.text}")


@app.command()
def audio(
    video_id: Annotated[int, typer.Argument()],
    force: Annotated[bool, typer.Option("--force", help="Recompute, ignoring the cache")] = False,
) -> None:
    """Compute per-second audio features, dramatic peaks and silence ranges."""
    from .audio_features import analyze_video

    cfg = get_ctx_config()
    f = analyze_video(video_id, cfg, force=force)
    console.print(
        f"frames={len(f.rms)} duration={_hms(f.duration_s)} "
        f"peaks={len(f.peaks)} silences>{cfg.audio.silence_min_ms}ms={len(f.silences)}"
    )
    for p in f.peaks[:10]:
        console.print(f"  peak {_hms(p.t)}  +{p.delta_db:.1f} dB  -> {p.level_db:.1f} dB")
    total_silence = sum(s.duration for s in f.silences)
    pct = total_silence / max(1.0, f.duration_s)
    console.print(f"  total silence: {total_silence:.0f}s ({pct:.0%})")


@app.command()
def analyze(
    video_id: Annotated[int, typer.Argument(help="Video id from `clipforge scan`")],
    top: Annotated[int, typer.Option("--top", help="Only score the N best-placed candidates")] = 0,
    force: Annotated[
        bool, typer.Option("--force", help="Redo every stage, ignoring caches")
    ] = False,
) -> None:
    """Scenes + transcription + audio features + scoring."""

    from .audio_features import analyze_video
    from .scenes import build_segments
    from .scoring import score_video
    from .transcribe import transcribe_video

    cfg = get_ctx_config()
    candidates = build_segments(video_id, cfg, force=force)
    transcript = transcribe_video(video_id, cfg, force=force)
    features = analyze_video(video_id, cfg, force=force)
    try:
        scored = score_video(
            video_id, cfg, transcript, features, candidates, top=top, force=force
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None
    _print_scores(scored, f"video {video_id} — scores")


def _print_scores(scored, title: str) -> None:
    from rich.table import Table

    table = Table(title=title, header_style="bold")
    columns = (
        "seg", "start", "end", "total",
        "hook", "emo", "clar", "quot", "vis", "3min",
        "hook_text",
    )
    for col in columns:
        table.add_column(col)
    for s in scored:
        sc = s.scores
        table.add_row(
            str(s.segment_id),
            _hms(s.start_s),
            _hms(s.end_s),
            f"{s.total:.1f}",
            str(sc.get("hook_strength", "")),
            str(sc.get("emotional_peak", "")),
            str(sc.get("standalone_clarity", "")),
            str(sc.get("quotability", "")),
            str(sc.get("visual_interest", "")),
            str(sc.get("sustains_3min", "")),
            s.hook_text[:48],
        )
    console.print(table)


@app.command()
def scores(
    video_id: Annotated[int, typer.Argument()],
    top: Annotated[int, typer.Option("--top")] = 0,
    recalibrate: Annotated[
        bool, typer.Option("--recalibrate", help="Recompute totals from config weights")
    ] = False,
) -> None:
    """Show stored scores for a video (no API calls)."""
    from .scoring import load_scores
    from .scoring import recalibrate as recalc

    cfg = get_ctx_config()
    rows = recalc(video_id, cfg) if recalibrate else load_scores(video_id, top=top)
    if not rows:
        console.print("[yellow]no scores stored — run `clipforge analyze` first[/yellow]")
        raise typer.Exit()
    _print_scores(rows[:top] if top else rows, f"video {video_id} — stored scores")


@app.command()
def render(
    video_id: Annotated[int, typer.Argument()],
    top: Annotated[int, typer.Option("--top", help="Render the N best-scored segments")] = 6,
) -> None:
    """Edit the N best-scored segments into vertical clips."""
    from .audio_features import analyze_video
    from .edit import render_video
    from .transcribe import transcribe_video

    cfg = get_ctx_config()
    transcript = transcribe_video(video_id, cfg)
    features = analyze_video(video_id, cfg)
    results, errors = render_video(video_id, cfg, transcript, features, top=top)
    _print_renders(results, errors)


def _print_renders(results, errors) -> None:
    from rich.table import Table

    if results:
        table = Table(title="rendered clips", header_style="bold")
        for col in ("clip", "duration", "trimmed", "re-hooks", "crop", "file"):
            table.add_column(col)
        for r in results:
            v = r.variants
            table.add_row(
                r.clip_uid,
                f"{r.duration_s:.1f}s",
                f"{v['dead_air_removed_s']:.1f}s",
                f"{v['rehooks_text']}t/{v['rehooks_zoom']}z",
                "dynamic" if v["crop_dynamic"] else "center",
                str(r.video_path),
            )
        console.print(table)
    for err in errors:
        console.print(f"[red]error:[/red] {err}")
    console.print(f"{len(results)} clip(s) in output/pending · {len(errors)} error(s)")


@app.command()
def run(
    video_id: Annotated[int, typer.Argument()],
    top: Annotated[int, typer.Option("--top")] = 6,
    keep_source: Annotated[
        bool, typer.Option("--keep-source", help="Never delete the source file")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would be deleted, delete nothing")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Redo every stage")] = False,
) -> None:
    """Full pipeline for one video, plus source cleanup."""
    from .audio_features import analyze_video
    from .cleanup import cleanup_source
    from .edit import render_video
    from .scenes import build_segments
    from .scoring import score_video
    from .transcribe import transcribe_video

    cfg = get_ctx_config()
    candidates = build_segments(video_id, cfg, force=force)
    transcript = transcribe_video(video_id, cfg, force=force)
    features = analyze_video(video_id, cfg, force=force)
    try:
        scored = score_video(
            video_id, cfg, transcript, features, candidates, top=top, force=force
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None
    _print_scores(scored[:top], f"video {video_id} — scores")

    results, errors = render_video(video_id, cfg, transcript, features, top=top)
    _print_renders(results, errors)

    decision = cleanup_source(
        video_id, cfg, keep_source=keep_source, dry_run=dry_run, had_errors=bool(errors)
    )
    if decision.deleted:
        console.print(f"[green]source deleted[/green]: {decision.path} "
                      f"({decision.bytes_freed / 1e6:.1f} MB freed)")
    elif decision.dry_run and decision.clips_verified:
        console.print(f"[yellow][dry-run][/yellow] would delete {decision.path} "
                      f"({decision.bytes_freed / 1e6:.1f} MB)")
    else:
        console.print(f"source kept: {decision.reason}")


@app.command()
def daily(
    top: Annotated[
        int | None, typer.Option("--top", help="How many titles to auto-queue")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what it would do, change nothing")
    ] = False,
    keep_source: Annotated[
        bool | None,
        typer.Option("--keep-source/--delete-source", help="Override daily.keep_source"),
    ] = None,
    skip_refresh: Annotated[
        bool, typer.Option("--skip-refresh", help="Reuse stored signals, query no source")
    ] = False,
) -> None:
    """Daily routine: refresh trends, queue titles, process new videos, report."""
    from rich.table import Table

    from .daily import run_daily

    cfg = get_ctx_config()
    report = run_daily(
        cfg, top=top, dry_run=dry_run, keep_source=keep_source, skip_refresh=skip_refresh
    )

    if dry_run:
        console.print("[yellow][dry-run] nothing was changed[/yellow]")

    # --- what to download
    new_slugs = {item.slug for item in report.queued}
    to_download = sorted(report.queued + report.already_queued, key=lambda r: -r.score)
    if to_download:
        table = Table(
            title="DESCARGAR Y DEJAR EN input/raw/" + (" (haría)" if dry_run else ""),
            header_style="bold",
        )
        for col in ("título", "año", "score", "señal", "motivo"):
            table.add_column(col)
        for item in to_download:
            reason = ""
            for source in ("seed_llm", "anniversaries", "manual_csv"):
                detail = item.detail.get(source) or {}
                if detail.get("reason") or detail.get("note"):
                    reason = str(detail.get("reason") or detail.get("note"))
                    break
                if source == "anniversaries" and detail.get("anniversary_years"):
                    reason = f"{detail['anniversary_years']}º aniversario"
                    break
            table.add_row(
                item.title,
                str(item.year or ""),
                f"{item.score:.0f}",
                ", ".join(item.sources),
                reason[:60],
                style="green" if item.slug in new_slugs else "",
            )
        console.print(table)
    else:
        console.print("[yellow]nada nuevo que descargar[/yellow]")

    # --- what was processed
    if report.processed:
        videos = Table(title="vídeos procesados", header_style="bold")
        for col in ("id", "título", "clips", "errores", "fuente"):
            videos.add_column(col)
        for outcome in report.processed:
            videos.add_row(
                str(outcome.video_id),
                outcome.title,
                str(outcome.clips),
                str(len(outcome.errors)) if outcome.errors else "-",
                "borrada" if outcome.source_deleted else "conservada",
                style="red" if outcome.errors else "",
            )
        console.print(videos)
        for outcome in report.processed:
            for err in outcome.errors:
                console.print(f"  [red]{outcome.title}:[/red] {err}")
    else:
        console.print("no hay vídeos nuevos en input/raw/")

    for err in report.errors:
        console.print(f"[red]{err}[/red]")

    console.print(
        f"\n[bold]{report.pending_clips}[/bold] clip(s) esperando revisión — "
        f"`clipforge review`"
    )


@app.command()
def review(
    host: Annotated[str | None, typer.Option("--host", help="Bind address")] = None,
    port: Annotated[int | None, typer.Option("--port")] = None,
) -> None:
    """Launch the review UI (open it from your phone at http://<local-ip>:8000)."""
    import socket

    from .review import serve

    cfg = get_ctx_config()
    bind = host or cfg.review.host
    chosen_port = port or cfg.review.port
    if bind == "0.0.0.0":  # noqa: S104 - binding to the LAN is the point (phone review)
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))
            local_ip = probe.getsockname()[0]
            probe.close()
            console.print(f"phone: [bold]http://{local_ip}:{chosen_port}[/bold]")
        except OSError:
            pass
    console.print(f"local: [bold]http://127.0.0.1:{chosen_port}[/bold]")
    serve(cfg, host=bind, port=chosen_port)


@app.command()
def trends(
    refresh_signals: Annotated[
        bool, typer.Option("--refresh", help="Query the sources again")
    ] = False,
    source: Annotated[
        list[str] | None, typer.Option("--source", help="Limit to these adapters")
    ] = None,
    limit: Annotated[int, typer.Option("--limit")] = 25,
    hide_processed: Annotated[
        bool, typer.Option("--hide-processed", help="Skip films already in the database")
    ] = False,
) -> None:
    """Refresh trend signals and rank titles."""
    from rich.table import Table

    from .trends import load_ranking, refresh

    cfg = get_ctx_config()
    if refresh_signals:
        refresh(cfg, only=list(source) if source else None)
    rows = load_ranking(limit=limit, include_processed=not hide_processed)
    if not rows:
        console.print(
            f"[yellow]No trend data. Fill {cfg.resolve(cfg.trends.manual_csv)} and run "
            "`clipforge trends --refresh`.[/yellow]"
        )
        raise typer.Exit()

    table = Table(title="títulos por prioridad", header_style="bold")
    for col in ("título", "año", "score", "pendiente", "saturación", "iconic", "señal", "estado"):
        table.add_column(col, justify="right" if col in ("score", "pendiente") else "left")
    for row in rows:
        table.add_row(
            row.title,
            str(row.year or ""),
            f"{row.score:.0f}",
            f"{row.trend_slope:.0f}",
            f"{row.saturation:.2f}",
            f"{row.iconic_density:.2f}",
            ", ".join(row.sources),
            row.status,
            style="dim" if row.processed else "",
        )
    console.print(table)
    console.print("mark titles to download in the dashboard: http://127.0.0.1:8000/trends")


@app.command()
def queue(limit: Annotated[int, typer.Option("--limit")] = 50) -> None:
    """List titles marked 'pendiente de descargar' — your download shopping list."""
    from .trends import load_ranking

    rows = [r for r in load_ranking(limit=limit) if r.status == "to_download"]
    if not rows:
        console.print("[yellow]nothing queued — mark titles in /trends[/yellow]")
        raise typer.Exit()
    for row in rows:
        console.print(f"  {row.score:6.0f}  {row.title}{f' ({row.year})' if row.year else ''}")


@app.command()
def stats() -> None:
    """Approval rate, rejection reasons, average scores."""
    from rich.table import Table

    from .review import compute_stats

    data = compute_stats()
    table = Table(title="clipforge stats", header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("clips totales", str(data["total_clips"]))
    table.add_row("pendientes", str(data["pending"]))
    table.add_row("aprobados", str(data["approved"]))
    table.add_row("rechazados", str(data["rejected"]))
    table.add_row("tasa de aprobación", f"{data['approval_rate']}%")
    table.add_row("score medio (todos)", str(data["avg_score_all"]))
    table.add_row("score medio (aprobados)", str(data["avg_score_approved"]))
    table.add_row("score medio (rechazados)", str(data["avg_score_rejected"]))
    console.print(table)

    if data["reject_reasons"]:
        reasons = Table(title="motivos de rechazo", header_style="bold")
        reasons.add_column("motivo")
        reasons.add_column("veces", justify="right")
        for reason, count in data["reject_reasons"].items():
            reasons.add_row(reason, str(count))
        console.print(reasons)


if __name__ == "__main__":
    app()
