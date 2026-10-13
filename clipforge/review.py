"""Module 7 — review UI (FastAPI + HTMX on localhost).

One clip at a time, full screen, autoplay, with the score breakdown and an editable caption
underneath. Everything is reachable from the keyboard so a batch can be dispatched fast:

    A  approve      R  reject      E  edit caption
    ← → previous / next            space  play / pause

Approving moves the clip and its JSON to ``output/approved/`` with the final caption;
rejecting moves them to ``output/rejected/`` with a reason. The layout is responsive and the
server binds to ``0.0.0.0`` by default so the verticals can be reviewed on a phone — the
device they are actually made for.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import Config
from .db import Clip, Video, select, session_scope, utcnow
from .logging_utils import get_logger

log = get_logger("review")

WEB_DIR = Path(__file__).parent / "web"


# --------------------------------------------------------------------------- data access


def _clip_payload(clip: Clip, video_title: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if clip.json_path and Path(clip.json_path).exists():
        try:
            meta = json.loads(Path(clip.json_path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
    return {
        "uid": clip.clip_uid,
        "video_id": clip.video_id,
        "title": video_title,
        "duration_s": clip.duration_s,
        "total": clip.total_score,
        "scores": meta.get("scores", {}),
        "reason": meta.get("reason", ""),
        "hook_text": meta.get("hook_text", ""),
        "caption": clip.caption or meta.get("caption", ""),
        "hashtags": clip.hashtags or meta.get("hashtags", []),
        "variants": clip.variants or meta.get("variants", {}),
        "status": clip.status,
    }


def list_pending(cfg: Config) -> list[dict[str, Any]]:
    """Pending clips, best score first — the order they get reviewed in."""
    with session_scope() as session:
        clips = list(
            session.exec(select(Clip).where(Clip.status == "pending")).all()
        )
        titles = {
            v.id: v.display_title for v in session.exec(select(Video)).all()
        }
        payload = [_clip_payload(c, titles.get(c.video_id, "")) for c in clips]
    payload.sort(key=lambda c: -c["total"])
    return payload


def get_clip(uid: str) -> dict[str, Any] | None:
    with session_scope() as session:
        clip = session.exec(select(Clip).where(Clip.clip_uid == uid)).first()
        if clip is None:
            return None
        video = session.get(Video, clip.video_id)
        return _clip_payload(clip, video.display_title if video else "")


def _move_clip(cfg: Config, uid: str, destination: Path, meta_update: dict[str, Any]) -> Path:
    """Move the mp4 + json into a destination folder, updating the JSON on the way."""
    destination.mkdir(parents=True, exist_ok=True)
    with session_scope() as session:
        clip = session.exec(select(Clip).where(Clip.clip_uid == uid)).first()
        if clip is None:
            raise LookupError(f"clip {uid} not found")
        src_video = Path(clip.path)
        src_json = Path(clip.json_path) if clip.json_path else None

        new_video = destination / src_video.name
        if src_video.exists():
            shutil.move(str(src_video), str(new_video))
        clip.path = str(new_video)

        if src_json and src_json.exists():
            try:
                meta = json.loads(src_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                meta = {}
            meta.update(meta_update)
            new_json = destination / src_json.name
            new_json.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            src_json.unlink(missing_ok=True)
            clip.json_path = str(new_json)
        session.add(clip)
        return new_video


def approve_clip(cfg: Config, uid: str, caption: str, hashtags: str | None = None) -> Path:
    """Approve: final caption wins, files land in output/approved/."""
    tags = (
        [t.strip() for t in hashtags.replace(",", " ").split() if t.strip()]
        if hashtags is not None
        else None
    )
    with session_scope() as session:
        clip = session.exec(select(Clip).where(Clip.clip_uid == uid)).first()
        if clip is None:
            raise LookupError(f"clip {uid} not found")
        clip.caption = caption.strip()
        if tags is not None:
            clip.hashtags = [t if t.startswith("#") else f"#{t}" for t in tags]
        clip.status = "approved"
        clip.reject_reason = None
        clip.decided_at = utcnow()
        final_caption = clip.caption
        final_tags = list(clip.hashtags or [])
        session.add(clip)

    path = _move_clip(
        cfg,
        uid,
        cfg.resolve(cfg.paths.output_approved),
        {
            "status": "approved",
            "final_caption": final_caption,
            "final_hashtags": final_tags,
            "decided_at": utcnow().isoformat(),
        },
    )
    log.info("approved %s -> %s", uid, path)
    return path


def reject_clip(cfg: Config, uid: str, reason: str, note: str = "") -> Path:
    with session_scope() as session:
        clip = session.exec(select(Clip).where(Clip.clip_uid == uid)).first()
        if clip is None:
            raise LookupError(f"clip {uid} not found")
        clip.status = "rejected"
        clip.reject_reason = reason
        clip.reject_note = note or None
        clip.decided_at = utcnow()
        session.add(clip)

    path = _move_clip(
        cfg,
        uid,
        cfg.resolve(cfg.paths.output_rejected),
        {
            "status": "rejected",
            "reject_reason": reason,
            "reject_note": note,
            "decided_at": utcnow().isoformat(),
        },
    )
    log.info("rejected %s (%s)", uid, reason)
    return path


def compute_stats() -> dict[str, Any]:
    """Approval rate, rejection reasons and average scores — also used by `clipforge stats`."""
    with session_scope() as session:
        clips = [
            {
                "status": c.status,
                "reason": c.reject_reason,
                "total": c.total_score,
                "variants": dict(c.variants or {}),
            }
            for c in session.exec(select(Clip)).all()
        ]

    decided = [c for c in clips if c["status"] in ("approved", "rejected")]
    approved = [c for c in decided if c["status"] == "approved"]
    rejected = [c for c in decided if c["status"] == "rejected"]
    reasons: dict[str, int] = {}
    for clip in rejected:
        key = clip["reason"] or "sin motivo"
        reasons[key] = reasons.get(key, 0) + 1

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 1) if values else 0.0

    return {
        "total_clips": len(clips),
        "pending": sum(1 for c in clips if c["status"] == "pending"),
        "approved": len(approved),
        "rejected": len(rejected),
        "approval_rate": round(100 * len(approved) / len(decided), 1) if decided else 0.0,
        "avg_score_all": mean([c["total"] for c in clips]),
        "avg_score_approved": mean([c["total"] for c in approved]),
        "avg_score_rejected": mean([c["total"] for c in rejected]),
        "reject_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


# --------------------------------------------------------------------------- app


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="clipforge review", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
    app.state.cfg = cfg

    def _queue_context(uid: str | None) -> dict[str, Any]:
        queue = list_pending(cfg)
        uids = [c["uid"] for c in queue]
        current = None
        index = 0
        if uid:
            current = get_clip(uid)
            index = uids.index(uid) if uid in uids else 0
        elif queue:
            current = queue[0]
        if current is None and queue:
            current = queue[0]
            index = 0
        prev_uid = uids[index - 1] if uids and index > 0 else None
        next_uid = uids[index + 1] if uids and index + 1 < len(uids) else None
        return {
            "clip": current,
            "queue_size": len(queue),
            "position": index + 1 if queue else 0,
            "prev_uid": prev_uid,
            "next_uid": next_uid,
            "reject_reasons": cfg.review.reject_reasons,
            "autoplay": cfg.review.autoplay,
        }

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(
            request=request, name="review.html", context=_queue_context(None)
        )

    @app.get("/clip/{uid}", response_class=HTMLResponse)
    def clip_page(request: Request, uid: str):
        if get_clip(uid) is None:
            raise HTTPException(status_code=404, detail="clip not found")
        return templates.TemplateResponse(
            request=request, name="review.html", context=_queue_context(uid)
        )

    @app.get("/panel/{uid}", response_class=HTMLResponse)
    def panel(request: Request, uid: str):
        """HTMX fragment: the whole reviewer for one clip."""
        return templates.TemplateResponse(
            request=request, name="_panel.html", context=_queue_context(uid)
        )

    @app.get("/panel", response_class=HTMLResponse)
    def next_panel(request: Request):
        return templates.TemplateResponse(
            request=request, name="_panel.html", context=_queue_context(None)
        )

    @app.get("/media/{uid}.mp4")
    def media(uid: str):
        clip = get_clip(uid)
        if clip is None:
            raise HTTPException(status_code=404, detail="clip not found")
        with session_scope() as session:
            row = session.exec(select(Clip).where(Clip.clip_uid == uid)).first()
            path = Path(row.path) if row else None
        if not path or not path.exists():
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/thumb/{uid}.jpg")
    def thumb(uid: str):
        with session_scope() as session:
            row = session.exec(select(Clip).where(Clip.clip_uid == uid)).first()
            path = Path(row.thumb_path) if row and row.thumb_path else None
        if not path or not path.exists():
            raise HTTPException(status_code=404, detail="no thumbnail")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/clip/{uid}/approve", response_class=HTMLResponse)
    def approve(
        request: Request,
        uid: str,
        caption: str = Form(""),
        hashtags: str = Form(None),
    ):
        try:
            approve_clip(cfg, uid, caption, hashtags)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        ctx = _queue_context(None)
        ctx["flash"] = "aprobado"
        return templates.TemplateResponse(request=request, name="_panel.html", context=ctx)

    @app.post("/clip/{uid}/reject", response_class=HTMLResponse)
    def reject(
        request: Request,
        uid: str,
        reason: str = Form("otro"),
        note: str = Form(""),
    ):
        try:
            reject_clip(cfg, uid, reason, note)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        ctx = _queue_context(None)
        ctx["flash"] = f"rechazado: {reason}"
        return templates.TemplateResponse(request=request, name="_panel.html", context=ctx)

    @app.get("/stats", response_class=HTMLResponse)
    def stats_page(request: Request):
        return templates.TemplateResponse(
            request=request, name="stats.html", context={"stats": compute_stats()}
        )

    @app.get("/trends", response_class=HTMLResponse)
    def trends_page(request: Request, sort: str = "score", hide_processed: bool = False):
        from .trends import hashtag_signals, load_ranking

        rows = load_ranking(include_processed=not hide_processed)
        keys = {
            "score": lambda r: -r.score,
            "title": lambda r: r.title.lower(),
            "slope": lambda r: -r.trend_slope,
            "saturation": lambda r: r.saturation,
            "source": lambda r: r.top_source,
            "status": lambda r: r.status,
        }
        rows.sort(key=keys.get(sort, keys["score"]))
        return templates.TemplateResponse(
            request=request,
            name="trends.html",
            context={
                "rows": rows,
                "sort": sort,
                "hide_processed": hide_processed,
                "hashtags": hashtag_signals(),
            },
        )

    @app.post("/trends/{slug}/status", response_class=HTMLResponse)
    def trend_status(request: Request, slug: str, status: str = Form("to_download")):
        from .trends import load_ranking, set_status

        try:
            set_status(slug, status)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        return templates.TemplateResponse(
            request=request,
            name="_trend_rows.html",
            context={"rows": load_ranking(), "sort": "score", "hide_processed": False},
        )

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "pending": len(list_pending(cfg))}

    @app.get("/favicon.ico")
    def favicon():
        return RedirectResponse("/static/favicon.svg")

    return app


def serve(cfg: Config, host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    uvicorn.run(
        create_app(cfg),
        host=host or cfg.review.host,
        port=port or cfg.review.port,
        log_level=cfg.logging.level.lower(),
    )
