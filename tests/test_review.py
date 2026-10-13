"""Review UI tests through FastAPI's TestClient — real routes, real file moves."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from clipforge.config import Config
from clipforge.db import Clip, Video, init_engine, select, session_scope
from clipforge.review import compute_stats, create_app


@pytest.fixture
def env(tmp_path):
    cfg = Config(project_root=tmp_path)
    cfg.ensure_dirs()
    init_engine(tmp_path / "review.db")

    with session_scope() as s:
        video = Video(file_hash="h", filename="Heat.mkv", path="/x/Heat.mkv", title="Heat")
        s.add(video)
        s.flush()
        video_id = video.id

    pending = cfg.resolve(cfg.paths.output_pending)
    for uid, score in (("aaa", 90.0), ("bbb", 60.0)):
        (pending / f"{uid}.mp4").write_bytes(b"v" * 2048)
        (pending / f"{uid}.json").write_text(
            json.dumps(
                {
                    "clip_uid": uid,
                    "scores": {"hook_strength": 80, "sustains_3min": 55},
                    "reason": "porque sí",
                    "hook_text": "mira la mano",
                }
            ),
            encoding="utf-8",
        )
        with session_scope() as s:
            s.add(
                Clip(
                    clip_uid=uid, video_id=video_id, segment_id=1,
                    path=str(pending / f"{uid}.mp4"),
                    json_path=str(pending / f"{uid}.json"),
                    duration_s=185.0, size_bytes=2048, total_score=score,
                    caption="caption sugerida", hashtags=["#cine"],
                    variants={"cold_open": True, "rehooks_total": 7},
                )
            )
    return cfg, TestClient(create_app(cfg))


def test_index_shows_the_best_clip_first(env):
    _, client = env
    body = client.get("/").text
    assert 'data-uid="aaa"' in body      # 90 beats 60
    assert "1/2" in body


def test_page_has_every_required_shortcut(env):
    _, client = env
    body = client.get("/").text
    js = client.get("/static/app.js").text
    for key in ("'a'", "'r'", "'e'", "arrowleft", "arrowright", "' '"):
        assert key in js
    assert "<kbd>A</kbd>" in body and "<kbd>R</kbd>" in body and "<kbd>E</kbd>" in body


def test_mobile_viewport_and_responsive_css(env):
    _, client = env
    assert "width=device-width" in client.get("/").text
    css = client.get("/static/style.css").text
    assert "@media (max-width: 900px)" in css
    assert "min-height: 56px" in css     # touch-sized buttons


def test_caption_is_editable_and_prefilled(env):
    _, client = env
    body = client.get("/clip/aaa").text
    assert 'name="caption"' in body
    assert "caption sugerida" in body


def test_score_breakdown_and_reason_are_shown(env):
    _, client = env
    body = client.get("/clip/aaa").text
    assert "80" in body and "porque sí" in body and "mira la mano" in body


def test_navigation_links_walk_the_queue(env):
    _, client = env
    body = client.get("/clip/aaa").text
    assert 'hx-get="/panel/bbb"' in body
    body_b = client.get("/clip/bbb").text
    assert 'hx-get="/panel/aaa"' in body_b


def test_approve_moves_files_and_saves_the_final_caption(env):
    cfg, client = env
    response = client.post(
        "/clip/aaa/approve", data={"caption": "mi caption final", "hashtags": "cine terror"}
    )
    assert response.status_code == 200

    approved = cfg.resolve(cfg.paths.output_approved)
    assert (approved / "aaa.mp4").exists()
    assert not (cfg.resolve(cfg.paths.output_pending) / "aaa.mp4").exists()

    meta = json.loads((approved / "aaa.json").read_text(encoding="utf-8"))
    assert meta["final_caption"] == "mi caption final"
    assert meta["final_hashtags"] == ["#cine", "#terror"]
    assert meta["status"] == "approved"

    with session_scope() as s:
        clip = s.exec(select(Clip).where(Clip.clip_uid == "aaa")).first()
        assert clip.status == "approved" and clip.decided_at is not None


def test_reject_records_the_reason(env):
    cfg, client = env
    response = client.post("/clip/bbb/reject", data={"reason": "subtítulos tapados"})
    assert response.status_code == 200

    rejected = cfg.resolve(cfg.paths.output_rejected)
    assert (rejected / "bbb.mp4").exists()
    meta = json.loads((rejected / "bbb.json").read_text(encoding="utf-8"))
    assert meta["reject_reason"] == "subtítulos tapados"

    with session_scope() as s:
        clip = s.exec(select(Clip).where(Clip.clip_uid == "bbb")).first()
        assert clip.status == "rejected" and clip.reject_reason == "subtítulos tapados"


def test_reject_reason_buttons_come_from_config(env):
    cfg, client = env
    body = client.get("/").text
    for reason in cfg.review.reject_reasons:
        assert reason in body


def test_deciding_advances_to_the_next_clip(env):
    _, client = env
    body = client.post("/clip/aaa/approve", data={"caption": "x"}).text
    assert 'data-uid="bbb"' in body
    assert "aprobado" in body


def test_empty_queue_renders_a_placeholder(env):
    _, client = env
    client.post("/clip/aaa/approve", data={"caption": "x"})
    body = client.post("/clip/bbb/reject", data={"reason": "aburrido"}).text
    assert "Nada pendiente" in body


def test_media_route_serves_the_file(env):
    _, client = env
    assert client.get("/media/aaa.mp4").status_code == 200
    assert client.get("/media/nope.mp4").status_code == 404


def test_unknown_clip_is_404(env):
    _, client = env
    assert client.get("/clip/zzz").status_code == 404


def test_stats_page_and_numbers(env):
    _, client = env
    client.post("/clip/aaa/approve", data={"caption": "x"})
    client.post("/clip/bbb/reject", data={"reason": "mal encuadre"})

    data = compute_stats()
    assert data["approved"] == 1 and data["rejected"] == 1
    assert data["approval_rate"] == 50.0
    assert data["avg_score_approved"] == 90.0
    assert data["reject_reasons"] == {"mal encuadre": 1}

    body = client.get("/stats").text
    assert "50.0%" in body and "mal encuadre" in body


def test_healthz(env):
    _, client = env
    payload = client.get("/healthz").json()
    assert payload["ok"] is True and payload["pending"] == 2


# --------------------------------------------------------------------------- trends dashboard


@pytest.fixture
def trends_env(tmp_path):
    from clipforge.trends import refresh

    cfg = Config(project_root=tmp_path)
    cfg.ensure_dirs()
    init_engine(tmp_path / "dash.db")
    csv_path = cfg.resolve(cfg.trends.manual_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "title,year,slope,saturation\nOld Boy,2003,80,0.2\nEl Padrino,1972,35,0.85\n",
        encoding="utf-8",
    )
    with session_scope() as s:
        s.add(Video(file_hash="h", filename="Heat.mkv", path="/x", title="Heat", year=1995))
    refresh(cfg)
    return cfg, TestClient(create_app(cfg))


def test_dashboard_lists_titles_by_score(trends_env):
    _, client = trends_env
    body = client.get("/trends").text
    assert body.index("Old Boy") < body.index("El Padrino")   # 400 beats 41
    assert "saturación" in body and "pendiente" in body


def test_dashboard_is_sortable(trends_env):
    _, client = trends_env
    body = client.get("/trends?sort=title").text
    assert body.index("El Padrino") < body.index("Old Boy")
    assert 'href="/trends?sort=score' in body


def test_mark_as_to_download(trends_env):
    _, client = trends_env
    response = client.post("/trends/old-boy-2003/status", data={"status": "to_download"})
    assert response.status_code == 200
    assert "pendiente de descargar" in response.text

    from clipforge.trends import load_ranking

    row = next(r for r in load_ranking() if r.slug == "old-boy-2003")
    assert row.status == "to_download"


def test_unmark_and_ignore(trends_env):
    _, client = trends_env
    client.post("/trends/old-boy-2003/status", data={"status": "to_download"})
    client.post("/trends/old-boy-2003/status", data={"status": "new"})

    from clipforge.trends import load_ranking

    assert next(r for r in load_ranking() if r.slug == "old-boy-2003").status == "new"
    client.post("/trends/el-padrino-1972/status", data={"status": "ignored"})
    assert next(r for r in load_ranking() if r.slug == "el-padrino-1972").status == "ignored"


def test_processed_titles_are_marked_and_hideable(trends_env):
    cfg, client = trends_env
    csv_path = cfg.resolve(cfg.trends.manual_csv)
    csv_path.write_text("title,year,slope\nHeat,1995,90\n", encoding="utf-8")
    from clipforge.trends import refresh

    refresh(cfg)

    body = client.get("/trends").text
    assert "procesada" in body
    hidden = client.get("/trends?hide_processed=true").text
    assert "Heat" not in hidden


def test_unknown_title_status_is_404(trends_env):
    _, client = trends_env
    assert client.post("/trends/nope/status", data={"status": "to_download"}).status_code == 404
