"""The daily routine: idempotency, failure isolation, and what it reports."""

from __future__ import annotations

import pytest

from clipforge.adapters.base import RawSignal
from clipforge.config import Config
from clipforge.daily import auto_queue, run_daily, videos_to_process
from clipforge.db import Clip, Video, init_engine, session_scope
from clipforge.trends import load_ranking, persist, rank, set_status


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = Config(project_root=tmp_path)
    cfg.ensure_dirs()
    init_engine(tmp_path / "daily.db")
    # Only the manual CSV is on: no network, no API key needed.
    cfg.trends.sources.seed_llm = False
    cfg.trends.sources.anniversaries = False
    cfg.trends.sources.pytrends = False
    csv_path = cfg.resolve(cfg.trends.manual_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "title,year,slope,saturation\n"
        "Old Boy,2003,80,0.2\n"
        "Heat,1995,70,0.3\n"
        "Zodiac,2007,55,0.25\n"
        "El Padrino,1972,35,0.85\n",
        encoding="utf-8",
    )
    return cfg


def add_video(cfg, name="Peli.2001.mkv", title="Peli", with_file=True, with_clip=False):
    path = cfg.resolve(cfg.paths.input_raw) / name
    if with_file:
        path.write_bytes(b"x" * 4096)
    with session_scope() as s:
        video = Video(
            file_hash=name, filename=name, path=str(path), title=title, duration_s=3600
        )
        s.add(video)
        s.flush()
        video_id = video.id
        if with_clip:
            s.add(
                Clip(
                    clip_uid=f"{name}-c", video_id=video_id, segment_id=1,
                    path=str(path) + ".mp4", json_path=str(path) + ".json",
                )
            )
    return video_id


def stub_pipeline(monkeypatch, clips=2, boom_on=None):
    """Replace the heavy pipeline with counters."""
    calls: list[int] = []

    def fake_render(video_id, cfg, transcript, features, top=6):
        calls.append(video_id)
        if boom_on == video_id:
            raise RuntimeError("ffmpeg exploded")
        return ([object()] * clips, [])

    monkeypatch.setattr("clipforge.scenes.build_segments", lambda *a, **k: [])
    monkeypatch.setattr("clipforge.transcribe.transcribe_video", lambda *a, **k: object())
    monkeypatch.setattr("clipforge.audio_features.analyze_video", lambda *a, **k: object())
    monkeypatch.setattr("clipforge.scoring.score_video", lambda *a, **k: [])
    monkeypatch.setattr("clipforge.edit.render_video", fake_render)
    return calls


# --------------------------------------------------------------------------- queueing


def test_queues_the_best_untouched_titles(env):
    persist(rank([], env), [])
    from clipforge.trends import refresh

    refresh(env)
    queued, already = auto_queue(env, 2, dry_run=False)
    assert [q.title for q in queued] == ["Old Boy", "Heat"]   # by score
    assert already == []
    assert {r.status for r in load_ranking() if r.title in ("Old Boy", "Heat")} == {"to_download"}


def test_running_twice_tops_up_instead_of_re_marking(env):
    from clipforge.trends import refresh

    refresh(env)
    first, _ = auto_queue(env, 2, dry_run=False)
    second, already = auto_queue(env, 2, dry_run=False)
    assert [q.title for q in first] == ["Old Boy", "Heat"]
    assert [q.title for q in second] == ["Zodiac", "El Padrino"]   # the next two
    assert {a.title for a in already} == {"Old Boy", "Heat"}


def test_processed_and_ignored_titles_are_never_queued(env):
    from clipforge.trends import refresh

    refresh(env)
    set_status("old-boy-2003", "ignored")
    with session_scope() as s:
        s.add(Video(file_hash="h", filename="Heat.mkv", path="/x", title="Heat", year=1995))
    refresh(env)

    queued, _ = auto_queue(env, 5, dry_run=False)
    titles = [q.title for q in queued]
    assert "Old Boy" not in titles      # ignored
    assert "Heat" not in titles         # already in the database


def test_dry_run_queues_nothing(env):
    from clipforge.trends import refresh

    refresh(env)
    queued, _ = auto_queue(env, 2, dry_run=True)
    assert [q.title for q in queued] == ["Old Boy", "Heat"]
    assert all(r.status == "new" for r in load_ranking())


# --------------------------------------------------------------------------- video selection


def test_only_unprocessed_videos_with_a_file_are_picked(env):
    ready = add_video(env, "Nueva.mkv", "Nueva")
    add_video(env, "YaHecha.mkv", "YaHecha", with_clip=True)
    add_video(env, "SinFichero.mkv", "SinFichero", with_file=False)
    assert [vid for vid, _ in videos_to_process(env)] == [ready]


# --------------------------------------------------------------------------- the routine


def test_full_routine_processes_and_reports(env, monkeypatch):
    video_id = add_video(env, "Nueva.mkv", "Nueva")
    calls = stub_pipeline(monkeypatch, clips=3)

    report = run_daily(env, top=2)
    assert report.refreshed is True
    assert [q.title for q in report.queued] == ["Old Boy", "Heat"]
    assert calls == [video_id]
    assert report.processed[0].clips == 3
    assert report.processed[0].source_deleted is False    # keep_source defaults to true


def test_one_bad_video_does_not_stop_the_others(env, monkeypatch):
    bad = add_video(env, "Rota.mkv", "Rota")
    good = add_video(env, "Buena.mkv", "Buena")
    calls = stub_pipeline(monkeypatch, boom_on=bad)

    report = run_daily(env)
    assert set(calls) == {bad, good}
    outcomes = {o.video_id: o for o in report.processed}
    assert outcomes[bad].errors and outcomes[bad].clips == 0
    assert outcomes[good].clips == 2


def test_dry_run_touches_nothing(env, monkeypatch):
    add_video(env, "Nueva.mkv", "Nueva")
    calls = stub_pipeline(monkeypatch)

    report = run_daily(env, dry_run=True)
    assert calls == []                                    # no rendering
    assert report.refreshed is False                      # no source was queried
    assert all(o.skipped == "dry run" for o in report.processed)
    assert all(r.status != "to_download" for r in load_ranking())


def test_second_run_does_not_reprocess_the_same_video(env, monkeypatch):
    add_video(env, "Nueva.mkv", "Nueva")

    def fake_render(video_id, cfg, transcript, features, top=6):
        with session_scope() as s:
            s.add(
                Clip(
                    clip_uid=f"c{video_id}", video_id=video_id, segment_id=1,
                    path="/tmp/x.mp4", json_path="/tmp/x.json", status="pending",
                )
            )
        return ([object()], [])

    stub_pipeline(monkeypatch)
    monkeypatch.setattr("clipforge.edit.render_video", fake_render)

    first = run_daily(env)
    second = run_daily(env)
    assert len(first.processed) == 1
    assert second.processed == []          # already has clips
    assert second.pending_clips == 1


def test_a_dead_trend_source_does_not_stop_the_routine(env, monkeypatch):
    add_video(env, "Nueva.mkv", "Nueva")
    stub_pipeline(monkeypatch)
    monkeypatch.setattr(
        "clipforge.daily.refresh", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("google 429"))
    )
    report = run_daily(env)
    assert any("trends" in e for e in report.errors)
    assert report.processed and report.processed[0].clips == 2


def test_skip_refresh_uses_stored_signals(env, monkeypatch):
    from clipforge.trends import refresh

    refresh(env)
    called = []
    monkeypatch.setattr("clipforge.daily.refresh", lambda *a, **k: called.append(1) or [])
    report = run_daily(env, skip_refresh=True)
    assert called == []
    assert [q.title for q in report.queued][:1] == ["Old Boy"]


def test_delete_source_can_be_forced_per_run(env, monkeypatch):
    add_video(env, "Nueva.mkv", "Nueva")
    stub_pipeline(monkeypatch)
    seen = {}
    monkeypatch.setattr(
        "clipforge.cleanup.cleanup_source",
        lambda video_id, cfg, keep_source=False, dry_run=False, had_errors=False: seen.setdefault(
            "keep", keep_source
        )
        or type("D", (), {"deleted": False})(),
    )
    run_daily(env, keep_source=False)
    assert seen["keep"] is False


def test_seed_titles_are_ranked_by_the_measuring_sources(env):
    """A seeded title enters as a candidate; a measured climb overtakes it."""
    seeded = RawSignal("Semilla", "seed_llm", 2010, slope=45.0, saturation=0.5)
    measured = RawSignal("Medida", "pytrends", 2011, slope=80.0)
    ranked = rank([seeded, measured], env)
    assert [r.title for r in ranked] == ["Medida", "Semilla"]
