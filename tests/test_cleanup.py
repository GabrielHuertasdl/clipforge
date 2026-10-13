from __future__ import annotations

import pytest

from clipforge.cleanup import cleanup_source, verify_outputs
from clipforge.config import Config
from clipforge.db import (
    Clip,
    ProcessingError,
    SourceDeletion,
    Video,
    init_engine,
    select,
    session_scope,
)


@pytest.fixture
def env(tmp_path):
    """A throwaway project: its own database, source file and rendered outputs."""
    cfg = Config(project_root=tmp_path)
    cfg.ensure_dirs()
    init_engine(tmp_path / "test.db")

    source = cfg.resolve(cfg.paths.input_raw) / "Peli.2011.mkv"
    source.write_bytes(b"x" * 5000)

    with session_scope() as s:
        video = Video(file_hash="h1", filename=source.name, path=str(source), title="Peli")
        s.add(video)
        s.flush()
        video_id = video.id
    return cfg, video_id, source


def add_clip(cfg, video_id, uid="c1", size=4096, with_json=True):
    out = cfg.resolve(cfg.paths.output_pending)
    mp4 = out / f"{uid}.mp4"
    mp4.write_bytes(b"v" * size)
    js = out / f"{uid}.json"
    if with_json:
        js.write_text("{}", encoding="utf-8")
    with session_scope() as s:
        s.add(
            Clip(
                clip_uid=uid, video_id=video_id, segment_id=1,
                path=str(mp4), json_path=str(js), duration_s=180.0, size_bytes=size,
            )
        )
    return mp4


def test_deletes_source_when_everything_rendered(env):
    cfg, video_id, source = env
    add_clip(cfg, video_id)
    decision = cleanup_source(video_id, cfg)
    assert decision.deleted is True
    assert not source.exists()
    with session_scope() as s:
        rows = [(r.file_hash, r.dry_run) for r in s.exec(select(SourceDeletion)).all()]
        deleted_at = s.get(Video, video_id).source_deleted_at
    assert rows == [("h1", False)]
    assert deleted_at is not None


def test_keep_source_never_deletes(env):
    cfg, video_id, source = env
    add_clip(cfg, video_id)
    decision = cleanup_source(video_id, cfg, keep_source=True)
    assert decision.deleted is False and source.exists()
    assert "keep-source" in decision.reason


def test_dry_run_reports_without_deleting(env):
    cfg, video_id, source = env
    add_clip(cfg, video_id)
    decision = cleanup_source(video_id, cfg, dry_run=True)
    assert decision.deleted is False and source.exists()
    assert decision.bytes_freed == 5000
    with session_scope() as s:
        dry_flags = [r.dry_run for r in s.exec(select(SourceDeletion)).all()]
    assert dry_flags == [True]  # the intent is still recorded


def test_errors_block_deletion(env):
    cfg, video_id, source = env
    add_clip(cfg, video_id)
    with session_scope() as s:
        s.add(ProcessingError(video_id=video_id, stage="edit", message="boom"))
    decision = cleanup_source(video_id, cfg)
    assert decision.deleted is False and source.exists()
    assert "error" in decision.reason


def test_errors_passed_in_block_deletion(env):
    cfg, video_id, source = env
    add_clip(cfg, video_id)
    assert cleanup_source(video_id, cfg, had_errors=True).deleted is False
    assert source.exists()


def test_no_clips_means_no_deletion(env):
    cfg, video_id, source = env
    decision = cleanup_source(video_id, cfg)
    assert decision.deleted is False and source.exists()
    assert "no clips" in decision.reason


def test_missing_output_file_blocks_deletion(env):
    cfg, video_id, source = env
    mp4 = add_clip(cfg, video_id)
    mp4.unlink()
    decision = cleanup_source(video_id, cfg)
    assert decision.deleted is False and source.exists()
    assert "missing output" in decision.reason


def test_zero_byte_output_blocks_deletion(env):
    cfg, video_id, source = env
    mp4 = add_clip(cfg, video_id)
    mp4.write_bytes(b"")
    decision = cleanup_source(video_id, cfg)
    assert decision.deleted is False and source.exists()
    assert "too small" in decision.reason


def test_missing_metadata_blocks_deletion(env):
    cfg, video_id, source = env
    add_clip(cfg, video_id, uid="c2", with_json=False)
    decision = cleanup_source(video_id, cfg)
    assert decision.deleted is False and source.exists()


def test_protected_extension_is_never_deleted(env):
    cfg, video_id, source = env
    cfg.cleanup.keep_source_extensions = [".mkv"]
    add_clip(cfg, video_id)
    decision = cleanup_source(video_id, cfg)
    assert decision.deleted is False and source.exists()


def test_verify_outputs_counts_clips(tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x" * 2000)
    j = tmp_path / "a.json"
    j.write_text("{}", encoding="utf-8")
    ok, reason, count = verify_outputs([(f, j)], 1024)
    assert ok and count == 1 and reason == ""
