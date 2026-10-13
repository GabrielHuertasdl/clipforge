"""Source cleanup.

After every clip of a video has rendered without a single error, the original file in
``input/raw/`` is deleted. The guard rails, in order:

* no errors anywhere in the pipeline for that video;
* at least one clip rendered, and **every** output file exists with size > 0;
* ``--keep-source`` disables deletion entirely;
* ``--dry-run`` reports what would be deleted without touching anything;
* the deletion is recorded in ``source_deletion`` (name, hash, size, date) so we know the
  file was already processed and never re-download it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .db import Clip, ProcessingError, SourceDeletion, Video, select, session_scope, utcnow
from .logging_utils import get_logger

log = get_logger("cleanup")


@dataclass(slots=True)
class CleanupDecision:
    deleted: bool
    dry_run: bool
    path: Path | None
    reason: str
    clips_verified: int = 0
    bytes_freed: int = 0


def verify_outputs(
    clip_files: list[tuple[Path, Path]], min_bytes: int
) -> tuple[bool, str, int]:
    """Every clip must exist on disk with a real size before we delete anything.

    Takes plain paths rather than ORM rows: the caller reads them inside its session, so
    nothing here can trip over a detached instance.
    """
    if not clip_files:
        return False, "no clips were rendered", 0
    for path, json_path in clip_files:
        if not path.exists():
            return False, f"missing output {path.name}", 0
        if path.stat().st_size < min_bytes:
            return False, f"output too small: {path.name} ({path.stat().st_size} bytes)", 0
        if not json_path.exists() or json_path.stat().st_size == 0:
            return False, f"missing metadata for {path.name}", 0
    return True, "", len(clip_files)


def cleanup_source(
    video_id: int,
    cfg: Config,
    keep_source: bool = False,
    dry_run: bool = False,
    had_errors: bool = False,
) -> CleanupDecision:
    """Decide on, and possibly perform, the deletion of one source file."""
    with session_scope() as session:
        video = session.get(Video, video_id)
        if video is None:
            raise LookupError(f"video {video_id} not found")
        path = Path(video.path)
        file_hash, filename = video.file_hash, video.filename
        already_deleted = video.source_deleted_at is not None
        clip_files = [
            (Path(c.path), Path(c.json_path))
            for c in session.exec(select(Clip).where(Clip.video_id == video_id)).all()
        ]
        error_count = len(
            list(
                session.exec(
                    select(ProcessingError).where(ProcessingError.video_id == video_id)
                ).all()
            )
        )

    if already_deleted or not path.exists():
        return CleanupDecision(False, dry_run, path, "source already gone")
    if keep_source or not cfg.cleanup.delete_source:
        return CleanupDecision(False, dry_run, path, "keep-source is set")
    if path.suffix.lower() in {e.lower() for e in cfg.cleanup.keep_source_extensions}:
        return CleanupDecision(False, dry_run, path, f"extension {path.suffix} is protected")
    if had_errors or (cfg.cleanup.require_all_clips_ok and error_count):
        return CleanupDecision(
            False, dry_run, path, f"pipeline had {error_count or 'some'} error(s)"
        )

    ok, reason, verified = verify_outputs(clip_files, cfg.cleanup.min_output_bytes)
    if not ok:
        return CleanupDecision(False, dry_run, path, reason)

    size = path.stat().st_size
    if dry_run:
        log.info("[dry-run] would delete %s (%.1f MB)", path.name, size / 1e6)
        with session_scope() as session:
            session.add(
                SourceDeletion(
                    video_id=video_id,
                    filename=filename,
                    file_hash=file_hash,
                    size_bytes=size,
                    clips_rendered=verified,
                    dry_run=True,
                )
            )
        return CleanupDecision(False, True, path, "dry run", verified, size)

    path.unlink()
    with session_scope() as session:
        session.add(
            SourceDeletion(
                video_id=video_id,
                filename=filename,
                file_hash=file_hash,
                size_bytes=size,
                clips_rendered=verified,
                dry_run=False,
            )
        )
        video = session.get(Video, video_id)
        if video:
            video.source_deleted_at = utcnow()
            video.status = "done"
            session.add(video)
    log.info("deleted source %s (%.1f MB freed, %d clips verified)", filename, size / 1e6, verified)
    return CleanupDecision(True, False, path, "deleted", verified, size)
