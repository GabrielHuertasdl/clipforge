"""SQLite persistence (SQLModel).

Everything the pipeline learns is stored here so any phase can be re-run without redoing
paid or slow work: scene lists, scores for *every* candidate segment (even ones that never
get rendered), per-video errors, and the record of deleted source files.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Column, UniqueConstraint

# ``select`` is re-exported so callers only need to import from clipforge.db.
from sqlmodel import Field, Session, SQLModel, create_engine, select  # noqa: F401

_ENGINE = None
_ENGINE_PATH: Path | None = None


def utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- models


class Video(SQLModel, table=True):
    """One source file in input/raw/."""

    __tablename__ = "video"

    id: int | None = Field(default=None, primary_key=True)
    file_hash: str = Field(index=True, unique=True)
    filename: str
    path: str
    title: str = ""
    title_override: str | None = None
    year: int | None = None
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    size_bytes: int = 0
    status: str = Field(default="new", index=True)  # new|analyzing|analyzed|rendering|done|failed
    scenes_done: bool = False
    transcript_done: bool = False
    audio_done: bool = False
    scoring_done: bool = False
    source_deleted_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @property
    def display_title(self) -> str:
        return self.title_override or self.title or Path(self.filename).stem


class Segment(SQLModel, table=True):
    """Candidate segment: a run of whole shots, 150-260 s, never cutting mid-shot."""

    __tablename__ = "segment"
    __table_args__ = (UniqueConstraint("video_id", "idx", name="uq_segment_video_idx"),)

    id: int | None = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    idx: int
    start_s: float
    end_s: float
    duration_s: float
    shot_start: int
    shot_end: int
    shot_count: int = 0
    dark_ratio: float = 0.0
    mean_luma: float = 0.0
    discarded: bool = False
    discard_reason: str | None = None
    priority: float = 0.0  # from the iconic-scene map; higher = analyse first
    created_at: datetime = Field(default_factory=utcnow)


class SegmentScore(SQLModel, table=True):
    """Model verdict for a segment. Kept for every scored segment, rendered or not."""

    __tablename__ = "segment_score"

    id: int | None = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    segment_id: int = Field(foreign_key="segment.id", index=True)
    model: str = ""
    start_s: float = 0.0
    end_s: float = 0.0
    total: float = Field(default=0.0, index=True)
    scores: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    cold_open_timestamp: float = 0.0
    hook_text: str = ""
    rehook_suggestions: list[Any] = Field(default_factory=list, sa_column=Column(JSON))
    suggested_caption: str = ""
    suggested_hashtags: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    reason: str = ""
    raw_response: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class Clip(SQLModel, table=True):
    """A rendered vertical clip awaiting (or past) review."""

    __tablename__ = "clip"

    id: int | None = Field(default=None, primary_key=True)
    clip_uid: str = Field(index=True, unique=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    segment_id: int = Field(foreign_key="segment.id", index=True)
    score_id: int | None = Field(default=None, foreign_key="segment_score.id")
    path: str = ""
    json_path: str = ""
    thumb_path: str = ""
    duration_s: float = 0.0
    size_bytes: int = 0
    total_score: float = 0.0
    caption: str = ""
    hashtags: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    variants: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: str = Field(default="pending", index=True)  # pending|approved|rejected
    reject_reason: str | None = None
    reject_note: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    decided_at: datetime | None = None


class ProcessingError(SQLModel, table=True):
    """One failure. A failing video never takes the batch down; it lands here."""

    __tablename__ = "processing_error"

    id: int | None = Field(default=None, primary_key=True)
    video_id: int | None = Field(default=None, foreign_key="video.id", index=True)
    stage: str = Field(index=True)
    message: str = ""
    detail: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class SourceDeletion(SQLModel, table=True):
    """Audit trail of deleted source files, so we know what was already processed."""

    __tablename__ = "source_deletion"

    id: int | None = Field(default=None, primary_key=True)
    video_id: int | None = Field(default=None, foreign_key="video.id", index=True)
    filename: str
    file_hash: str = Field(index=True)
    size_bytes: int = 0
    clips_rendered: int = 0
    dry_run: bool = False
    deleted_at: datetime = Field(default_factory=utcnow)


class TrendTitle(SQLModel, table=True):
    """A film worth making clips of, ranked by the trend detector."""

    __tablename__ = "trend_title"
    __table_args__ = (UniqueConstraint("slug", name="uq_trend_title_slug"),)

    id: int | None = Field(default=None, primary_key=True)
    slug: str = Field(index=True)
    title: str
    year: int | None = None
    score: float = Field(default=0.0, index=True)
    trend_slope: float = 0.0
    saturation: float = 0.5
    iconic_density: float = 1.0
    top_source: str = ""
    sources: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    detail: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: str = Field(default="new", index=True)  # new|to_download|processed|ignored
    processed_video_id: int | None = None
    first_seen: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class TrendSignal(SQLModel, table=True):
    """One raw reading from one source, kept so scores can be recomputed offline."""

    __tablename__ = "trend_signal"

    id: int | None = Field(default=None, primary_key=True)
    slug: str = Field(index=True)
    source: str = Field(index=True)
    title: str = ""
    year: int | None = None
    value: float = 0.0
    slope: float = 0.0
    saturation: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- engine


def init_engine(db_path: Path | str, echo: bool = False):
    global _ENGINE, _ENGINE_PATH
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _ENGINE is not None and _ENGINE_PATH == path:
        return _ENGINE
    _ENGINE = create_engine(
        f"sqlite:///{path}",
        echo=echo,
        connect_args={"check_same_thread": False},
    )
    _ENGINE_PATH = path
    SQLModel.metadata.create_all(_ENGINE)
    return _ENGINE


def get_engine():
    if _ENGINE is None:
        raise RuntimeError("database engine not initialised; call init_engine() first")
    return _ENGINE


@contextmanager
def session_scope() -> Iterator[Session]:
    session = Session(get_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def log_error(video_id: int | None, stage: str, message: str, detail: str = "") -> None:
    """Record a failure without ever raising (error logging must not break the pipeline)."""
    try:
        with session_scope() as s:
            s.add(
                ProcessingError(
                    video_id=video_id,
                    stage=stage,
                    message=str(message)[:2000],
                    detail=detail[:8000],
                )
            )
    except Exception:  # pragma: no cover - last-resort guard
        pass
