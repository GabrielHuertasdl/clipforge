"""Module 1 — ingest.

Watches ``input/raw/``, registers every video in the database with its hash, duration,
resolution and date, and guesses the movie title from the filename.

Idempotent: a file whose hash is already known is skipped unless ``force=True``. Hashing is
partial by default (first and last N bytes plus the file size) because full sha256 of an 8 GB
remux is slow and adds nothing for de-duplication.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .db import Video, log_error, select, session_scope, utcnow
from .ffmpeg import MediaInfo, probe
from .logging_utils import get_logger

log = get_logger("ingest")

# Separators used by release naming: dots, underscores, dashes between words.
_SEP_RE = re.compile(r"[._]+")
_MULTISPACE_RE = re.compile(r"\s{2,}")
_TRAILING_JUNK_RE = re.compile(r"[\s\-_.]+$")
_LEADING_JUNK_RE = re.compile(r"^[\s\-_.]+")


@dataclass(slots=True)
class ScanResult:
    video_id: int
    path: Path
    title: str
    year: int | None
    duration_s: float
    width: int
    height: int
    status: str  # "new" | "skipped" | "updated" | "error"
    reason: str = ""


def file_hash(path: Path, chunk_bytes: int = 8 * 1024 * 1024, full: bool = False) -> str:
    """Content hash. Partial by default: size + head chunk + tail chunk."""
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as fh:
        if full or size <= chunk_bytes * 2:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(block)
        else:
            h.update(fh.read(chunk_bytes))
            fh.seek(-chunk_bytes, 2)
            h.update(fh.read(chunk_bytes))
    return h.hexdigest()


def extract_year(stem: str, year_pattern: str) -> tuple[int | None, str]:
    """Pull a release year out of the filename and return it with the year removed.

    Only the *last* year-looking token is treated as the release year, so titles such as
    "2001 A Space Odyssey 1968" keep their leading number.
    """
    matches = list(re.finditer(year_pattern, stem))
    if not matches:
        return None, stem
    m = matches[-1]
    # A year at the very start is part of the title, not metadata.
    if m.start() == 0:
        return None, stem
    year = int(m.group(1))
    cleaned = stem[: m.start()] + " " + stem[m.end() :]
    return year, cleaned


def parse_title(filename: str, cfg: Config) -> tuple[str, int | None]:
    """Best-effort movie title from a release-style filename.

    ``Heat.1995.1080p.BluRay.x264-GROUP.mkv`` -> ``("Heat", 1995)``
    """
    stem = Path(filename).stem
    stem = _SEP_RE.sub(" ", stem)

    year, stem = extract_year(stem, cfg.ingest.title_year_pattern)

    for pattern in cfg.ingest.title_strip_patterns:
        stem = re.sub(pattern, " ", stem)

    # Anything after a release-group dash at the tail ("- GROUP") is noise.
    stem = re.sub(r"\s+-\s*\w+\s*$", " ", stem)

    # Everything from the first surviving quality/codec-ish token onwards is noise too.
    stem = _MULTISPACE_RE.sub(" ", stem)
    stem = _LEADING_JUNK_RE.sub("", stem)
    stem = _TRAILING_JUNK_RE.sub("", stem)
    stem = stem.strip(" -_")

    if not stem:
        stem = Path(filename).stem
    # Title-case only if the filename was shouting or all lowercase; keep mixed case as is.
    if stem.isupper() or stem.islower():
        stem = stem.title()
    return stem, year


def iter_video_files(cfg: Config) -> list[Path]:
    raw_dir = cfg.resolve(cfg.paths.input_raw)
    exts = {e.lower() for e in cfg.ingest.extensions}
    if not raw_dir.exists():
        return []
    return sorted(
        p
        for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")
    )


def register_file(path: Path, cfg: Config, force: bool = False) -> ScanResult:
    """Register one file. Never raises: failures are logged and returned as status='error'."""
    try:
        digest = file_hash(
            path, chunk_bytes=cfg.ingest.hash_chunk_bytes, full=cfg.ingest.hash_full
        )
    except OSError as exc:
        log_error(None, "ingest", f"cannot hash {path.name}", str(exc))
        return ScanResult(-1, path, "", None, 0, 0, 0, "error", str(exc))

    with session_scope() as session:
        existing = session.exec(select(Video).where(Video.file_hash == digest)).first()
        if existing and not force:
            return ScanResult(
                existing.id or -1,
                path,
                existing.display_title,
                existing.year,
                existing.duration_s,
                existing.width,
                existing.height,
                "skipped",
                "already registered",
            )

    try:
        info: MediaInfo = probe(path, ffprobe=cfg.runtime.ffprobe)
    except Exception as exc:
        log_error(None, "ingest", f"ffprobe failed on {path.name}", str(exc))
        return ScanResult(-1, path, "", None, 0, 0, 0, "error", str(exc))

    if info.duration_s < cfg.ingest.min_duration_s:
        return ScanResult(
            -1,
            path,
            "",
            None,
            info.duration_s,
            info.width,
            info.height,
            "skipped",
            f"shorter than min_duration_s ({info.duration_s:.0f}s)",
        )

    title, year = parse_title(path.name, cfg)

    with session_scope() as session:
        video = session.exec(select(Video).where(Video.file_hash == digest)).first()
        if video is None:
            video = Video(file_hash=digest)
            status = "new"
        else:
            status = "updated"
        video.filename = path.name
        video.path = str(path)
        video.duration_s = info.duration_s
        video.width = info.width
        video.height = info.height
        video.fps = info.fps
        video.size_bytes = info.size_bytes
        if not video.title or force:
            video.title = title
            video.year = year
        video.source_deleted_at = None
        video.updated_at = utcnow()
        session.add(video)
        session.flush()
        vid = video.id or -1
        display = video.display_title
        vyear = video.year

    log.info(
        "%s: %s (%s) — %.0fs %dx%d",
        status,
        display,
        vyear or "?",
        info.duration_s,
        info.width,
        info.height,
    )
    return ScanResult(vid, path, display, vyear, info.duration_s, info.width, info.height, status)


def scan(cfg: Config, force: bool = False) -> list[ScanResult]:
    """Scan input/raw/ and register everything new. One bad file never stops the rest."""
    files = iter_video_files(cfg)
    log.info("scanning %s — %d candidate file(s)", cfg.resolve(cfg.paths.input_raw), len(files))
    results: list[ScanResult] = []
    for path in files:
        try:
            results.append(register_file(path, cfg, force=force))
        except Exception as exc:  # defensive: register_file already swallows the usual ones
            log_error(None, "ingest", f"unexpected failure on {path.name}", repr(exc))
            results.append(ScanResult(-1, path, "", None, 0, 0, 0, "error", repr(exc)))
            if cfg.runtime.fail_fast:
                raise
    return results


def set_title(video_id: int, title: str) -> str:
    """Manual override of the auto-detected title."""
    with session_scope() as session:
        video = session.get(Video, video_id)
        if video is None:
            raise LookupError(f"video {video_id} not found")
        video.title_override = title.strip() or None
        video.updated_at = utcnow()
        session.add(video)
        session.flush()
        return video.display_title
