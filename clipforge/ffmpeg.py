"""Thin wrappers around the ffmpeg/ffprobe binaries.

Command *building* is kept separate from command *running* so the builders can be unit
tested without touching ffmpeg (see tests/test_ffmpeg_cmd.py).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .logging_utils import get_logger

log = get_logger("ffmpeg")


class FFmpegError(RuntimeError):
    def __init__(self, cmd: list[str], returncode: int, stderr: str):
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        tail = "\n".join(stderr.strip().splitlines()[-12:])
        super().__init__(f"ffmpeg exited {returncode}\ncmd: {' '.join(cmd)}\n{tail}")


@dataclass(slots=True)
class MediaInfo:
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool
    size_bytes: int


def require_binaries(ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
    missing = [b for b in (ffmpeg, ffprobe) if shutil.which(b) is None]
    if missing:
        raise FileNotFoundError(
            f"binaries not found on PATH: {', '.join(missing)}. Install ffmpeg (see README)."
        )


def run(cmd: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    log.debug("run: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
    )
    if check and proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, proc.stderr or "")
    return proc


def _parse_fps(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe(path: Path | str, ffprobe: str = "ffprobe") -> MediaInfo:
    path = Path(path)
    cmd = [
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = run(cmd)
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or video.get("duration") or 0.0)
    return MediaInfo(
        duration_s=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        has_audio=has_audio,
        size_bytes=path.stat().st_size if path.exists() else 0,
    )
