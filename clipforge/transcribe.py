"""Module 3 — transcription.

faster-whisper with ``word_timestamps=True``. The word-level timing is not optional: the
subtitle renderer and the dead-air trimmer both key off it, so a transcript without word
timings is treated as a failure.

Results are cached by (file hash, model, language) under ``data/cache/transcripts/``.
The 16 kHz mono WAV extracted here is cached too and reused by ``audio_features``.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config
from .db import Video, session_scope
from .logging_utils import get_logger

log = get_logger("transcribe")


@dataclass(slots=True)
class Word:
    start: float
    end: float
    text: str
    prob: float = 0.0


@dataclass(slots=True)
class Utterance:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class Transcript:
    language: str = ""
    duration_s: float = 0.0
    model: str = ""
    words: list[Word] = field(default_factory=list)
    utterances: list[Utterance] = field(default_factory=list)

    # ---- windowing helpers used by scoring and by the editor

    def words_between(self, start: float, end: float) -> list[Word]:
        """Words whose midpoint falls inside [start, end)."""
        return [w for w in self.words if start <= (w.start + w.end) / 2 < end]

    def text_between(self, start: float, end: float) -> str:
        return " ".join(w.text.strip() for w in self.words_between(start, end)).strip()

    def utterances_between(self, start: float, end: float) -> list[Utterance]:
        return [u for u in self.utterances if u.end > start and u.start < end]

    def dialogue_density(self, start: float, end: float) -> float:
        """Words per second inside the window (0.0 for silent stretches)."""
        span = max(1e-6, end - start)
        return len(self.words_between(start, end)) / span

    def speech_ratio(self, start: float, end: float) -> float:
        """Fraction of the window covered by spoken words."""
        span = max(1e-6, end - start)
        covered = sum(
            min(w.end, end) - max(w.start, start)
            for w in self.words
            if w.end > start and w.start < end
        )
        return max(0.0, min(1.0, covered / span))

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "duration_s": self.duration_s,
            "model": self.model,
            "words": [asdict(w) for w in self.words],
            "utterances": [asdict(u) for u in self.utterances],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Transcript:
        return cls(
            language=data.get("language", ""),
            duration_s=data.get("duration_s", 0.0),
            model=data.get("model", ""),
            words=[Word(**w) for w in data.get("words", [])],
            utterances=[Utterance(**u) for u in data.get("utterances", [])],
        )


# --------------------------------------------------------------------------- audio extraction


def extract_audio(
    video_path: Path | str, cfg: Config, file_hash: str, force: bool = False
) -> Path:
    """Extract a 16 kHz mono WAV once and reuse it for transcription and audio features."""
    out_dir = cfg.resolve(cfg.paths.cache_dir) / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / f"{file_hash}.wav"
    if wav.exists() and wav.stat().st_size > 0 and not force:
        return wav

    cmd = [
        cfg.runtime.ffmpeg,
        "-y",
        "-v", "error",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(cfg.audio.sample_rate),
        "-c:a", "pcm_s16le",
        str(wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not wav.exists():
        raise RuntimeError(
            f"audio extraction failed: {proc.stderr.decode(errors='ignore')[-500:]}"
        )
    log.debug("extracted audio -> %s", wav.name)
    return wav


# --------------------------------------------------------------------------- transcription


def _cache_file(cfg: Config, file_hash: str, model: str, language: str | None) -> Path:
    d = cfg.resolve(cfg.paths.cache_dir) / "transcripts"
    d.mkdir(parents=True, exist_ok=True)
    lang = language or "auto"
    safe_model = model.replace("/", "_")
    return d / f"{file_hash}.{safe_model}.{lang}.json"


def _resolve_device(cfg: Config) -> tuple[str, str]:
    device = cfg.transcribe.device
    compute = cfg.transcribe.compute_type
    if device == "auto":
        try:  # torch is optional; absence just means CPU
            import torch  # noqa: PLC0415

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def transcribe_file(
    video_path: Path | str,
    cfg: Config,
    file_hash: str,
    force: bool = False,
) -> Transcript:
    """Transcribe one file, using the cache unless ``force``."""
    video_path = Path(video_path)
    cache = _cache_file(cfg, file_hash, cfg.transcribe.model, cfg.transcribe.language)

    if cfg.transcribe.cache and cache.exists() and not force:
        try:
            transcript = Transcript.from_dict(json.loads(cache.read_text(encoding="utf-8")))
            if transcript.words:
                log.info("transcript from cache (%d words)", len(transcript.words))
                return transcript
            log.warning("cached transcript has no word timings — retranscribing")
        except (json.JSONDecodeError, OSError, TypeError):
            log.warning("ignoring corrupt transcript cache %s", cache.name)

    if not cfg.transcribe.word_timestamps:
        raise ValueError(
            "transcribe.word_timestamps must stay true: subtitles and dead-air trimming "
            "depend on word-level timing"
        )

    from faster_whisper import WhisperModel

    wav = extract_audio(video_path, cfg, file_hash, force=force)
    device, compute_type = _resolve_device(cfg)
    log.info(
        "transcribing %s with whisper '%s' on %s/%s",
        video_path.name,
        cfg.transcribe.model,
        device,
        compute_type,
    )

    model = WhisperModel(cfg.transcribe.model, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(wav),
        language=cfg.transcribe.language,
        beam_size=cfg.transcribe.beam_size,
        vad_filter=cfg.transcribe.vad_filter,
        word_timestamps=True,
    )

    transcript = Transcript(
        language=getattr(info, "language", "") or "",
        duration_s=float(getattr(info, "duration", 0.0) or 0.0),
        model=cfg.transcribe.model,
    )
    for seg in segments:  # generator: this is where the work actually happens
        text = (seg.text or "").strip()
        if text:
            transcript.utterances.append(Utterance(float(seg.start), float(seg.end), text))
        for w in seg.words or []:
            word = (w.word or "").strip()
            if not word:
                continue
            transcript.words.append(
                Word(
                    start=float(w.start),
                    end=float(w.end),
                    text=word,
                    prob=float(getattr(w, "probability", 0.0) or 0.0),
                )
            )

    log.info(
        "transcribed: %d words, %d utterances, language=%s",
        len(transcript.words),
        len(transcript.utterances),
        transcript.language,
    )
    if cfg.transcribe.cache:
        cache.write_text(json.dumps(transcript.to_dict()), encoding="utf-8")
    return transcript


def transcribe_video(video_id: int, cfg: Config, force: bool = False) -> Transcript:
    """Module-3 entry point for a registered video."""
    with session_scope() as session:
        video = session.get(Video, video_id)
        if video is None:
            raise LookupError(f"video {video_id} not found")
        path, file_hash = Path(video.path), video.file_hash

    if not path.exists():
        raise FileNotFoundError(f"source file gone: {path}")

    transcript = transcribe_file(path, cfg, file_hash, force=force)

    with session_scope() as session:
        video = session.get(Video, video_id)
        if video:
            video.transcript_done = bool(transcript.words)
            session.add(video)
    return transcript


def load_cached(video_id: int, cfg: Config) -> Transcript | None:
    """Return the cached transcript for a video without running whisper."""
    with session_scope() as session:
        video = session.get(Video, video_id)
        if video is None:
            return None
        file_hash = video.file_hash
    cache = _cache_file(cfg, file_hash, cfg.transcribe.model, cfg.transcribe.language)
    if not cache.exists():
        return None
    try:
        return Transcript.from_dict(json.loads(cache.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, TypeError):
        return None
