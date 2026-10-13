"""Module 5 — scoring. The core of the pipeline.

For every candidate segment we build a compact context (transcript window, audio summary,
relative position, duration, dialogue density) and ask Claude to score it on six axes and
hand back everything the editor needs: where to take the cold open from, the hook line, the
re-hook beats, a caption and hashtags.

Two things make this cheap enough to run on a whole film:

* **Iconic-scene priming.** Before touching the video we ask the model which scenes of this
  film people actually remember and roughly when they happen. That map only reorders the
  work — with ``--top N`` we score the promising zones first. An unrecognised title simply
  falls back to analysing everything.
* **Every verdict is persisted**, including segments that never get rendered, so weights can
  be re-tuned later without paying for the API again.

Model output is validated with Pydantic through the SDK's structured-output support; a bad
or failed response is retried with exponential backoff.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .audio_features import AudioFeatures
from .config import Config
from .db import Segment, SegmentScore, Video, log_error, select, session_scope
from .logging_utils import get_logger
from .scenes import Candidate
from .transcribe import Transcript

log = get_logger("scoring")

SCORE_FIELDS = (
    "hook_strength",
    "emotional_peak",
    "standalone_clarity",
    "quotability",
    "visual_interest",
    "sustains_3min",
)


# --------------------------------------------------------------------------- schemas


class Scores(BaseModel):
    hook_strength: int = Field(ge=0, le=100)
    emotional_peak: int = Field(ge=0, le=100)
    standalone_clarity: int = Field(ge=0, le=100)
    quotability: int = Field(ge=0, le=100)
    visual_interest: int = Field(ge=0, le=100)
    sustains_3min: int = Field(ge=0, le=100)


class Rehook(BaseModel):
    """A re-hook beat, timed relative to the start of the segment."""

    at_s: float = Field(ge=0)
    text: str = Field(min_length=1, max_length=90)


class SegmentVerdict(BaseModel):
    """Strict shape the model must return for one candidate segment."""

    scores: Scores
    cold_open_at_s: float = Field(ge=0, description="Peak visual tension, relative to start")
    hook_text: str = Field(min_length=3, max_length=90)
    rehook_suggestions: list[Rehook]
    suggested_caption: str = Field(min_length=3, max_length=300)
    suggested_hashtags: list[str]
    reason: str = Field(min_length=10, max_length=1200)

    @field_validator("suggested_hashtags")
    @classmethod
    def _normalise_hashtags(cls, tags: list[str]) -> list[str]:
        out: list[str] = []
        for tag in tags:
            tag = tag.strip().lstrip("#").strip()
            if tag:
                out.append(f"#{tag}")
        return out[:12]


class IconicScene(BaseModel):
    description: str
    approx_minute: float = Field(ge=0)
    why_memorable: str = ""


class IconicSceneMap(BaseModel):
    recognized: bool
    scenes: list[IconicScene] = Field(default_factory=list)


@dataclass(slots=True)
class ScoredSegment:
    """A verdict mapped back onto absolute timestamps in the source video."""

    segment_id: int
    idx: int
    start_s: float
    end_s: float
    total: float
    scores: dict[str, int]
    cold_open_timestamp: float
    hook_text: str
    rehook_suggestions: list[dict[str, Any]]
    suggested_caption: str
    suggested_hashtags: list[str]
    reason: str
    raw_response: str = ""


# --------------------------------------------------------------------------- prompts

SYSTEM_PROMPT = """\
Eres un editor de clips verticales para TikTok/Reels/Shorts especializado en cine. Tu trabajo \
es juzgar si un fragmento de una película funciona como clip vertical de ~3 minutos para \
alguien que hace scroll y NO ha visto la película.

Criterios de puntuación (0-100 cada uno):
- hook_strength: ¿los primeros segundos frenan el scroll?
- emotional_peak: ¿hay un pico emocional real (tensión, sorpresa, rabia, ternura)?
- standalone_clarity: ¿se entiende sin contexto previo de la película?
- quotability: ¿hay frases que la gente repetiría o citaría?
- visual_interest: ¿pasan cosas visualmente, o son dos personas quietas hablando?
- sustains_3min: ¿aguanta 3 minutos sin decaer, o se agota en 20 segundos?

Sé severo. La mayoría de los fragmentos de una película NO funcionan como clip: puntúa 30-50 \
lo mediocre y reserva 80+ para lo que de verdad retendría a un desconocido.

REGLAS DEL HOOK (hook_text): abre un hueco de información y NO lo resuelve. Es una \
observación concreta que obliga a seguir mirando.
- BIEN: "nadie se fija en lo que hace con la mano"
- BIEN: "escucha lo que dice justo antes de girarse"
- MAL: "escena épica de la película" (descriptivo, no abre hueco)
- MAL: "el protagonista mata al villano" (resuelve, spoiler)
Máximo 90 caracteres, minúsculas, sin emojis, sin nombre de la película.

rehook_suggestions: entre {rehooks_min} y {rehooks_max} beats repartidos por el fragmento \
(no todos al principio), cada uno con el segundo relativo al inicio del fragmento y un texto \
corto de anticipación que reinicie la atención ("espera al plano siguiente", "fíjate en el \
reflejo"). No repitas el hook_text.

cold_open_at_s: el segundo (relativo al inicio del fragmento) de MAYOR tensión visual, para \
abrir el vídeo con él.

suggested_caption: una línea para el pie del vídeo, en español, sin hashtags dentro.
suggested_hashtags: 5-8 hashtags en español, sin almohadilla duplicada.
reason: 2-4 frases explicando la decisión, mencionando lo que sostiene o hunde el fragmento.

TODOS los timestamps son segundos relativos al inicio del fragmento, entre 0 y su duración.
"""

ICONIC_PROMPT = """\
¿Reconoces la película "{title}"{year_part}?

Si la reconoces, lista sus escenas MÁS reconocibles (las que la gente recuerda, cita o busca \
en redes), con el minuto aproximado en que ocurren dentro de la película. Máximo {max_scenes}.
Si no la reconoces con seguridad, devuelve recognized=false y una lista vacía. No inventes.
"""


# --------------------------------------------------------------------------- client


def get_client(api_key: str | None = None):
    """Build an Anthropic client. Raises a clear error when the key is missing.

    Identity-linked API keys additionally require the workspace the request acts in; set
    ``ANTHROPIC_WORKSPACE_ID`` and it is sent as the ``anthropic-workspace-id`` header.
    Without it those keys fail with a 400 that names the missing header.
    """
    import anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Put it in your environment or in .env "
            "(see .env.example) before running the scoring stage."
        )
    headers = {}
    workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    if workspace:
        headers["anthropic-workspace-id"] = workspace
    return anthropic.Anthropic(api_key=key, default_headers=headers or None)


def _sleep_backoff(attempt: int, cfg: Config) -> None:
    delay = min(cfg.scoring.backoff_base_s * (2**attempt), cfg.scoring.backoff_max_s)
    time.sleep(delay + random.uniform(0, 0.5))


def _call_structured(
    client,
    cfg: Config,
    *,
    system: str | None,
    user: str,
    schema: type[BaseModel],
    label: str,
):
    """One structured-output call with retry/backoff. Returns a validated model instance."""
    import anthropic

    last_error: Exception | None = None
    for attempt in range(cfg.scoring.max_retries):
        try:
            kwargs: dict[str, Any] = {
                "model": cfg.scoring.model,
                "max_tokens": cfg.scoring.max_tokens,
                "messages": [{"role": "user", "content": user}],
                "output_format": schema,
            }
            if system:
                kwargs["system"] = system
            response = client.messages.parse(**kwargs)
            parsed = response.parsed_output
            if parsed is None:
                raise ValueError("model returned no parseable structured output")
            return parsed, response
        except (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APIStatusError,
        ) as exc:
            status = getattr(exc, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            last_error = exc
            log.warning("%s: API error (attempt %d): %s", label, attempt + 1, exc)
        except TypeError:  # bad call signature: retrying cannot help
            raise
        except Exception as exc:  # malformed/invalid JSON -> retry, it is often transient
            last_error = exc
            log.warning("%s: invalid response (attempt %d): %s", label, attempt + 1, exc)
        if attempt < cfg.scoring.max_retries - 1:
            _sleep_backoff(attempt, cfg)

    raise RuntimeError(f"{label}: giving up after {cfg.scoring.max_retries} attempts: {last_error}")


# --------------------------------------------------------------------------- iconic scenes


def _iconic_cache_path(cfg: Config, title: str, year: int | None) -> Path:
    d = cfg.resolve(cfg.paths.cache_dir) / "iconic"
    d.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "unknown"
    return d / f"{slug}.{year or 'na'}.json"


def get_iconic_scene_map(
    title: str,
    year: int | None,
    cfg: Config,
    client=None,
    force: bool = False,
) -> IconicSceneMap:
    """Ask the model which scenes of this film are memorable, and roughly when."""
    if not cfg.scoring.iconic_scenes.enabled:
        return IconicSceneMap(recognized=False)

    cache = _iconic_cache_path(cfg, title, year)
    if cache.exists() and not force:
        try:
            return IconicSceneMap.model_validate_json(cache.read_text(encoding="utf-8"))
        except Exception:
            log.warning("ignoring corrupt iconic-scene cache %s", cache.name)

    client = client or get_client()
    prompt = ICONIC_PROMPT.format(
        title=title,
        year_part=f" ({year})" if year else "",
        max_scenes=cfg.scoring.iconic_scenes.max_scenes,
    )
    try:
        scene_map, _ = _call_structured(
            client, cfg, system=None, user=prompt, schema=IconicSceneMap, label="iconic-scenes"
        )
    except Exception as exc:
        log.warning("iconic-scene map unavailable (%s) — analysing the whole film", exc)
        return IconicSceneMap(recognized=False)

    if not scene_map.recognized:
        log.info("title '%s' not recognised — analysing the whole film", title)
    else:
        log.info("iconic-scene map: %d known scenes for '%s'", len(scene_map.scenes), title)
    cache.write_text(scene_map.model_dump_json(), encoding="utf-8")
    return scene_map


def prioritize(
    candidates: list[Candidate], scene_map: IconicSceneMap, cfg: Config
) -> list[Candidate]:
    """Order candidates so the zones around known iconic scenes get analysed first.

    Priority is the number of iconic scenes whose approximate timestamp falls within
    ``priority_window_s`` of the segment, weighted by how close they land. Ties keep the
    original chronological order, so behaviour is identical for unrecognised films.
    """
    window = cfg.scoring.iconic_scenes.priority_window_s
    for cand in candidates:
        score = 0.0
        centre = (cand.start_s + cand.end_s) / 2
        for scene in scene_map.scenes:
            distance = abs(scene.approx_minute * 60.0 - centre)
            if distance <= window:
                score += 1.0 - (distance / window) * 0.5
        cand.priority = round(score, 4)
    return sorted(candidates, key=lambda c: (-c.priority, c.start_s))


# --------------------------------------------------------------------------- context building


def build_context(
    candidate: Candidate,
    transcript: Transcript,
    features: AudioFeatures,
    video_duration_s: float,
    title: str,
    year: int | None,
    cfg: Config,
) -> str:
    """Compact, token-bounded description of one candidate segment."""
    text = transcript.text_between(candidate.start_s, candidate.end_s)
    budget = cfg.scoring.transcript_char_budget
    if len(text) > budget:
        head = text[: int(budget * 0.6)]
        tail = text[-int(budget * 0.35) :]
        text = f"{head}\n[...corte...]\n{tail}"

    audio = features.summary(candidate.start_s, candidate.end_s)
    peaks = [
        round(p.t - candidate.start_s, 1)
        for p in features.peaks_between(candidate.start_s, candidate.end_s)
    ][:15]
    position = candidate.start_s / max(1.0, video_duration_s)

    payload = {
        "pelicula": {"titulo": title, "anio": year},
        "fragmento": {
            "duracion_s": round(candidate.duration_s, 1),
            "posicion_relativa_en_la_pelicula": round(position, 3),
            "planos": candidate.shot_count,
            "luminancia_media": round(candidate.mean_luma, 1),
        },
        "dialogo": {
            "palabras_por_segundo": round(
                transcript.dialogue_density(candidate.start_s, candidate.end_s), 2
            ),
            "fraccion_con_voz": round(
                transcript.speech_ratio(candidate.start_s, candidate.end_s), 3
            ),
        },
        "audio": audio,
        "picos_dramaticos_s": peaks,
    }
    return (
        "Datos del fragmento:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nTranscripción del fragmento:\n"
        + (text or "[sin diálogo]")
    )


# --------------------------------------------------------------------------- scoring


def weighted_total(scores: dict[str, int], cfg: Config) -> float:
    """Config-weighted 0-100 total. Computed locally so weights can be re-tuned offline."""
    weights = cfg.scoring.weights.normalized()
    return round(sum(scores.get(k, 0) * w for k, w in weights.items()), 2)


def _clamp_verdict(verdict: SegmentVerdict, duration: float, cfg: Config) -> SegmentVerdict:
    """Keep every timestamp inside the segment and trim the re-hook list to the configured range."""
    verdict.cold_open_at_s = min(max(0.0, verdict.cold_open_at_s), max(0.0, duration - 1.5))
    rehooks = [r for r in verdict.rehook_suggestions if 0 <= r.at_s <= duration]
    rehooks.sort(key=lambda r: r.at_s)
    verdict.rehook_suggestions = rehooks[: cfg.scoring.rehooks_max]
    return verdict


def score_segment(
    candidate: Candidate,
    segment_id: int,
    transcript: Transcript,
    features: AudioFeatures,
    video_duration_s: float,
    title: str,
    year: int | None,
    cfg: Config,
    client,
) -> ScoredSegment:
    """Score one candidate segment. Timestamps come back relative and are made absolute here."""
    context = build_context(
        candidate, transcript, features, video_duration_s, title, year, cfg
    )
    system = SYSTEM_PROMPT.format(
        rehooks_min=cfg.scoring.rehooks_min, rehooks_max=cfg.scoring.rehooks_max
    )
    verdict, response = _call_structured(
        client,
        cfg,
        system=system,
        user=context,
        schema=SegmentVerdict,
        label=f"segment {candidate.idx}",
    )
    verdict = _clamp_verdict(verdict, candidate.duration_s, cfg)

    scores = verdict.scores.model_dump()
    return ScoredSegment(
        segment_id=segment_id,
        idx=candidate.idx,
        start_s=candidate.start_s,
        end_s=candidate.end_s,
        total=weighted_total(scores, cfg),
        scores=scores,
        cold_open_timestamp=round(candidate.start_s + verdict.cold_open_at_s, 3),
        hook_text=verdict.hook_text,
        rehook_suggestions=[
            {"t": round(candidate.start_s + r.at_s, 3), "rel_t": round(r.at_s, 3), "text": r.text}
            for r in verdict.rehook_suggestions
        ],
        suggested_caption=verdict.suggested_caption,
        suggested_hashtags=verdict.suggested_hashtags,
        reason=verdict.reason,
        raw_response=verdict.model_dump_json(),
    )


def score_video(
    video_id: int,
    cfg: Config,
    transcript: Transcript,
    features: AudioFeatures,
    candidates: list[Candidate],
    top: int = 0,
    force: bool = False,
    client=None,
) -> list[ScoredSegment]:
    """Score the candidates of one video concurrently and persist every verdict."""
    with session_scope() as session:
        video = session.get(Video, video_id)
        if video is None:
            raise LookupError(f"video {video_id} not found")
        title, year, duration = video.display_title, video.year, video.duration_s
        rows = session.exec(select(Segment).where(Segment.video_id == video_id)).all()
        id_by_idx = {row.idx: row.id for row in rows}
        already = {
            s.segment_id
            for s in session.exec(
                select(SegmentScore).where(SegmentScore.video_id == video_id)
            ).all()
        }

    client = client or get_client()
    scene_map = get_iconic_scene_map(title, year, cfg, client=client)

    usable = [c for c in candidates if not c.discarded]
    ordered = prioritize(usable, scene_map, cfg)
    if top > 0:
        ordered = ordered[:top]

    pending = []
    for cand in ordered:
        seg_id = id_by_idx.get(cand.idx)
        if seg_id is None:
            continue
        if seg_id in already and not force:
            log.debug("segment %d already scored — skipping", cand.idx)
            continue
        pending.append((cand, seg_id))

    if not pending:
        log.info("nothing to score for video %d (use --force to rescore)", video_id)
        return load_scores(video_id)

    log.info(
        "scoring %d segments with %s (concurrency %d)",
        len(pending),
        cfg.scoring.model,
        cfg.scoring.concurrency,
    )

    results: list[ScoredSegment] = []
    with ThreadPoolExecutor(max_workers=max(1, cfg.scoring.concurrency)) as pool:
        futures = {
            pool.submit(
                score_segment,
                cand,
                seg_id,
                transcript,
                features,
                duration,
                title,
                year,
                cfg,
                client,
            ): cand
            for cand, seg_id in pending
        }
        for future in as_completed(futures):
            cand = futures[future]
            try:
                scored = future.result()
            except Exception as exc:  # one bad segment must not sink the video
                log.error("segment %d failed to score: %s", cand.idx, exc)
                log_error(video_id, "scoring", f"segment {cand.idx}: {exc}")
                continue
            results.append(scored)
            log.info(
                "segment %d [%.0fs-%.0fs] total=%.1f — %s",
                scored.idx,
                scored.start_s,
                scored.end_s,
                scored.total,
                scored.hook_text,
            )

    persist_scores(video_id, results, cfg, replace=force)
    results.sort(key=lambda s: -s.total)
    return results


# --------------------------------------------------------------------------- persistence


def persist_scores(
    video_id: int, scored: list[ScoredSegment], cfg: Config, replace: bool = False
) -> None:
    """Store every verdict — including segments that will never be rendered."""
    if not scored:
        return
    with session_scope() as session:
        if replace:
            ids = {s.segment_id for s in scored}
            for old in session.exec(
                select(SegmentScore).where(SegmentScore.video_id == video_id)
            ).all():
                if old.segment_id in ids:
                    session.delete(old)
            session.flush()
        for s in scored:
            session.add(
                SegmentScore(
                    video_id=video_id,
                    segment_id=s.segment_id,
                    model=cfg.scoring.model,
                    start_s=s.start_s,
                    end_s=s.end_s,
                    total=s.total,
                    scores=s.scores,
                    cold_open_timestamp=s.cold_open_timestamp,
                    hook_text=s.hook_text,
                    rehook_suggestions=s.rehook_suggestions,
                    suggested_caption=s.suggested_caption,
                    suggested_hashtags=s.suggested_hashtags,
                    reason=s.reason,
                    raw_response=s.raw_response,
                )
            )
        video = session.get(Video, video_id)
        if video:
            video.scoring_done = True
            session.add(video)


def load_scores(video_id: int, top: int = 0) -> list[ScoredSegment]:
    """Read stored verdicts back, best first — no API calls."""
    with session_scope() as session:
        rows = session.exec(
            select(SegmentScore).where(SegmentScore.video_id == video_id)
        ).all()
        out = [
            ScoredSegment(
                segment_id=r.segment_id,
                idx=-1,
                start_s=r.start_s,
                end_s=r.end_s,
                total=r.total,
                scores=dict(r.scores or {}),
                cold_open_timestamp=r.cold_open_timestamp,
                hook_text=r.hook_text,
                rehook_suggestions=list(r.rehook_suggestions or []),
                suggested_caption=r.suggested_caption,
                suggested_hashtags=list(r.suggested_hashtags or []),
                reason=r.reason,
                raw_response=r.raw_response,
            )
            for r in rows
        ]
    out.sort(key=lambda s: -s.total)
    return out[:top] if top > 0 else out


def recalibrate(video_id: int, cfg: Config) -> list[ScoredSegment]:
    """Recompute totals from stored per-axis scores after changing the weights in config."""
    with session_scope() as session:
        rows = session.exec(
            select(SegmentScore).where(SegmentScore.video_id == video_id)
        ).all()
        for row in rows:
            row.total = weighted_total(dict(row.scores or {}), cfg)
            session.add(row)
    return load_scores(video_id)
