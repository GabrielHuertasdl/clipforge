# clipforge

Local pipeline that turns long movie files into short vertical clips optimised for retention,
scores them with Claude, and hands them to you in a review UI. You supply the videos, you do
the uploading — clipforge only does the middle part.

**Out of scope, by design:** no video downloading or scraping, no publishing to any platform,
no multi-account management. Input comes from `input/raw/`; output stops at `output/approved/`.

## Requirements

- Python 3.11 or 3.12 (3.13+ is not supported yet by faster-whisper/librosa wheels)
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` and `ffprobe` on your `PATH`

Install ffmpeg:

```bash
# Arch
sudo pacman -S ffmpeg
# Debian/Ubuntu
sudo apt install ffmpeg
# macOS
brew install ffmpeg
```

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
make setup                      # uv venv + editable install + folder tree
cp .env.example .env            # then put your real key in it
```

`ANTHROPIC_API_KEY` is required for the scoring stage (`clipforge analyze`) and for the
`seed_llm` trend source. If yours is an identity-linked key, also set `ANTHROPIC_WORKSPACE_ID`
— the API rejects those keys with a 400 naming the missing workspace header. Everything else
runs offline.

## End-to-end example

```bash
# 1. Drop a long video into the input folder
cp ~/Videos/Heat.1995.1080p.BluRay.x264.mkv input/raw/

# 2. Register it (idempotent — already-known files are skipped unless --force)
uv run clipforge scan
#  id  title  duration  resolution
#   1  Heat   2:50:12   1920x1080

# 3. Scenes + transcription + audio features + scoring
uv run clipforge analyze 1

# 4. Render the 6 best candidates into vertical clips
uv run clipforge render 1 --top 6
#  -> output/pending/<clip_id>.mp4 + <clip_id>.json

# Or do all of it, plus source cleanup, in one shot:
uv run clipforge run 1 --top 6 --dry-run     # shows what it would delete
uv run clipforge run 1 --top 6               # deletes input/raw/Heat...mkv on full success
uv run clipforge run 1 --top 6 --keep-source # never deletes

# 5. Review (phase 2) — open from your phone at http://<local-ip>:8000
uv run clipforge review

# 6. Trend signals (phase 3)
uv run clipforge trends --refresh
uv run clipforge stats
```

## Configuration

Everything lives in `config.yaml`: scene thresholds, target durations, scoring weights,
subtitle style and safe zone, Whisper model, number of re-hooks, audio compression
aggressiveness, and the source-deletion policy. Defaults are defined in `clipforge/config.py`
and the YAML file deep-merges on top, so a partial config is always valid.

Point at a different file with `--config path/to.yaml` or `CLIPFORGE_CONFIG=...`.

## Layout

```
input/raw/          videos you drop in (deleted after a fully successful render)
input/manual/       manual CSVs for the trend detector
output/pending/     rendered clips awaiting review
output/approved/    approved clips + final caption JSON  <- your upload queue
output/rejected/    rejected clips + reason
data/clipforge.db   SQLite: videos, segments, every score, errors, deletions
data/cache/         transcripts, scene lists, audio features (keyed by file hash)
```

## Development

```bash
make test    # pytest
make lint    # ruff
make fmt     # ruff --fix + format
```

## Review UI

`uv run clipforge review` serves on `0.0.0.0:8000` and prints both URLs, including the LAN one
so you can review the verticals on your phone.

Keyboard: `A` approve · `R` reject (then `1`-`5` for the reason) · `E` edit the caption
(`Ctrl`/`Cmd`+`Enter` approves from the textarea) · `←` `→` navigate · `space` play/pause.

Approved clips move to `output/approved/` with the final caption in their JSON; rejected ones
move to `output/rejected/` with the reason. `uv run clipforge stats` (or `/stats` in the UI)
shows the approval rate, rejection reasons and average scores.

## Daily routine — two commands

```bash
uv run clipforge daily      # refresh trends, queue titles, process new videos, report
uv run clipforge review     # approve or reject the clips it produced
```

`daily` is the whole round in one command, and it is safe to run every day:

1. Refreshes every enabled trend source.
2. Auto-marks the best `daily.auto_queue` titles (default 3) as **pendiente de descargar**,
   skipping anything already processed, queued or ignored — so it tops the queue up instead
   of re-marking the same films.
3. Scans `input/raw/` and processes every new video end to end (analyze + render `--top 6`).
   A video that fails is recorded and the rest continue.
4. Prints what to download and how many clips are waiting for review.

It never deletes source files until you set `daily.keep_source: false` (or pass
`--delete-source`). `--dry-run` reports what it would do and changes nothing; `--skip-refresh`
reuses the stored signals instead of querying anything.

Your part of the loop stays manual by design: `daily` tells you which films to get, you get
them and drop them in `input/raw/`, and the next `daily` turns them into clips.

## Trend detector

Every source is an optional adapter; if one is disabled, missing a dependency or broken, the
run carries on without it.

| source | what it measures | default |
|---|---|---|
| `manual_csv` | titles you write by hand | on |
| `seed_llm` | Claude proposes candidate films with a reason, once a month | on |
| `anniversaries` | 10/20/25/30 years from release, ±30 days | on |
| `pytrends` | Google Trends slope over 7 days, `geo=ES` | on |
| `reddit` | r/movies + r/NetflixBestOf weekly top | off |
| `letterboxd` | popular this week | off |
| `justwatch` | arrival dates on Spanish catalogues | off |
| `tiktok_cc` | trending hashtags (context, not films) | off |

The three that depend on undocumented endpoints ship **off**; enable them in `config.yaml`
under `trends.sources` if you want them.

```
input/manual/trending.csv        title,year,slope,saturation,note   (only `title` is required)
input/manual/release_dates.csv   title,year,release_date,country    (300 films ship with the repo)
```

```bash
uv run clipforge trends --refresh          # query the enabled sources and rank
uv run clipforge trends --source reddit    # refresh one adapter, keeping the others' readings
uv run clipforge queue                     # the titles marked "pendiente de descargar"
```

Ranking: `score = trend_slope × (1 / saturation) × iconic_scene_density`. The slope is what
counts — a film climbing 20 → 45 outranks one flat at 80. Saturation comes from the source or
your CSV, and is otherwise estimated from how many sources are already shouting about it.
Iconic density reuses the scoring module's cached scene map, so it costs no API calls.

**How the sources feed each other**: `seed_llm` proposes new titles once a month, and they
enter with a fixed mid-table slope — its own confidence is never trusted as a score. On the
next refresh `pytrends` measures those titles for real and `anniversaries` adds any round
date, so a seeded film is ranked by measurement rather than by the model's opinion.

The dashboard lives at `/trends` in the same app: sortable table, a button to mark a title as
"pendiente de descargar", and films already in the database are greyed out as processed so
they never get queued twice.

## Status

- Phase 1 — engine (modules 1-6): done
- Phase 2 — review UI: done
- Phase 3 — trend detector + dashboard: done
