"""Rich-based logging and progress helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

T = TypeVar("T")

console = Console()
_CONFIGURED = False


def setup_logging(level: str = "INFO", file: Path | None = None, rich_tracebacks: bool = True):
    global _CONFIGURED
    if _CONFIGURED:
        return logging.getLogger("clipforge")

    handlers: list[logging.Handler] = [
        RichHandler(
            console=console,
            rich_tracebacks=rich_tracebacks,
            show_path=False,
            markup=True,
            log_time_format="%H:%M:%S",
        )
    ]
    if file is not None:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        handlers.append(fh)

    logging.basicConfig(level=level.upper(), format="%(message)s", handlers=handlers, force=True)
    for noisy in ("httpx", "httpcore", "urllib3", "faster_whisper", "numba", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True
    return logging.getLogger("clipforge")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"clipforge.{name}")


def make_progress(enabled: bool = True) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not enabled,
        transient=False,
    )


def track(
    items: Iterable[T],
    description: str,
    total: int | None = None,
    enabled: bool = True,
) -> Iterator[T]:
    """Progress-wrapped iteration that degrades to a plain loop when disabled."""
    seq = list(items) if total is None else items
    total = total if total is not None else len(seq)  # type: ignore[arg-type]
    if not enabled:
        yield from seq  # type: ignore[misc]
        return
    with make_progress(enabled) as progress:
        task = progress.add_task(description, total=total)
        for item in seq:  # type: ignore[union-attr]
            yield item
            progress.advance(task)
