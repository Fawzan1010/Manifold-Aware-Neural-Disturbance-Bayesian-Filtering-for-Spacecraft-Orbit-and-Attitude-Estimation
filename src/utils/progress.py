from __future__ import annotations

"""Console progress reporting.

Purely presentational: nothing in this module touches the numerical path.
Every helper here writes to the console only, so removing the calls would
leave the produced tables, figures and JSON reports bit-for-bit identical.

Three things are provided:

``progress`` / ``progress_bar``
    Thin wrappers around :mod:`tqdm` with one shared bar format, so every
    loop in the repository reports percentage, completed/total iterations,
    elapsed time, ETA and rate in the same way.  Bars are written to stderr
    (tqdm's default) so that anything a stage prints to stdout, and every
    file it writes, stays uncontaminated.

``stage_header`` / ``stage_complete`` / ``StageTimer``
    The banner and the "Completed successfully / Elapsed Time" footer that
    delimit the seven pipeline stages.

``format_elapsed`` / ``banner`` / ``section``
    Small formatting helpers shared by the above.
"""

import sys
import time
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

# NOTE: the plain console tqdm, never ``tqdm.auto``.  The auto variant pulls in
# the notebook/ipywidgets machinery, which perturbs the global torch RNG stream
# and therefore reshuffles DataLoader batch order -- changing training results.
from tqdm import tqdm

__all__ = [
    "RULE_WIDTH",
    "BAR_FORMAT",
    "format_elapsed",
    "banner",
    "section",
    "progress",
    "progress_bar",
    "stage_header",
    "stage_complete",
    "stage_failed",
    "StageTimer",
    "stage",
    "pipeline_header",
    "pipeline_summary",
    "info",
]

RULE_WIDTH = 60

# percentage | bar | completed/total | elapsed<ETA | rate
BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}]"
)

# Same information for loops whose length is not known ahead of time.
BAR_FORMAT_NOTOTAL = "{desc}: {n_fmt} [{elapsed}, {rate_fmt}]"


def format_elapsed(seconds: float) -> str:
    """Seconds as HH:MM:SS (hours are not wrapped at 24)."""
    seconds = max(float(seconds), 0.0)
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def banner(title: str, subtitle: str | None = None, width: int = RULE_WIDTH) -> None:
    """A rule-delimited title block."""
    rule = "=" * width
    print(f"\n{rule}")
    if subtitle is not None:
        print(subtitle)
    print(title)
    print(rule, flush=True)


def section(message: str) -> None:
    """A short, indented note inside a stage (used between major steps)."""
    print(f"  {message}", flush=True)


def info(label: str, value: Any) -> None:
    """A ``label: value`` line inside the opening summary."""
    print(f"{label}:\n{value}\n", flush=True)


def progress(
    iterable: Iterable,
    desc: str,
    total: int | None = None,
    leave: bool = True,
    unit: str = "it",
    disable: bool = False,
    position: int | None = None,
) -> Iterable:
    """Wrap ``iterable`` in a tqdm bar with the repository-wide format.

    ``leave=False`` is the right choice for any loop nested inside another
    reporting loop: the bar erases itself on completion so that the outer
    bar is not pushed off by one line per inner pass.
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=leave,
        unit=unit,
        disable=disable,
        position=position,
        dynamic_ncols=True,
        bar_format=BAR_FORMAT if total is not None else BAR_FORMAT_NOTOTAL,
        file=sys.stderr,
    )


def progress_bar(
    desc: str,
    total: int | None = None,
    leave: bool = True,
    unit: str = "it",
    disable: bool = False,
) -> tqdm:
    """A manually driven bar, for sequences that are not a single loop."""
    return tqdm(
        desc=desc,
        total=total,
        leave=leave,
        unit=unit,
        disable=disable,
        dynamic_ncols=True,
        bar_format=BAR_FORMAT if total is not None else BAR_FORMAT_NOTOTAL,
        file=sys.stderr,
    )


def stage_header(index: int, total: int, name: str, width: int = RULE_WIDTH) -> None:
    """The header printed before each pipeline stage."""
    rule = "=" * width
    print(f"\n{rule}")
    print(f"Stage {index}/{total}")
    print(name.upper())
    print(rule, flush=True)


def stage_complete(elapsed_s: float) -> None:
    print("\nCompleted successfully.\n")
    print("Elapsed Time:")
    print(format_elapsed(elapsed_s), flush=True)


def stage_failed(index: int, total: int, name: str, elapsed_s: float) -> None:
    """Report which stage was running when an exception escaped.

    The exception itself is never swallowed; this only labels it.
    """
    print(f"\nFAILED during Stage {index}/{total}: {name.upper()}")
    print("Elapsed Time:")
    print(format_elapsed(elapsed_s), flush=True)


class StageTimer:
    """Wall-clock timer for one stage."""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start


@contextmanager
def stage(index: int, total: int, name: str) -> Iterator[StageTimer]:
    """Header, timing and footer for one stage.

    On failure the stage is named and the exception is re-raised unchanged.
    """
    stage_header(index, total, name)
    timer = StageTimer()
    try:
        yield timer
    except BaseException:
        stage_failed(index, total, name, timer.elapsed)
        raise
    stage_complete(timer.elapsed)


def pipeline_header(title: str, config_path: Any, mode: str, output_dir: Any,
                    width: int = RULE_WIDTH) -> None:
    rule = "=" * width
    print(f"\n{rule}")
    print(title.upper())
    print(f"{rule}\n")
    info("Configuration", config_path)
    info("Execution Mode", mode)
    info("Output Directory", output_dir)
    print("Project Initialized", flush=True)


def pipeline_summary(completed: int, total: int, elapsed_s: float,
                     output_dir: Any, width: int = RULE_WIDTH) -> None:
    rule = "=" * width
    print(f"\n{rule}")
    print("PIPELINE COMPLETE")
    print(f"{rule}\n")
    info("Stages Completed", f"{completed}/{total}")
    info("Total Runtime", format_elapsed(elapsed_s))
    info("Results Saved To", output_dir)
