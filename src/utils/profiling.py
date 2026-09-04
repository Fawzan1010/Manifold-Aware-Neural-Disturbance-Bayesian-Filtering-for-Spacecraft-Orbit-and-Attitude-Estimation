from __future__ import annotations

"""Runtime and memory measurement helpers.

Timing a numerical estimator once, inline with the accuracy pass, is not
reliable: the first execution pays one-off costs (BLAS thread-pool creation,
lazy imports, page faults, allocator warm-up), and any drift in machine load
over the wall-clock span of the run is aliased onto whichever method happened
to be executing at the time.

``repeat_timing`` addresses the first effect by executing the callable
``repeats + discard_first`` times and dropping the leading samples.  It also
returns the minimum, which is the appropriate point estimate for the intrinsic
cost of an algorithm when the machine is not isolated: contention can only ever
make a sample slower, never faster.
"""

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Callable, Any

import numpy as np
import psutil

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass
class ProfileResult:
    runtime_s: float
    memory_mb: float


@dataclass
class TimingStats:
    """Summary of a repeated timing measurement.

    ``mean`` is the statistic requested for reporting; ``minimum`` is the
    contention-robust estimate and should be quoted alongside it.
    """

    mean: float
    std: float
    median: float
    minimum: float
    maximum: float
    n_kept: int
    n_discarded: int
    samples: tuple[float, ...] = ()

    def as_dict(self, prefix: str = "") -> dict[str, Any]:
        d = asdict(self)
        d.pop("samples")
        return {f"{prefix}{k}": v for k, v in d.items()}


@contextmanager
def profile_block():
    proc = psutil.Process()
    start_mem = proc.memory_info().rss
    start = time.perf_counter()
    yield lambda: ProfileResult(
        time.perf_counter() - start,
        (proc.memory_info().rss - start_mem) / (1024**2),
    )


@contextmanager
def pin_threads(n: int = 1):
    """Force single-threaded BLAS for the duration of a benchmark.

    Thread-pool scheduling is the largest single source of run-to-run variance
    in these measurements.  Values are restored on exit.

    Note: most BLAS backends read these variables when the library is first
    loaded, so this is only fully effective if applied before NumPy performs
    its first threaded operation.  It is still set here so the value is
    recorded with the results and so backends that honour it dynamically
    (OpenBLAS via ``threadpoolctl``-style updates) take effect.
    """
    previous = {k: os.environ.get(k) for k in _THREAD_ENV_VARS}
    for k in _THREAD_ENV_VARS:
        os.environ[k] = str(int(n))
    try:
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def repeat_timing(
    fn: Callable[[], Any],
    repeats: int = 5,
    discard_first: int = 1,
    collect_result: bool = False,
) -> tuple[TimingStats, Any]:
    """Execute ``fn`` repeatedly, discard the leading runs, summarise the rest.

    Parameters
    ----------
    fn
        Zero-argument callable performing the work to be timed.
    repeats
        Number of samples to *keep*.
    discard_first
        Number of leading samples to discard as warm-up.

    Returns
    -------
    (TimingStats, last_result)
        ``last_result`` is the return value of the final call when
        ``collect_result`` is set, otherwise ``None``.
    """
    repeats = max(int(repeats), 1)
    discard_first = max(int(discard_first), 0)
    total = repeats + discard_first

    samples: list[float] = []
    result = None
    for i in range(total):
        t0 = time.perf_counter()
        out = fn()
        elapsed = time.perf_counter() - t0
        samples.append(elapsed)
        if collect_result and i == total - 1:
            result = out

    kept = np.asarray(samples[discard_first:], dtype=float)
    stats = TimingStats(
        mean=float(np.mean(kept)),
        std=float(np.std(kept, ddof=1)) if kept.size > 1 else 0.0,
        median=float(np.median(kept)),
        minimum=float(np.min(kept)),
        maximum=float(np.max(kept)),
        n_kept=int(kept.size),
        n_discarded=int(discard_first),
        samples=tuple(float(s) for s in samples),
    )
    return stats, result


def current_memory_mb() -> float:
    return float(psutil.Process().memory_info().rss / (1024**2))
