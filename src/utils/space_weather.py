from __future__ import annotations

"""OMNI2 hourly space-weather loader.

Column mapping
--------------
OMNI2 documents its fields as 1-based *words*.  The three used here are

    word 50 -> ap index (nT, 3-hourly, repeated across the hours of a block)
    word 51 -> F10.7 daily solar radio flux (10^-22 W m^-2 Hz^-1)
    word 41 -> Dst (nT), retained for storm identification and reporting

which are zero-based column indices 49, 50 and 40.

F10.7A (the 81-day centred average) is **not** an OMNI2 field and is computed
here from the daily F10.7 series.

Fill values are encoded as a run of 9s sized to the field width (999.9 for
F10.7, 999 for ap, 99999 for Dst) and must be removed before use; they are
masked and filled by interpolation over the gap.
"""

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

COL_YEAR = 0
COL_DOY = 1
COL_HOUR = 2
COL_DST = 40
COL_AP = 49
COL_F107 = 50

# Any value at or above the per-field threshold is a fill marker.
FILL_THRESHOLD = {COL_AP: 999.0, COL_F107: 999.0, COL_DST: 99999.0}

F107A_WINDOW_DAYS = 81


@dataclass(frozen=True)
class SpaceWeatherRecord:
    time: datetime
    ap: float
    f107: float
    f107a: float
    dst: float = np.nan


def _fill_gaps(values: np.ndarray, valid: np.ndarray, label: str) -> np.ndarray:
    """Linearly interpolate across fill-value gaps; edge-extend at the ends."""
    out = values.astype(float).copy()
    if valid.all():
        return out
    if not valid.any():
        raise ValueError(f"OMNI column '{label}' contains no valid samples.")
    idx = np.arange(out.size)
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


class OmniDatabase:
    """In-memory OMNI2 hourly database with O(log n) time lookup.

    Values are held as NumPy arrays rather than per-hour objects: a full year
    is 8760 records and the density model queries it once per integration
    sub-step, so a linear scan per lookup would dominate the runtime.
    """

    def __init__(self, filename: str | Path):
        self.filename = Path(filename)
        (
            self.times,
            self.epochs,
            self.ap,
            self.f107,
            self.f107a,
            self.dst,
        ) = self._load()

    # -- loading ---------------------------------------------------------
    def _load(self):
        if not self.filename.exists():
            raise FileNotFoundError(f"OMNI2 data file not found: {self.filename}")

        data = np.loadtxt(self.filename)
        if data.ndim != 2 or data.shape[1] <= COL_F107:
            raise ValueError(
                f"{self.filename} has {data.shape[1] if data.ndim == 2 else '?'} columns; "
                f"expected at least {COL_F107 + 1} (OMNI2 hourly format)."
            )

        years = data[:, COL_YEAR].astype(int)
        doys = data[:, COL_DOY].astype(int)
        hours = data[:, COL_HOUR].astype(int)

        base = np.array(
            [datetime(int(y), 1, 1, tzinfo=timezone.utc) for y in np.unique(years)],
            dtype=object,
        )
        year_base = {b.year: b for b in base}
        times = np.array(
            [
                year_base[y] + timedelta(days=int(d) - 1, hours=int(h))
                for y, d, h in zip(years, doys, hours)
            ],
            dtype=object,
        )
        epochs = np.array([t.timestamp() for t in times], dtype=float)

        raw_ap = data[:, COL_AP]
        raw_f107 = data[:, COL_F107]
        raw_dst = data[:, COL_DST]

        ap = _fill_gaps(raw_ap, raw_ap < FILL_THRESHOLD[COL_AP], "ap")
        f107 = _fill_gaps(raw_f107, raw_f107 < FILL_THRESHOLD[COL_F107], "f107")
        dst = _fill_gaps(raw_dst, np.abs(raw_dst) < FILL_THRESHOLD[COL_DST], "dst")

        f107a = self._centred_81day(epochs, f107)
        return times, epochs, ap, f107, f107a, dst

    @staticmethod
    def _centred_81day(epochs: np.ndarray, f107: np.ndarray) -> np.ndarray:
        """81-day centred running mean of F10.7, evaluated hourly.

        Near the ends of the record the window is truncated rather than
        padded, so the average is taken over whatever days are available.
        """
        half = (F107A_WINDOW_DAYS // 2) * 86400.0
        out = np.empty_like(f107)
        lo_i = np.searchsorted(epochs, epochs - half, side="left")
        hi_i = np.searchsorted(epochs, epochs + half, side="right")
        csum = np.concatenate([[0.0], np.cumsum(f107)])
        counts = np.maximum(hi_i - lo_i, 1)
        out[:] = (csum[hi_i] - csum[lo_i]) / counts
        return out

    # -- lookup ----------------------------------------------------------
    def __len__(self) -> int:
        return int(self.epochs.size)

    @property
    def start(self) -> datetime:
        return self.times[0]

    @property
    def end(self) -> datetime:
        return self.times[-1]

    def _index(self, when: datetime) -> int:
        ts = when.timestamp() if when.tzinfo else when.replace(tzinfo=timezone.utc).timestamp()
        i = int(np.searchsorted(self.epochs, ts, side="left"))
        if i <= 0:
            return 0
        if i >= self.epochs.size:
            return int(self.epochs.size - 1)
        return i if abs(self.epochs[i] - ts) < abs(ts - self.epochs[i - 1]) else i - 1

    def _floor_index(self, when: datetime) -> int:
        """Index of the last sample at or before ``when`` (step-hold)."""
        ts = when.timestamp() if when.tzinfo else when.replace(tzinfo=timezone.utc).timestamp()
        i = int(np.searchsorted(self.epochs, ts, side="right")) - 1
        return int(np.clip(i, 0, self.epochs.size - 1))

    def _record(self, i: int, when: datetime | None = None) -> SpaceWeatherRecord:
        return SpaceWeatherRecord(
            time=when if when is not None else self.times[i],
            ap=float(self.ap[i]),
            f107=float(self.f107[i]),
            f107a=float(self.f107a[i]),
            dst=float(self.dst[i]),
        )

    def nearest(self, when: datetime) -> SpaceWeatherRecord:
        return self._record(self._index(when))

    def step_hold(self, when: datetime) -> SpaceWeatherRecord:
        """Zero-order hold.

        ap is a 3-hourly *index*, constant within its block; OMNI repeats each
        value across the three hours.  Linearly interpolating it would invent
        intermediate activity levels that the index does not represent, so
        step-hold is the physically correct sampling for ap.
        """
        return self._record(self._floor_index(when), when)

    def interpolate(self, when: datetime) -> SpaceWeatherRecord:
        """Linear interpolation of the smooth solar-flux channels.

        ap and Dst are held rather than interpolated, for the reason above.
        """
        ts = when.timestamp() if when.tzinfo else when.replace(tzinfo=timezone.utc).timestamp()
        if ts <= self.epochs[0]:
            return self._record(0, when)
        if ts >= self.epochs[-1]:
            return self._record(int(self.epochs.size - 1), when)
        j = int(np.searchsorted(self.epochs, ts, side="right"))
        i = j - 1
        span = self.epochs[j] - self.epochs[i]
        a = 0.0 if span <= 0 else (ts - self.epochs[i]) / span
        return SpaceWeatherRecord(
            time=when,
            ap=float(self.ap[i]),
            f107=float((1 - a) * self.f107[i] + a * self.f107[j]),
            f107a=float((1 - a) * self.f107a[i] + a * self.f107a[j]),
            dst=float(self.dst[i]),
        )

    def ap_history(self, when: datetime) -> np.ndarray:
        """The 7-element Ap array NRLMSISE-00 expects.

        Layout required by the model:
            [0] daily Ap
            [1] current 3-hour ap
            [2] ap 3 hours before
            [3] ap 6 hours before
            [4] ap 9 hours before
            [5] mean of the eight ap values 12-33 hours before
            [6] mean of the eight ap values 36-57 hours before

        Storm-time density responds to the recent history, not just the daily
        mean, so this is what makes 17 March 2015 look like a storm to the
        model rather than a moderately active day.
        """
        i = self._floor_index(when)

        def at(hours_back: int) -> float:
            return float(self.ap[max(i - hours_back, 0)])

        daily_start = max(i - 23, 0)
        daily = float(np.mean(self.ap[daily_start : i + 1]))
        prior_12_33 = float(np.mean(self.ap[max(i - 33, 0) : max(i - 11, 1)]))
        prior_36_57 = float(np.mean(self.ap[max(i - 57, 0) : max(i - 35, 1)]))

        return np.array(
            [daily, at(0), at(3), at(6), at(9), prior_12_33, prior_36_57],
            dtype=float,
        )


@lru_cache(maxsize=8)
def load_omni(filename: str) -> OmniDatabase:
    """Cached loader so repeated construction does not re-parse the file."""
    return OmniDatabase(filename)
