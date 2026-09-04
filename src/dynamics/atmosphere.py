from __future__ import annotations

"""Atmospheric density models.

Two implementations behind one interface:

``ExponentialAtmosphere``
    The analytic scale-height profile with a scalar activity index.  Cheap,
    smooth and differentiable, and the default so that existing results are
    reproduced bit-for-bit.

``NRLMSISE00Atmosphere``
    NRLMSISE-00 evaluated through ``pymsis``, driven by measured F10.7 and Ap
    from an OMNI2 record.  Requires an absolute epoch and a geodetic position,
    which is why the simulator carries ``epoch_utc``.

Selecting between them is the job of :func:`make_atmosphere`; nothing outside
this module needs to know which one is active.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import numpy as np

from src.utils.frames import eci_to_geodetic, epoch_plus_seconds, parse_epoch

# NRLMSISE-00 is version 0 in pymsis.  The library default is 2.1, which is a
# different model; passing this explicitly is required.
MSIS_NRLMSISE00 = 0

# Index of total mass density (kg/m^3) in the pymsis output vector.
MSIS_MASS_DENSITY = 0


class AtmosphereModel(ABC):
    """Density as a function of inertial position and mission-elapsed time."""

    name: str = "abstract"

    @abstractmethod
    def density(self, r_eci_km: np.ndarray, t_s: float, weather_index: float = 0.0) -> float:
        """Mass density in kg/m^3."""

    def describe(self) -> dict[str, Any]:
        return {"model": self.name}


class ExponentialAtmosphere(AtmosphereModel):
    """Scale-height profile referenced to 400 km.

    ``weather_index`` is the simulator's dimensionless activity proxy; this is
    the model the trained networks saw, so its behaviour is frozen.
    """

    name = "exponential"

    def __init__(self, rho0: float = 3.614e-13, h_scale: float = 88.0,
                 ref_alt_km: float = 400.0, radius_earth: float = 6378.137,
                 activity_gain: float = 0.9):
        self.rho0 = float(rho0)
        self.h_scale = float(h_scale)
        self.ref_alt_km = float(ref_alt_km)
        self.radius_earth = float(radius_earth)
        self.activity_gain = float(activity_gain)

    def density(self, r_eci_km: np.ndarray, t_s: float, weather_index: float = 0.0) -> float:
        alt = float(np.linalg.norm(r_eci_km)) - self.radius_earth
        rho0 = self.rho0 * (1.0 + self.activity_gain * float(weather_index))
        return float(rho0 * np.exp(-(alt - self.ref_alt_km) / self.h_scale))

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "rho0": self.rho0,
            "scale_height_km": self.h_scale,
            "reference_altitude_km": self.ref_alt_km,
        }


class NRLMSISE00Atmosphere(AtmosphereModel):
    """NRLMSISE-00 driven by measured OMNI2 F10.7 and Ap."""

    name = "nrlmsise00"

    def __init__(
        self,
        omni,
        epoch_utc: datetime | str,
        use_ap_history: bool = True,
        cache_seconds: float = 60.0,
        cache_altitude_km: float = 5.0,
        cache_angle_deg: float = 5.0,
        min_altitude_km: float = 0.0,
        max_altitude_km: float = 1000.0,
    ):
        try:
            import pymsis  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "NRLMSISE-00 requires the 'pymsis' package: pip install pymsis"
            ) from exc

        self.omni = omni
        self.epoch = parse_epoch(epoch_utc)
        self.use_ap_history = bool(use_ap_history)
        self.cache_seconds = float(cache_seconds)
        self.cache_altitude_km = float(cache_altitude_km)
        self.cache_angle_deg = float(cache_angle_deg)
        self.min_altitude_km = float(min_altitude_km)
        self.max_altitude_km = float(max_altitude_km)
        self._cache: dict[tuple, float] = {}
        self._calls = 0
        self._hits = 0

        self._check_coverage()

    def _check_coverage(self) -> None:
        if not (self.omni.start <= self.epoch <= self.omni.end):
            raise ValueError(
                f"Epoch {self.epoch.isoformat()} lies outside the OMNI record "
                f"{self.omni.start.isoformat()} .. {self.omni.end.isoformat()}."
            )

    # -- core ------------------------------------------------------------
    def _evaluate(self, when: datetime, lat: float, lon: float, alt_km: float) -> float:
        import pymsis

        rec = self.omni.step_hold(when)
        if self.use_ap_history:
            aps = self.omni.ap_history(when).reshape(1, 7)
            options = [1.0] * 25
            options[9] = -1.0          # enable the 7-element ap history
        else:
            aps = np.array([rec.ap], dtype=float)
            options = None

        out = pymsis.calculate(
            np.array([np.datetime64(when.replace(tzinfo=None), "s")]),
            np.array([lon], dtype=float),
            np.array([lat], dtype=float),
            np.array([alt_km], dtype=float),
            np.array([rec.f107], dtype=float),
            np.array([rec.f107a], dtype=float),
            aps,
            version=MSIS_NRLMSISE00,
            options=options,
        )
        rho = float(np.asarray(out).ravel()[MSIS_MASS_DENSITY])
        return rho if np.isfinite(rho) else 0.0

    def density(self, r_eci_km: np.ndarray, t_s: float, weather_index: float = 0.0) -> float:
        when = epoch_plus_seconds(self.epoch, t_s)
        lat, lon, alt_km = eci_to_geodetic(r_eci_km, when)

        if alt_km < self.min_altitude_km or alt_km > self.max_altitude_km:
            return 0.0

        # Quantised cache key: the model is called once per RK4 sub-step, and
        # density varies far more slowly than the integrator samples it.  The
        # key is deterministic, so runs remain reproducible.
        key = (
            int(t_s // self.cache_seconds) if self.cache_seconds > 0 else t_s,
            round(lat / self.cache_angle_deg),
            round(lon / self.cache_angle_deg),
            round(alt_km / self.cache_altitude_km),
        )
        self._calls += 1
        hit = self._cache.get(key)
        if hit is not None:
            self._hits += 1
            return hit

        rho = self._evaluate(when, lat, lon, alt_km)
        self._cache[key] = rho
        return rho

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "msis_version": MSIS_NRLMSISE00,
            "epoch_utc": self.epoch.isoformat(),
            "space_weather_file": str(getattr(self.omni, "filename", "")),
            "use_ap_history": self.use_ap_history,
            "cache_calls": self._calls,
            "cache_hit_rate": (self._hits / self._calls) if self._calls else 0.0,
        }


def make_atmosphere(cfg: dict, override_model: str | None = None) -> AtmosphereModel:
    """Build the atmosphere model named in ``cfg['atmosphere']``.

    ``override_model`` lets one experiment run truth and filter dynamics under
    different atmospheres without mutating the shared config.
    """
    acfg = dict(cfg.get("atmosphere", {}) or {})
    model = (override_model or acfg.get("model", "exponential")).lower()

    if model in {"exponential", "exp", "none"}:
        sim = cfg.get("simulation", {}) or {}
        return ExponentialAtmosphere(
            radius_earth=float(sim.get("radius_earth", 6378.137)),
            rho0=float(acfg.get("rho0", 3.614e-13)),
            h_scale=float(acfg.get("scale_height_km", 88.0)),
            ref_alt_km=float(acfg.get("reference_altitude_km", 400.0)),
        )

    if model in {"nrlmsise00", "nrlmsise", "msis", "msis00"}:
        from src.utils.space_weather import load_omni

        path = acfg.get("space_weather_file", "data/space_weather/omni2_2015.dat")
        epoch = cfg.get("simulation", {}).get("epoch_utc") or acfg.get("epoch_utc")
        if epoch is None:
            raise ValueError(
                "NRLMSISE-00 needs an absolute epoch; set simulation.epoch_utc."
            )
        return NRLMSISE00Atmosphere(
            omni=load_omni(str(path)),
            epoch_utc=epoch,
            use_ap_history=bool(acfg.get("use_ap_history", True)),
            cache_seconds=float(acfg.get("cache_seconds", 60.0)),
        )

    raise ValueError(f"Unknown atmosphere model: {model!r}")
