from __future__ import annotations

"""Time and reference-frame conversions.

The simulator propagates position in an Earth-centred inertial frame and
indexes time in seconds from an epoch.  Evaluating an empirical atmosphere
model requires geodetic latitude, longitude and altitude at an absolute UTC
instant, which needs Earth-rotation (GMST) and an ECEF-to-geodetic conversion.

Accuracy is at the arc-second / metre level, which is far finer than the
density model itself; NRLMSISE-00 has no meaningful sensitivity below about
a degree of latitude.
"""

from datetime import datetime, timedelta, timezone

import numpy as np

# WGS-84
WGS84_A = 6378.137          # semi-major axis, km
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2

EARTH_ROTATION_RATE = 7.2921158553e-5   # rad/s


def parse_epoch(value: str | datetime) -> datetime:
    """Parse an ISO-8601 epoch into a timezone-aware UTC datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def julian_date(utc: datetime) -> float:
    """Julian date from a UTC datetime."""
    utc = parse_epoch(utc)
    y, m = utc.year, utc.month
    d = (
        utc.day
        + (utc.hour + (utc.minute + (utc.second + utc.microsecond * 1e-6) / 60.0) / 60.0)
        / 24.0
    )
    if m <= 2:
        y -= 1
        m += 12
    a = int(y / 100)
    b = 2 - a + int(a / 4)
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


def gmst_rad(utc: datetime) -> float:
    """Greenwich Mean Sidereal Time in radians (IAU 1982 polynomial)."""
    jd = julian_date(utc)
    T = (jd - 2451545.0) / 36525.0
    # seconds of sidereal time
    gmst_s = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * T**2
        - 6.2e-6 * T**3
    )
    return np.deg2rad((gmst_s % 86400.0) / 240.0) % (2.0 * np.pi)


def eci_to_ecef(r_eci_km: np.ndarray, theta_rad: float) -> np.ndarray:
    """Rotate an ECI position into ECEF about the z-axis by GMST."""
    r = np.asarray(r_eci_km, dtype=float)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array(
        [
            c * r[0] + s * r[1],
            -s * r[0] + c * r[1],
            r[2],
        ]
    )


def ecef_to_geodetic(r_ecef_km: np.ndarray) -> tuple[float, float, float]:
    """WGS-84 geodetic latitude (deg), longitude (deg), altitude (km).

    Bowring's closed-form approximation; sub-millimetre for near-Earth orbits
    and free of the convergence checks an iterative solver would need.
    """
    x, y, z = (float(v) for v in np.asarray(r_ecef_km, dtype=float))
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    if p < 1e-12:                       # on the polar axis
        lat = np.pi / 2.0 * np.sign(z if z != 0.0 else 1.0)
        alt = abs(z) - WGS84_B
        return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(alt)

    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    lat = np.arctan2(
        z + WGS84_EP2 * WGS84_B * np.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3,
    )
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    lon_deg = (np.rad2deg(lon) + 180.0) % 360.0 - 180.0
    return float(np.rad2deg(lat)), float(lon_deg), float(alt)


def eci_to_geodetic(r_eci_km: np.ndarray, utc: datetime) -> tuple[float, float, float]:
    """Geodetic latitude (deg), longitude (deg), altitude (km) from ECI + UTC."""
    return ecef_to_geodetic(eci_to_ecef(r_eci_km, gmst_rad(utc)))


def epoch_plus_seconds(epoch: datetime, t_s: float) -> datetime:
    return parse_epoch(epoch) + timedelta(seconds=float(t_s))
