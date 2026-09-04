import numpy as np
import pytest
from datetime import datetime, timezone

from src.utils.frames import (
    julian_date, gmst_rad, eci_to_ecef, ecef_to_geodetic, eci_to_geodetic,
    parse_epoch, epoch_plus_seconds, WGS84_A, WGS84_E2,
)

J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_julian_date_j2000():
    assert julian_date(J2000) == pytest.approx(2451545.0, abs=1e-6)


def test_gmst_at_j2000():
    # Standard reference value: 280.46062 deg
    assert np.rad2deg(gmst_rad(J2000)) == pytest.approx(280.46062, abs=1e-3)


def test_gmst_advances_one_sidereal_day():
    a = gmst_rad(J2000)
    b = gmst_rad(datetime(2000, 1, 2, 12, 0, 0, tzinfo=timezone.utc))
    # A solar day is ~4 min longer than a sidereal day.
    assert np.rad2deg((b - a) % (2 * np.pi)) == pytest.approx(0.9856, abs=1e-2)


def test_eci_ecef_identity_at_zero_rotation():
    r = np.array([1.0, 2.0, 3.0])
    assert np.allclose(eci_to_ecef(r, 0.0), r)


def test_eci_ecef_preserves_norm():
    r = np.array([7000.0, -1200.0, 300.0])
    for th in np.linspace(0, 2 * np.pi, 9):
        assert np.linalg.norm(eci_to_ecef(r, th)) == pytest.approx(np.linalg.norm(r))


@pytest.mark.parametrize("lat,lon,alt", [
    (0.0, 0.0, 500.0), (45.0, -73.5, 500.0),
    (-33.9, 151.2, 400.0), (81.0, 10.0, 800.0),
])
def test_geodetic_roundtrip(lat, lon, alt):
    sl, cl = np.sin(np.deg2rad(lat)), np.cos(np.deg2rad(lat))
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sl**2)
    r = np.array([(N + alt) * cl * np.cos(np.deg2rad(lon)),
                  (N + alt) * cl * np.sin(np.deg2rad(lon)),
                  (N * (1 - WGS84_E2) + alt) * sl])
    glat, glon, galt = ecef_to_geodetic(r)
    assert glat == pytest.approx(lat, abs=1e-6)
    assert glon == pytest.approx(lon, abs=1e-6)
    assert galt == pytest.approx(alt, abs=1e-4)


def test_altitude_is_plausible_for_leo():
    lat, lon, alt = eci_to_geodetic(np.array([6878.137, 0.0, 0.0]), J2000)
    assert 480.0 < alt < 520.0
    assert -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def test_parse_epoch_handles_z_suffix_and_naive():
    a = parse_epoch("2015-03-17T00:00:00Z")
    b = parse_epoch(datetime(2015, 3, 17))
    assert a == b and a.tzinfo is not None


def test_epoch_plus_seconds():
    t = epoch_plus_seconds(parse_epoch("2015-03-17T00:00:00Z"), 3600.0)
    assert t.hour == 1
