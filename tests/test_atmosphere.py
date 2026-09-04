import numpy as np
import pytest
from pathlib import Path

from src.dynamics.atmosphere import (
    make_atmosphere, ExponentialAtmosphere, NRLMSISE00Atmosphere, MSIS_NRLMSISE00,
)
from src.dynamics.spacecraft import SpacecraftParams, atmospheric_density, drag_acceleration

DATA = Path("data/space_weather/omni2_2015.dat")
R = np.array([6378.137 + 500.0, 0.0, 0.0])

CFG = {
    "simulation": {"radius_earth": 6378.137, "epoch_utc": "2015-03-17T00:00:00Z"},
    "atmosphere": {
        "model": "exponential",
        "space_weather_file": str(DATA),
        "use_ap_history": True,
        "cache_seconds": 60.0,
    },
}


def test_factory_returns_requested_model():
    assert isinstance(make_atmosphere(CFG), ExponentialAtmosphere)
    assert make_atmosphere(CFG).name == "exponential"


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        make_atmosphere({**CFG, "atmosphere": {"model": "banana"}})


@pytest.mark.parametrize("wi", [0.0, 0.1, 0.7, 1.0, 1.6])
def test_exponential_matches_legacy_formula_exactly(wi):
    """The default path must not perturb previously published results."""
    model = make_atmosphere(CFG)
    params = SpacecraftParams()
    assert model.density(R, 0.0, wi) == atmospheric_density(R, wi, params)


def test_exponential_decays_with_altitude():
    m = make_atmosphere(CFG)
    lo = m.density(np.array([6378.137 + 300.0, 0, 0]), 0.0, 0.5)
    hi = m.density(np.array([6378.137 + 700.0, 0, 0]), 0.0, 0.5)
    assert lo > hi > 0.0


def test_drag_unit_scale_matches_si_calculation():
    """0.5*rho*Cd*A/m has units 1/m; speed is km/s, so a 1e3 factor is needed."""
    p = SpacecraftParams()
    v = np.array([0.0, 7.6, 0.0])
    a_km = np.linalg.norm(drag_acceleration(R, v, 0.1, p))
    rho = atmospheric_density(R, 0.1, p)
    a_si = 0.5 * rho * p.cd * (p.area / p.mass) * (7600.0 ** 2)
    assert a_km * 1e3 == pytest.approx(a_si, rel=1e-9)


def test_drag_opposes_velocity():
    p = SpacecraftParams()
    v = np.array([0.0, 7.6, 0.0])
    assert np.dot(drag_acceleration(R, v, 0.1, p), v) < 0.0


@pytest.fixture(scope="module")
def msis():
    pytest.importorskip("pymsis")
    return make_atmosphere(CFG, override_model="nrlmsise00")


@pytest.mark.skipif(not DATA.exists(), reason="OMNI2 data not present")
class TestNRLMSISE00:
    def test_uses_nrlmsise00_not_msis21(self, msis):
        # pymsis defaults to version 2.1, which is a different model.
        assert MSIS_NRLMSISE00 == 0
        assert msis.describe()["msis_version"] == 0

    def test_density_is_physical_at_leo(self, msis):
        rho = msis.density(R, 0.0)
        assert 1e-14 < rho < 1e-10

    def test_density_increases_through_the_storm(self, msis):
        quiet = msis.density(R, 0.0)
        storm = msis.density(R, 21 * 3600.0)
        assert storm > quiet * 1.2

    def test_decays_with_altitude(self, msis):
        lo = msis.density(np.array([6378.137 + 300.0, 0, 0]), 0.0)
        hi = msis.density(np.array([6378.137 + 700.0, 0, 0]), 0.0)
        assert lo > hi > 0.0

    def test_returns_zero_outside_altitude_guards(self, msis):
        assert msis.density(np.array([6378.137 + 5000.0, 0, 0]), 0.0) == 0.0

    def test_cache_is_deterministic(self, msis):
        a = msis.density(R, 100.0)
        b = msis.density(R, 100.0)
        assert a == b

    def test_epoch_outside_record_raises(self):
        pytest.importorskip("pymsis")
        cfg = {**CFG, "simulation": {**CFG["simulation"],
                                     "epoch_utc": "2020-01-01T00:00:00Z"}}
        with pytest.raises(ValueError, match="outside the OMNI record"):
            make_atmosphere(cfg, override_model="nrlmsise00")

    def test_denser_than_exponential_at_500km(self, msis):
        """The exponential profile is tuned low at 500 km; this is the model
        error the mismatched storm pairing is designed to expose."""
        assert msis.density(R, 0.0) > make_atmosphere(CFG).density(R, 0.0, 0.1)
