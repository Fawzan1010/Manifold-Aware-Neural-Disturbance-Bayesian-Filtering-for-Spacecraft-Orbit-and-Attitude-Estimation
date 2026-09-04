import numpy as np
import pytest
from datetime import datetime, timezone
from pathlib import Path

from src.utils.space_weather import load_omni, OmniDatabase, COL_AP, COL_F107, COL_DST

DATA = Path("data/space_weather/omni2_2015.dat")
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="OMNI2 data not present")


@pytest.fixture(scope="module")
def db():
    return load_omni(str(DATA))


def test_full_year_loaded(db):
    assert len(db) == 8760
    assert db.start.year == 2015 and db.end.year == 2015


def test_column_indices_are_the_documented_words():
    # OMNI2 documents 1-based words; these are the 0-based equivalents.
    assert (COL_DST, COL_AP, COL_F107) == (40, 49, 50)


def test_storm_peak_ap(db):
    rec = db.step_hold(datetime(2015, 3, 17, 12, tzinfo=timezone.utc))
    assert rec.ap == pytest.approx(179.0)


def test_dst_minimum_identifies_st_patricks_day_storm(db):
    i = int(np.argmin(db.dst))
    assert db.times[i].month == 3 and db.times[i].day == 17
    assert db.dst[i] < -200.0


def test_fill_values_are_removed(db):
    # 999.9 / 999 / 99999 are fill markers, not measurements.
    assert db.f107.max() < 999.0
    assert db.ap.max() < 999.0
    assert np.abs(db.dst).max() < 99999.0
    assert np.isfinite(db.f107).all()


def test_f107a_is_a_centred_average_not_a_column(db):
    assert db.f107a.min() > db.f107.min()
    assert db.f107a.max() < db.f107.max()
    assert db.f107a.mean() == pytest.approx(db.f107.mean(), rel=0.05)


def test_ap_history_layout(db):
    h = db.ap_history(datetime(2015, 3, 17, 12, tzinfo=timezone.utc))
    assert h.shape == (7,)
    assert np.isfinite(h).all()
    assert h[1] == pytest.approx(179.0)   # current 3-hour ap
    assert h[0] < h[1]                    # daily mean below the storm peak


def test_step_hold_is_constant_within_a_three_hour_block(db):
    vals = [db.step_hold(datetime(2015, 3, 17, h, tzinfo=timezone.utc)).ap
            for h in (12, 13, 14)]
    assert len(set(vals)) == 1


def test_interpolate_does_not_invent_intermediate_ap(db):
    a = db.interpolate(datetime(2015, 3, 17, 13, tzinfo=timezone.utc)).ap
    b = db.step_hold(datetime(2015, 3, 17, 13, tzinfo=timezone.utc)).ap
    assert a == b


def test_out_of_range_lookups_clamp(db):
    assert db.step_hold(datetime(2010, 1, 1, tzinfo=timezone.utc)).ap == db.ap[0]
    assert db.step_hold(datetime(2020, 1, 1, tzinfo=timezone.utc)).ap == db.ap[-1]


def test_loader_is_cached():
    assert load_omni(str(DATA)) is load_omni(str(DATA))


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        OmniDatabase("does/not/exist.dat")
