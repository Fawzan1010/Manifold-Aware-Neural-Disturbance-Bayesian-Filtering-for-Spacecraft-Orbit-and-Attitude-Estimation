import os
import time
import numpy as np
import pytest

from src.utils.profiling import repeat_timing, pin_threads, TimingStats, current_memory_mb


def test_discards_leading_samples():
    calls = []

    def work():
        calls.append(1)
        time.sleep(0.03 if len(calls) == 1 else 0.005)

    st, _ = repeat_timing(work, repeats=3, discard_first=1)
    assert len(calls) == 4          # repeats + discard_first
    assert st.n_kept == 3 and st.n_discarded == 1
    assert st.mean < 0.02           # the slow first sample is excluded
    assert st.samples[0] > 0.02     # but is still recorded


def test_minimum_is_never_above_mean():
    st, _ = repeat_timing(lambda: sum(range(1000)), repeats=5, discard_first=1)
    assert st.minimum <= st.median <= st.maximum
    assert st.minimum <= st.mean


def test_zero_discard_keeps_everything():
    st, _ = repeat_timing(lambda: None, repeats=3, discard_first=0)
    assert st.n_kept == 3 and st.n_discarded == 0 and len(st.samples) == 3


def test_collect_result_returns_last_value():
    counter = {"n": 0}

    def work():
        counter["n"] += 1
        return counter["n"]

    st, res = repeat_timing(work, repeats=2, discard_first=1, collect_result=True)
    assert res == 3


def test_result_is_none_without_collect():
    _, res = repeat_timing(lambda: 42, repeats=2, discard_first=0)
    assert res is None


def test_single_sample_has_zero_std():
    st, _ = repeat_timing(lambda: None, repeats=1, discard_first=0)
    assert st.std == 0.0 and st.n_kept == 1


def test_as_dict_prefix_and_drops_samples():
    st, _ = repeat_timing(lambda: None, repeats=2, discard_first=0)
    d = st.as_dict(prefix="rt_")
    assert "rt_mean" in d and "rt_minimum" in d
    assert not any("samples" in k for k in d)


def test_pin_threads_sets_and_restores():
    before = os.environ.get("OMP_NUM_THREADS")
    with pin_threads(1):
        assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ.get("OMP_NUM_THREADS") == before


def test_pin_threads_restores_on_exception():
    before = os.environ.get("MKL_NUM_THREADS")
    with pytest.raises(RuntimeError):
        with pin_threads(1):
            raise RuntimeError("boom")
    assert os.environ.get("MKL_NUM_THREADS") == before


def test_current_memory_is_positive():
    assert current_memory_mb() > 0.0
