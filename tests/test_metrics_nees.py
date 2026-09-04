"""Partitioned NEES: dimensions, calibration and index disjointness."""

import numpy as np
import pytest
from scipy import stats

from src.evaluation.metrics import (
    compute_metrics, FULL_DOF, ROT_DOF, TRANS_DOF, _nees_stats,
)
from src.evaluation.experiments import (
    ROT_ERR_IDX, TRANS_ERR_IDX, _partition_nees, _error_covariance,
)


def _states(n=64, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.standard_normal((n, 25))
    truth[:, :4] /= np.linalg.norm(truth[:, :4], axis=1, keepdims=True)
    est = truth + 0.01 * rng.standard_normal((n, 25))
    est[:, :4] /= np.linalg.norm(est[:, :4], axis=1, keepdims=True)
    return truth, est


def test_partitions_are_disjoint_and_exhaustive():
    assert set(ROT_ERR_IDX).isdisjoint(TRANS_ERR_IDX)
    assert sorted(np.concatenate([ROT_ERR_IDX, TRANS_ERR_IDX])) == list(range(24))


def test_partition_sizes_match_declared_dof():
    assert len(ROT_ERR_IDX) == ROT_DOF == 12
    assert len(TRANS_ERR_IDX) == TRANS_DOF == 12
    assert ROT_DOF + TRANS_DOF == FULL_DOF == 24


def test_nees_stats_calibrated_on_chi_square_draws():
    rng = np.random.default_rng(1)
    st = _nees_stats(rng.chisquare(12, 20000), 12, "x")
    assert st["x_normalized"] == pytest.approx(1.0, abs=0.03)
    assert st["x_within_95"] == pytest.approx(0.95, abs=0.02)


def test_nees_stats_flags_optimistic_filter():
    rng = np.random.default_rng(2)
    st = _nees_stats(rng.chisquare(12, 5000) * 3.0, 12, "x")
    assert st["x_normalized"] > 2.0
    assert st["x_within_95"] < 0.5


def test_partition_nees_matches_hand_computation():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((24, 24))
    P = A @ A.T + 24 * np.eye(24)
    e = rng.standard_normal(24)
    full, rot, trans = _partition_nees(e, P)
    assert full == pytest.approx(e @ np.linalg.solve(P, e))
    ir = np.ix_(ROT_ERR_IDX, ROT_ERR_IDX)
    assert rot == pytest.approx(e[ROT_ERR_IDX] @ np.linalg.solve(P[ir], e[ROT_ERR_IDX]))


def test_partitions_do_not_sum_to_full_when_coupled():
    """Cross-covariance means the partitions are not additive."""
    rng = np.random.default_rng(4)
    A = rng.standard_normal((24, 24))
    P = A @ A.T + 24 * np.eye(24)
    e = rng.standard_normal(24)
    full, rot, trans = _partition_nees(e, P)
    assert not np.isclose(full, rot + trans, rtol=1e-6)


def test_partitions_sum_to_full_when_block_diagonal():
    """With no cross-covariance the decomposition is exact."""
    rng = np.random.default_rng(5)
    P = np.zeros((24, 24))
    for idx in (ROT_ERR_IDX, TRANS_ERR_IDX):
        B = rng.standard_normal((12, 12))
        P[np.ix_(idx, idx)] = B @ B.T + 12 * np.eye(12)
    e = rng.standard_normal(24)
    full, rot, trans = _partition_nees(e, P)
    assert full == pytest.approx(rot + trans, rel=1e-9)


def test_error_covariance_maps_25_to_24():
    rng = np.random.default_rng(6)
    A = rng.standard_normal((25, 25))
    P = A @ A.T
    assert _error_covariance(P).shape == (24, 24)
    P24 = np.eye(24)
    assert np.allclose(_error_covariance(P24), P24)


def test_compute_metrics_emits_all_three_partitions():
    truth, est = _states()
    rng = np.random.default_rng(7)
    m = compute_metrics(
        truth, est,
        nees=rng.chisquare(24, 64), nis=rng.chisquare(20, 64),
        nees_rot=rng.chisquare(12, 64), nees_trans=rng.chisquare(12, 64),
        nis_dof=20,
    )
    for p, dof in (("nees", 24), ("nees_rot", 12), ("nees_trans", 12)):
        assert m[f"{p}_dof"] == dof
        assert np.isfinite(m[f"{p}_mean"]) and np.isfinite(m[f"{p}_normalized"])


def test_compute_metrics_handles_missing_nees():
    """Learning-only baselines carry no covariance."""
    truth, est = _states()
    m = compute_metrics(truth, est)
    assert np.isnan(m["nees_mean"]) and np.isnan(m["nees_rot_mean"])
    assert np.isfinite(m["position_rmse"])


def test_nan_values_are_ignored_not_propagated():
    v = np.array([12.0, np.nan, 12.0, np.inf])
    assert _nees_stats(v, 12, "x")["x_mean"] == pytest.approx(12.0)
