"""Ablation logic that can be checked without torch."""
import importlib.util
import numpy as np
import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="ablation imports torch")


def test_relevance_map_partitions_methods():
    from src.evaluation.ablation import (
        relevant_methods, PINN_METHODS, FUSION_METHODS, INVARIANT_METHODS,
        TRAINING_AXES, DEFAULT_AXES,
    )
    from src.evaluation.experiments import METHODS

    for axis in TRAINING_AXES:
        assert relevant_methods(axis) == PINN_METHODS
    assert relevant_methods("pinn_r_scale") == FUSION_METHODS

    # Invariant methods must not appear in any axis's relevant set.
    touched = set()
    for axis in DEFAULT_AXES:
        touched |= set(relevant_methods(axis))
    assert not (set(INVARIANT_METHODS) & touched)
    assert set(INVARIANT_METHODS) | touched == set(METHODS)


def test_classical_filters_are_invariant():
    from src.evaluation.ablation import INVARIANT_METHODS
    for m in ["EKF", "Adaptive-EKF", "UKF", "MEKF"]:
        assert m in INVARIANT_METHODS


def test_pinn_only_excluded_from_fusion_axis():
    """PINN-only has no covariance, so pinn_r_scale cannot affect it."""
    from src.evaluation.ablation import relevant_methods
    assert "PINN-only" not in relevant_methods("pinn_r_scale")


def test_metric_columns_include_all_channels_and_nees_partitions():
    from src.evaluation.ablation import METRIC_COLUMNS
    for m in ["attitude_geodesic_rmse", "position_rmse", "velocity_rmse",
              "angular_rate_rmse", "gyro_bias_rmse", "accel_bias_rmse",
              "disturbance_torque_rmse", "disturbance_accel_rmse",
              "nees_normalized", "nees_rot_normalized", "nees_trans_normalized"]:
        assert m in METRIC_COLUMNS


def test_relevance_map_reduces_work():
    from src.evaluation.ablation import relevant_methods, DEFAULT_AXES
    from src.evaluation.experiments import METHODS
    naive = sum(len(v) * len(METHODS) for v in DEFAULT_AXES.values())
    actual = sum(len(v) * len(relevant_methods(a)) for a, v in DEFAULT_AXES.items())
    assert actual < naive / 2


def test_cost_estimate_scales_linearly():
    from src.evaluation.ablation import _estimate_cost
    a = _estimate_cost(10, 3, 20, 240, 5.0)
    b = _estimate_cost(20, 3, 20, 240, 5.0)
    assert b == pytest.approx(2 * a)
    assert a > 0
