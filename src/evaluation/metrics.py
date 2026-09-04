from __future__ import annotations

"""Accuracy and consistency metrics.

The error state is 24-dimensional and is partitioned into two disjoint,
exhaustive 12-dimensional blocks:

    rotational    : attitude(3) + omega(3) + gyro bias(3) + disturbance torque(3)
    translational : position(3) + velocity(3) + accel bias(3) + disturbance accel(3)

NEES is reported for the full state and for each partition.  Because the
partitions are coupled through the off-diagonal covariance blocks, the two
partition NEES values do not sum to the full-state NEES; each is evaluated
against its own covariance sub-block and its own chi-square degrees of freedom.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.quaternion import quat_geodesic_distance, normalize_quaternion

FULL_DOF = 24
ROT_DOF = 12
TRANS_DOF = 12


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _clean(x: np.ndarray | None) -> np.ndarray:
    if x is None:
        return np.array([], dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def _nees_stats(values: np.ndarray | None, dof: int, prefix: str) -> dict[str, float]:
    """Mean NEES plus chi-square consistency diagnostics for one partition."""
    v = _clean(values)
    if v.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_dof": float(dof),
            f"{prefix}_normalized": np.nan,
            f"{prefix}_within_95": np.nan,
            f"{prefix}_below_95": np.nan,
        }
    lo, hi = stats.chi2.ppf([0.025, 0.975], dof)
    mean = float(np.mean(v))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_median": float(np.median(v)),
        f"{prefix}_dof": float(dof),
        # 1.0 == perfectly consistent; > 1 optimistic, < 1 conservative.
        f"{prefix}_normalized": mean / float(dof),
        f"{prefix}_within_95": float(np.mean((v >= lo) & (v <= hi))),
        f"{prefix}_below_95": float(np.mean(v <= hi)),
    }


def compute_metrics(
    truth: np.ndarray,
    est: np.ndarray,
    nees: np.ndarray | None = None,
    nis: np.ndarray | None = None,
    nees_rot: np.ndarray | None = None,
    nees_trans: np.ndarray | None = None,
    nis_dof: int | None = None,
) -> dict[str, float]:
    qerr = np.array(
        [quat_geodesic_distance(truth[i, :4], est[i, :4]) for i in range(len(truth))]
    )
    out: dict[str, float] = {
        "attitude_geodesic_rmse": float(np.sqrt(np.mean(qerr**2))),
        "quaternion_error_mean": float(np.mean(qerr)),
        "position_rmse": rmse(truth[:, 7:10], est[:, 7:10]),
        "velocity_rmse": rmse(truth[:, 10:13], est[:, 10:13]),
        "angular_rate_rmse": rmse(truth[:, 4:7], est[:, 4:7]),
        "gyro_bias_rmse": rmse(truth[:, 13:16], est[:, 13:16]),
        "accel_bias_rmse": rmse(truth[:, 16:19], est[:, 16:19]),
        "disturbance_torque_rmse": rmse(truth[:, 19:22], est[:, 19:22]),
        "disturbance_accel_rmse": rmse(truth[:, 22:25], est[:, 22:25]),
    }

    out.update(_nees_stats(nees, FULL_DOF, "nees"))
    out.update(_nees_stats(nees_rot, ROT_DOF, "nees_rot"))
    out.update(_nees_stats(nees_trans, TRANS_DOF, "nees_trans"))

    nis_v = _clean(nis)
    if nis_v.size and nis_dof:
        lo, hi = stats.chi2.ppf([0.025, 0.975], nis_dof)
        out["nis_mean"] = float(np.mean(nis_v))
        out["nis_dof"] = float(nis_dof)
        out["nis_normalized"] = float(np.mean(nis_v)) / float(nis_dof)
        out["nis_within_95"] = float(np.mean((nis_v >= lo) & (nis_v <= hi)))
    else:
        out["nis_mean"] = float(np.mean(nis_v)) if nis_v.size else np.nan
        out["nis_dof"] = float(nis_dof) if nis_dof else np.nan
        out["nis_normalized"] = np.nan
        out["nis_within_95"] = np.nan

    return out


def summarize_results(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return df.groupby(group_cols).agg(["mean", "std"])
