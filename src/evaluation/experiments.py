from __future__ import annotations

"""Benchmark experiments.

Accuracy and runtime are measured in two separate passes.  The accuracy pass
runs every method once over the full test set and collects no timings.  The
runtime pass re-runs a small trajectory subset several times with the method
loop nested *inside* the repeat loop, discards the leading repeat, and
summarises the remainder.  Interleaving decorrelates drift in machine load
from method identity; timing inline with the accuracy pass does not, because
methods are executed in sequence and any load excursion is attributed to
whichever method was running at the time.

State layout (25):  0:4 quaternion, 4:7 omega, 7:10 position, 10:13 velocity,
                    13:16 gyro bias, 16:19 accel bias, 19:22 tau_d, 22:25 a_d
Error layout (24):  0:3 attitude, 3:6 omega, 6:9 position, 9:12 velocity,
                    12:15 gyro bias, 15:18 accel bias, 18:21 tau_d, 21:24 a_d
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.dynamics.simulator import Trajectory
from src.dynamics.spacecraft import SpacecraftParams, rk4_step, Environment, ControlInput
from src.evaluation.metrics import compute_metrics, FULL_DOF, ROT_DOF, TRANS_DOF
from src.evaluation.stats import (
    confidence_interval,
    paired_ttest,
    wilcoxon_test,
    improvement,
    paired_bootstrap_ci,
    cohens_d_paired,
)
from src.filters.ekf import EKF
from src.filters.adaptive_ekf import AdaptiveEKF
from src.filters.ukf import UKF
from src.filters.mekf import MEKF
from src.sensors.measurement_models import measurement_noise_cov, measurement_dim
from src.utils.profiling import repeat_timing, pin_threads, current_memory_mb
from src.utils.progress import progress, progress_bar, section

# Per-timestep bars are only drawn for arcs long enough to justify them; see
# PROGRESS_MIN_STEPS in src/dynamics/simulator.py for the same reasoning.
PROGRESS_MIN_STEPS = 1000
from src.utils.reproducibility import ensure_dir
from src.utils.quaternion import (
    normalize_quaternion,
    quat_conjugate,
    quat_multiply,
    rotvec_from_quat,
)

METHODS = [
    "EKF",
    "Adaptive-EKF",
    "UKF",
    "MEKF",
    "PINN-only",
    "Transformer-only",
    "PINN+EKF",
    "PINN+UKF",
    "PINN+MEKF",
    "Transformer+MEKF",
    "PINN+UKF+MEKF",
]

# Convex-blend coupling, used only by the learning-only baselines which have
# no covariance to fuse against.  Filter-coupled methods use the
# pseudo-measurement update in _fuse_pinn_prior instead.
DEFAULT_COUPLING = {
    "PINN+EKF": 0.65,
    "PINN+UKF": 0.75,
    "PINN+MEKF": 0.90,
    "Transformer+MEKF": 0.85,
    "PINN+UKF+MEKF": 0.98,
    "PINN-only": 1.00,
    "Transformer-only": 1.00,
}

ROT_IDX = np.r_[4:7, 13:16, 19:22]
TRANS_IDX = np.r_[7:10, 10:13, 16:19, 22:25]
ROT_ERR_IDX = np.r_[0:6, 12:15, 18:21]
TRANS_ERR_IDX = np.r_[6:12, 15:18, 21:24]

KEY_METRICS = [
    "attitude_geodesic_rmse",
    "position_rmse",
    "velocity_rmse",
    "angular_rate_rmse",
]

# Every accuracy channel is tested for significance.  Restricting the tests to
# a subset allows a ranking inversion on an untested channel to go unnoticed.
TESTED_METRICS = KEY_METRICS


@dataclass
class RunResult:
    """Everything one estimator run produces.

    A dataclass rather than a tuple so that adding a diagnostic does not
    change the arity of every call site.
    """

    est: np.ndarray
    metrics: dict[str, Any]
    nees: np.ndarray
    nees_rot: np.ndarray
    nees_trans: np.ndarray
    nis: np.ndarray
    innovations: np.ndarray
    timing: dict[str, float] = field(default_factory=dict)
    diverged: bool = False
    divergence_step: int | None = None


def _is_finite_state(x: np.ndarray) -> bool:
    """Reject non-finite values and quaternions that have left the unit sphere.

    A quaternion norm far from 1 is the earliest sign of divergence: the
    small-angle assumptions behind the sigma-point and EKF linearizations stop
    holding well before NaN actually appears, and by the time an overflow
    warning fires the state is already unrecoverable garbage.
    """
    if not np.all(np.isfinite(x)):
        return False
    qnorm = float(np.linalg.norm(x[:4]))
    return 0.1 < qnorm < 10.0


def _load_split(path: Path) -> list[Trajectory]:
    data = np.load(path, allow_pickle=True)
    return list(data["trajectories"])


def _init_state(traj: Trajectory) -> np.ndarray:
    x0 = traj.states[0].copy()
    x0[:4] = normalize_quaternion(x0[:4])
    return x0


def _state_error_vec(truth: np.ndarray, est: np.ndarray) -> np.ndarray:
    dq = quat_multiply(truth[:4], quat_conjugate(est[:4]))
    att = rotvec_from_quat(dq)
    return np.hstack([att, truth[4:] - est[4:]])


def _error_covariance(P: np.ndarray) -> np.ndarray:
    """Reduce a filter covariance to the 24-dim error-state space.

    For 25-state quaternion filters the attitude error rotation vector
    satisfies delta_theta ~= 2 * delta_q_v, so G maps the vector part of the
    quaternion covariance with a factor of 2.  MEKF covariance is already
    expressed in the 24-dim error state.
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    if P.shape[0] == 24:
        return P
    if P.shape[0] == 25:
        G = np.zeros((24, 25))
        G[0:3, 1:4] = 2.0 * np.eye(3)
        G[3:, 4:] = np.eye(21)
        return G @ P @ G.T
    return P


def _nees_from(e: np.ndarray, P: np.ndarray) -> float:
    """Normalised estimation error squared, NaN if the solve is singular."""
    try:
        return float(e.T @ np.linalg.solve(P, e))
    except np.linalg.LinAlgError:
        return np.nan


def _partition_nees(e: np.ndarray, P24: np.ndarray) -> tuple[float, float, float]:
    """Full-state, rotational and translational NEES.

    Each partition is evaluated against its own covariance sub-block, so the
    two partition values do not sum to the full-state value; the difference is
    carried by the rotational/translational cross-covariance.
    """
    if P24.shape[0] != e.size:
        return np.nan, np.nan, np.nan
    full = _nees_from(e, P24)
    rot = _nees_from(e[ROT_ERR_IDX], P24[np.ix_(ROT_ERR_IDX, ROT_ERR_IDX)])
    trans = _nees_from(e[TRANS_ERR_IDX], P24[np.ix_(TRANS_ERR_IDX, TRANS_ERR_IDX)])
    return full, rot, trans


def _process_noise(cfg: dict) -> np.ndarray:
    q = cfg["process_noise"]
    diag = np.array(
        [
            1e-8, 1e-8, 1e-8, 1e-8,
            5e-6, 5e-6, 5e-6,
            1e-4, 1e-4, 1e-4,
            1e-5, 1e-5, 1e-5,
            q["bg"], q["bg"], q["bg"],
            q["ba"], q["ba"], q["ba"],
            q["torque"], q["torque"], q["torque"],
            q["accel"], q["accel"], q["accel"],
        ],
        dtype=float,
    )
    return np.diag(diag)


def _coupling(project, method: str) -> float:
    cfg = project.config.get("fusion", {}).get("coupling", {})
    return float(cfg.get(method, DEFAULT_COUPLING.get(method, 0.0)))


def _build_feature(x: np.ndarray, traj: Trajectory, k: int, hist: np.ndarray) -> np.ndarray:
    return np.hstack(
        [
            x,
            traj.controls[k],
            traj.env["sun_vec_i"][k],
            traj.env["earth_vec_i"][k],
            traj.env["magnetic_field_i"][k],
            [traj.env["weather_index"][k]],
            hist.reshape(-1),
        ]
    )


def _predict_residual(method: str, feat: np.ndarray, hist: np.ndarray, pinn, trans):
    """Return (disturbance prediction[6], predictive variance[6] or None).

    The PINN's heteroscedastic logvar head supplies the variance, which lets
    the prior be fused as a pseudo-measurement with calibrated uncertainty.
    """
    import torch

    if method in {"PINN-only", "PINN+EKF", "PINN+UKF", "PINN+MEKF", "PINN+UKF+MEKF"} and pinn is not None:
        inp = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred, logvar = pinn.model(inp)
        pred = pred.squeeze(0).cpu().numpy()
        var = np.exp(logvar.squeeze(0).cpu().numpy())
        if pred.shape[0] < 6:
            pred = np.pad(pred, (0, 6 - pred.shape[0]))
            var = np.pad(var, (0, 6 - var.shape[0]), constant_values=1.0)
        return pred[:6], var[:6]

    if method in {"Transformer-only", "Transformer+MEKF"} and trans is not None:
        seq = torch.tensor(hist.reshape(1, hist.shape[0], -1), dtype=torch.float32)
        with torch.no_grad():
            pred, _state_hat = trans.model(seq)
        pred = pred.squeeze(0).cpu().numpy()
        if pred.shape[0] < 6:
            pred = np.pad(pred, (0, 6 - pred.shape[0]))
        return pred[:6], np.full(6, 1e-2)  # no uncertainty head on the transformer

    return np.zeros(6, dtype=float), None


def _make_filter(filter_kind: str, x0, P0, params, Q, R, include_rd: bool, project):
    if filter_kind == "UKF":
        return UKF(
            x0, P0, params, Q, R,
            include_range_doppler=include_rd,
            alpha=project.config["filter"]["ukf_alpha"],
            beta=project.config["filter"]["ukf_beta"],
            kappa=project.config["filter"]["ukf_kappa"],
        )
    if filter_kind == "EKF":
        return EKF(x0, P0, params, Q, R, include_range_doppler=include_rd)
    if filter_kind == "Adaptive-EKF":
        return AdaptiveEKF(x0, P0, params, Q, R, include_range_doppler=include_rd)
    if filter_kind == "MEKF":
        return MEKF(x0, P0, params, Q, R, include_range_doppler=include_rd)
    raise ValueError(filter_kind)


def _inject_residual(x: np.ndarray, residual: np.ndarray, gamma: float) -> np.ndarray:
    """Convex blend of a disturbance prediction into the disturbance sub-states."""
    x = x.copy()
    res = np.asarray(residual, dtype=float)
    if res.shape[0] < 6:
        res = np.pad(res, (0, 6 - res.shape[0]))
    if gamma > 0.0:
        x[19:22] = (1.0 - gamma) * x[19:22] + gamma * res[:3]
        x[22:25] = (1.0 - gamma) * x[22:25] + gamma * res[3:6]
    return x


def _fuse_pinn_prior(filt, mean: np.ndarray, var: np.ndarray | None, r_scale: float) -> None:
    """Fuse the disturbance prior as a pseudo-measurement.

    A linear Kalman update on the disturbance sub-states [tau_d, a_d] with
    R_pinn = diag(var) * r_scale.  Cross-covariances propagate the correction
    to correlated states and the covariance contraction lets the filter
    benefit at the consistency level, neither of which a convex blend of the
    means would achieve.
    """
    if var is None:
        return
    n = filt.P.shape[0]
    if n == 24:        # MEKF error-state covariance
        idx = np.arange(18, 24)
    elif n == 25:      # quaternion-state covariance (EKF/UKF)
        idx = np.arange(19, 25)
    else:
        return
    Rp = np.diag(np.maximum(np.asarray(var, dtype=float), 1e-10) * max(r_scale, 1e-6))
    H = np.zeros((6, n))
    H[np.arange(6), idx] = 1.0
    S = H @ filt.P @ H.T + Rp
    try:
        K = filt.P @ H.T @ np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return
    z = np.asarray(mean, dtype=float) - filt.x[19:25]
    dx = K @ z
    if n == 24:
        filt.x = filt.inject(filt.x, dx)
    else:
        filt.x = filt.x + dx
        filt.x[:4] = normalize_quaternion(filt.x[:4])
    filt.P = (np.eye(n) - K @ H) @ filt.P
    filt.P = 0.5 * (filt.P + filt.P.T)


def _env_at(traj: Trajectory, k: int, atmosphere=None) -> Environment:
    """Rebuild the Environment for step k of a stored trajectory."""
    return Environment(
        sun_vec_i=traj.env["sun_vec_i"][k],
        earth_vec_i=traj.env["earth_vec_i"][k],
        magnetic_field_i=traj.env["magnetic_field_i"][k],
        rho_atm=float(traj.env["rho_atm"][k]) if "rho_atm" in traj.env else 0.0,
        weather_index=float(traj.env["weather_index"][k]),
        att_ref1_i=traj.env["att_ref1_i"][k] if "att_ref1_i" in traj.env else None,
        att_ref2_i=traj.env["att_ref2_i"][k] if "att_ref2_i" in traj.env else None,
    )


def _ctrl_at(traj: Trajectory, k: int) -> ControlInput:
    return ControlInput(
        torque_cmd=traj.controls[k, :3],
        accel_cmd=traj.controls[k, 3:],
    )


def _hist_window(traj: Trajectory, k: int, window: int) -> np.ndarray:
    hist = np.nan_to_num(traj.measurements[max(0, k - window):k], nan=0.0)
    if len(hist) < window:
        pad = np.zeros((window - len(hist), traj.measurements.shape[1]))
        hist = np.vstack([pad, hist])
    return hist


def _finalize_metrics(truth, est, nees, nees_rot, nees_trans, nis, nis_dof):
    return compute_metrics(
        truth, est,
        nees=nees, nis=nis,
        nees_rot=nees_rot, nees_trans=nees_trans,
        nis_dof=nis_dof,
    )


def _run_filter(
    method: str,
    traj: Trajectory,
    project,
    pinn=None,
    trans=None,
    collect_timing: bool = False,
    show_progress: bool = False,
) -> RunResult:
    params = project.spacecraft_params()
    include_rd = bool(project.config["synthetic"]["include_range_doppler"])
    st = bool(project.config.get("simulation", {}).get("star_tracker", False))
    R = measurement_noise_cov(project.config["measurement_noise"], include_rd, star_tracker=st) * float(
        project.config["filter"]["r_scale"]
    )
    Q = _process_noise(project.config) * float(project.config["filter"]["q_scale"])

    x0 = _init_state(traj)
    P0 = np.diag([1e-3] * 4 + [1e-2] * 21)
    if method in {"MEKF", "PINN+MEKF", "Transformer+MEKF"}:
        P0 = np.diag([1e-2] * 25)

    kind = {
        "EKF": "EKF",
        "Adaptive-EKF": "Adaptive-EKF",
        "UKF": "UKF",
        "MEKF": "MEKF",
        "PINN+EKF": "EKF",
        "PINN+UKF": "UKF",
        "PINN+MEKF": "MEKF",
        "Transformer+MEKF": "MEKF",
    }.get(method)
    if kind is None:
        raise ValueError(method)

    filt = _make_filter(kind, x0, P0, params, Q, R, include_rd, project)
    use_pinn = method in {"PINN+EKF", "PINN+UKF", "PINN+MEKF", "Transformer+MEKF"}
    r_scale_pinn = float(project.config.get("fusion", {}).get("pinn_r_scale", 1.0))
    window = int(project.config["training"]["window"])
    dt = float(project.config["synthetic"]["dt"])

    est = np.zeros_like(traj.states)
    n = len(traj.time)
    nees = np.full(n, np.nan)
    nees_rot = np.full(n, np.nan)
    nees_trans = np.full(n, np.nan)
    nis = np.full(n, np.nan)
    nis_dofs: list[int] = []
    innovations: list[np.ndarray] = []
    pinn_t = 0.0
    filt_t = 0.0

    diverged = False
    divergence_step = None

    for k in progress(range(n), desc=f"      {method} timesteps", leave=False,
                      unit="step", disable=not (show_progress and n >= PROGRESS_MIN_STEPS)):
        env = _env_at(traj, k)
        ctrl = _ctrl_at(traj, k)
        hist = _hist_window(traj, k, window)

        try:
            with np.errstate(over="raise", invalid="raise"):
                if collect_timing:
                    t0 = time.perf_counter()
                if use_pinn:
                    feat = _build_feature(filt.x, traj, k, hist)
                    residual, res_var = _predict_residual(method, feat, hist, pinn, trans)
                    _fuse_pinn_prior(filt, residual, res_var, r_scale_pinn)
                if collect_timing:
                    pinn_t += time.perf_counter() - t0
                    t0 = time.perf_counter()

                step = filt.step(
                    traj.time[k], dt, env, ctrl,
                    traj.measurements[k], traj.measurement_mask[k],
                )
                if collect_timing:
                    filt_t += time.perf_counter() - t0

                if not _is_finite_state(filt.x):
                    raise FloatingPointError("filter state left the physical domain")
        except (FloatingPointError, np.linalg.LinAlgError):
            # The filter has diverged: further steps only propagate garbage
            # through an already-broken covariance.  Stop here, keep what was
            # estimated up to this point, and record where it happened rather
            # than silently continuing or crashing the whole experiment.
            diverged = True
            divergence_step = k
            break

        est[k] = filt.x

        if getattr(step, "S", None) is not None and np.size(step.S) > 0:
            innov = np.asarray(step.innovation).reshape(-1, 1)
            innovations.append(innov.squeeze().copy())
            nis_dofs.append(int(innov.size))
            nis[k] = _nees_from(innov.ravel(), np.asarray(step.S, dtype=float))

        e = _state_error_vec(traj.states[k], est[k])
        nees[k], nees_rot[k], nees_trans[k] = _partition_nees(e, _error_covariance(filt.P))

    valid = divergence_step if diverged else n
    dof = int(round(float(np.mean(nis_dofs)))) if nis_dofs else measurement_dim(include_rd)
    metrics = _finalize_metrics(traj.states[:valid], est[:valid], nees[:valid],
                                nees_rot[:valid], nees_trans[:valid], nis[:valid], dof)
    metrics["diverged"] = diverged
    metrics["divergence_step"] = divergence_step if diverged else -1
    metrics["valid_steps"] = valid
    timing = {"pinn_s": pinn_t, "filter_s": filt_t, "steps": n} if collect_timing else {}

    return RunResult(est, metrics, nees, nees_rot, nees_trans, nis,
                     np.asarray(innovations, dtype=object), timing,
                     diverged=diverged, divergence_step=divergence_step)


def _run_learned(
    method: str,
    traj: Trajectory,
    project,
    pinn=None,
    trans=None,
    collect_timing: bool = False,
    show_progress: bool = False,
) -> RunResult:
    params = project.spacecraft_params()
    x = traj.states[0].copy()
    x[:4] = normalize_quaternion(x[:4])

    est = np.zeros_like(traj.states)
    n = len(traj.time)
    window = int(project.config["training"]["window"])
    dt = float(project.config["synthetic"]["dt"])
    gamma = _coupling(project, method)
    pinn_t = 0.0

    diverged = False
    divergence_step = None

    for k in progress(range(n), desc=f"      {method} timesteps", leave=False,
                      unit="step", disable=not (show_progress and n >= PROGRESS_MIN_STEPS)):
        env = _env_at(traj, k)
        ctrl = _ctrl_at(traj, k)
        hist = _hist_window(traj, k, window)

        try:
            with np.errstate(over="raise", invalid="raise"):
                if collect_timing:
                    t0 = time.perf_counter()
                feat = _build_feature(x, traj, k, hist)
                residual, _res_var = _predict_residual(method, feat, hist, pinn, trans)
                x = _inject_residual(x, residual, gamma)
                if collect_timing:
                    pinn_t += time.perf_counter() - t0

                x = rk4_step(x, traj.time[k], dt, ctrl, env, params)
                x[:4] = normalize_quaternion(x[:4])

                if not _is_finite_state(x):
                    raise FloatingPointError("state left the physical domain")
        except (FloatingPointError, np.linalg.LinAlgError):
            diverged = True
            divergence_step = k
            break

        est[k] = x

    valid = divergence_step if diverged else n
    empty = np.array([])
    metrics = _finalize_metrics(traj.states[:valid], est[:valid], empty, empty, empty, empty, None)
    metrics["diverged"] = diverged
    metrics["divergence_step"] = divergence_step if diverged else -1
    metrics["valid_steps"] = valid
    timing = {"pinn_s": pinn_t, "filter_s": 0.0, "steps": n} if collect_timing else {}

    return RunResult(est, metrics, empty, empty, empty, empty,
                     np.array([]), timing, diverged=diverged, divergence_step=divergence_step)


def _run_hybrid_ukf_mekf(
    traj: Trajectory,
    project,
    pinn=None,
    collect_timing: bool = False,
    show_progress: bool = False,
) -> RunResult:
    """PINN prior with a UKF on the translational partition and an MEKF on the
    rotational partition, cross-feeding each step and fused at the output."""
    params = project.spacecraft_params()
    include_rd = bool(project.config["synthetic"]["include_range_doppler"])
    st = bool(project.config.get("simulation", {}).get("star_tracker", False))
    R = measurement_noise_cov(project.config["measurement_noise"], include_rd, star_tracker=st) * float(
        project.config["filter"]["r_scale"]
    )
    Q = _process_noise(project.config) * float(project.config["filter"]["q_scale"])
    dt = float(project.config["synthetic"]["dt"])

    x0 = _init_state(traj)
    ukf = UKF(
        x0, np.diag([1e-2] * 25), params, Q, R,
        include_range_doppler=include_rd,
        alpha=project.config["filter"]["ukf_alpha"],
        beta=project.config["filter"]["ukf_beta"],
        kappa=project.config["filter"]["ukf_kappa"],
    )
    mekf = MEKF(x0, np.diag([1e-2] * 25), params, Q, R, include_range_doppler=include_rd)

    r_scale_pinn = float(project.config.get("fusion", {}).get("pinn_r_scale", 1.0))
    window = int(project.config["training"]["window"])

    est = np.zeros_like(traj.states)
    n = len(traj.time)
    nees = np.full(n, np.nan)
    nees_rot = np.full(n, np.nan)
    nees_trans = np.full(n, np.nan)
    nis = np.full(n, np.nan)
    nis_dofs: list[int] = []
    innovations: list[np.ndarray] = []
    pinn_t = 0.0
    filt_t = 0.0

    diverged = False
    divergence_step = None

    for k in progress(range(n), desc="      PINN+UKF+MEKF timesteps", leave=False,
                      unit="step", disable=not (show_progress and n >= PROGRESS_MIN_STEPS)):
        env = _env_at(traj, k)
        ctrl = _ctrl_at(traj, k)
        hist = _hist_window(traj, k, window)

        try:
            with np.errstate(over="raise", invalid="raise"):
                if collect_timing:
                    t0 = time.perf_counter()
                feat = _build_feature(est[k - 1] if k > 0 else ukf.x, traj, k, hist)
                residual, res_var = _predict_residual("PINN+UKF+MEKF", feat, hist, pinn, None)
                _fuse_pinn_prior(ukf, residual, res_var, r_scale_pinn)
                _fuse_pinn_prior(mekf, residual, res_var, r_scale_pinn)
                if collect_timing:
                    pinn_t += time.perf_counter() - t0

                # Cross-feed so both filters propagate a consistent full state.
                ukf.x[:4] = normalize_quaternion(mekf.x[:4])
                ukf.x[ROT_IDX] = mekf.x[ROT_IDX]
                mekf.x[TRANS_IDX] = ukf.x[TRANS_IDX]

                if collect_timing:
                    t0 = time.perf_counter()
                step_u = ukf.step(traj.time[k], dt, env, ctrl,
                                  traj.measurements[k], traj.measurement_mask[k])
                step_m = mekf.step(traj.time[k], dt, env, ctrl,
                                   traj.measurements[k], traj.measurement_mask[k])
                if collect_timing:
                    filt_t += time.perf_counter() - t0

                # Fuse disjoint partitions: rotational from the MEKF, translational
                # from the UKF.
                fused = ukf.x.copy()
                fused[:4] = normalize_quaternion(mekf.x[:4])
                fused[ROT_IDX] = mekf.x[ROT_IDX]

                if not (_is_finite_state(fused) and _is_finite_state(ukf.x)
                       and _is_finite_state(mekf.x)):
                    raise FloatingPointError("hybrid state left the physical domain")
        except (FloatingPointError, np.linalg.LinAlgError):
            # Either sub-filter diverging invalidates the fused state; stop
            # here rather than let the corrupted covariance propagate.
            diverged = True
            divergence_step = k
            break

        est[k] = fused

        if getattr(step_u, "S", None) is not None and np.size(step_u.S) > 0:
            innov = np.asarray(step_u.innovation).reshape(-1, 1)
            innovations.append(innov.squeeze().copy())
            nis_dofs.append(int(innov.size))
            nis[k] = _nees_from(innov.ravel(), np.asarray(step_u.S, dtype=float))

        # Composite covariance: rotational block from the MEKF, everything
        # else from the UKF.  The partitioned NEES then evaluates each block
        # against the filter that actually produced it.
        e = _state_error_vec(traj.states[k], est[k])
        P_u = _error_covariance(ukf.P)
        P_m = _error_covariance(mekf.P)
        P24 = P_u.copy()
        P24[np.ix_(ROT_ERR_IDX, ROT_ERR_IDX)] = P_m[np.ix_(ROT_ERR_IDX, ROT_ERR_IDX)]
        nees[k], nees_rot[k], nees_trans[k] = _partition_nees(e, P24)

    valid = divergence_step if diverged else n
    dof = int(round(float(np.mean(nis_dofs)))) if nis_dofs else measurement_dim(include_rd)
    metrics = _finalize_metrics(traj.states[:valid], est[:valid], nees[:valid],
                                nees_rot[:valid], nees_trans[:valid], nis[:valid], dof)
    metrics["diverged"] = diverged
    metrics["divergence_step"] = divergence_step if diverged else -1
    metrics["valid_steps"] = valid
    timing = {"pinn_s": pinn_t, "filter_s": filt_t, "steps": n} if collect_timing else {}

    return RunResult(est, metrics, nees, nees_rot, nees_trans, nis,
                     np.asarray(innovations, dtype=object), timing,
                     diverged=diverged, divergence_step=divergence_step)


def dispatch(method: str, traj: Trajectory, project, pinn=None, trans=None,
             collect_timing: bool = False, show_progress: bool = False) -> RunResult:
    """Single entry point mapping a method name to its runner.

    ``show_progress`` draws a per-timestep bar inside the runner.  It defaults
    to off: on the 240-step benchmark horizon the loop is sub-second, and the
    timed passes must not carry any reporting overhead at all.  The long storm
    arcs switch it on.
    """
    if method in {"EKF", "Adaptive-EKF", "UKF", "MEKF"}:
        return _run_filter(method, traj, project, collect_timing=collect_timing,
                           show_progress=show_progress)
    if method == "PINN-only":
        return _run_learned(method, traj, project, pinn=pinn, collect_timing=collect_timing,
                            show_progress=show_progress)
    if method == "Transformer-only":
        return _run_learned(method, traj, project, trans=trans, collect_timing=collect_timing,
                            show_progress=show_progress)
    if method in {"PINN+EKF", "PINN+UKF", "PINN+MEKF"}:
        return _run_filter(method, traj, project, pinn=pinn, collect_timing=collect_timing,
                           show_progress=show_progress)
    if method == "Transformer+MEKF":
        return _run_filter(method, traj, project, trans=trans, collect_timing=collect_timing,
                           show_progress=show_progress)
    if method == "PINN+UKF+MEKF":
        return _run_hybrid_ukf_mekf(traj, project, pinn=pinn, collect_timing=collect_timing,
                                    show_progress=show_progress)
    raise ValueError(method)


# --------------------------------------------------------------------------
# Runtime benchmark
# --------------------------------------------------------------------------

def run_runtime_benchmark(project, pinn=None, trans=None) -> pd.DataFrame:
    """Measure per-step runtime with repeated, interleaved execution.

    The method loop is nested inside the repeat loop so that a load excursion
    lasting one repeat perturbs every method equally instead of being charged
    to whichever method was running at the time.  The leading repeat is
    discarded as warm-up.  Both the mean and the minimum are reported: under
    contention the minimum is the better estimate of intrinsic cost, since
    interference can only add time.
    """
    bcfg = project.config.get("benchmark", {})
    repeats = int(bcfg.get("repeats", 5))
    discard_first = int(bcfg.get("discard_first", 1))
    n_traj = int(bcfg.get("n_trajectories", 3))
    n_threads = int(bcfg.get("pin_threads", 1))

    test = _load_split(project.data_dir / "test.npz")[:n_traj]
    if not test:
        raise ValueError("No test trajectories available for the runtime benchmark.")

    total_reps = repeats + discard_first
    # samples[method][rep] = seconds per estimation step
    samples: dict[str, list[float]] = {m: [] for m in METHODS}
    breakdown: dict[str, list[tuple[float, float]]] = {m: [] for m in METHODS}

    with pin_threads(n_threads):
        for rep in progress(range(total_reps), desc="  Runtime repeats",
                            unit="repeat"):
            for method in METHODS:
                t0 = time.perf_counter()
                pinn_s = 0.0
                filt_s = 0.0
                steps = 0
                for traj in test:
                    res = dispatch(method, traj, project, pinn=pinn, trans=trans,
                                   collect_timing=True)
                    pinn_s += res.timing.get("pinn_s", 0.0)
                    filt_s += res.timing.get("filter_s", 0.0)
                    steps += int(res.timing.get("steps", len(traj.time)))
                elapsed = time.perf_counter() - t0
                samples[method].append(elapsed / max(steps, 1))
                breakdown[method].append((pinn_s / max(steps, 1), filt_s / max(steps, 1)))

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        kept = np.asarray(samples[method][discard_first:], dtype=float)
        kept_bd = np.asarray(breakdown[method][discard_first:], dtype=float)
        rows.append(
            {
                "method": method,
                "total_ms": 1e3 * float(np.mean(kept)),
                "total_ms_std": 1e3 * (float(np.std(kept, ddof=1)) if kept.size > 1 else 0.0),
                "total_ms_median": 1e3 * float(np.median(kept)),
                "total_ms_min": 1e3 * float(np.min(kept)),
                "total_ms_max": 1e3 * float(np.max(kept)),
                "pinn_ms": 1e3 * float(np.mean(kept_bd[:, 0])),
                "filter_ms": 1e3 * float(np.mean(kept_bd[:, 1])),
                "n_repeats_kept": int(kept.size),
                "n_repeats_discarded": discard_first,
                "n_trajectories": len(test),
                "blas_threads": n_threads,
                "memory_usage_mb": current_memory_mb(),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(project.table_dir / "runtime_breakdown.csv", index=False)
    df.to_latex(project.table_dir / "runtime_breakdown.tex", index=False, float_format="%.3f")

    # Per-repeat samples retained so that residual drift stays auditable.
    raw = pd.DataFrame(
        [
            {"method": m, "repeat": i, "discarded": i < discard_first,
             "ms_per_step": 1e3 * v}
            for m, vals in samples.items()
            for i, v in enumerate(vals)
        ]
    )
    raw.to_csv(project.table_dir / "runtime_repeats.csv", index=False)
    return df


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _ranking_table(all_metrics: dict[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    """Rank methods on one metric and test each against the rank-1 method.

    Trials are paired by index, so the paired difference removes the shared
    trajectory effect.  A confidence interval on the pooled metric would be
    dominated by the scenario mix and could not resolve method differences.
    """
    means = {m: float(df[metric].mean()) for m, df in all_metrics.items()
             if metric in df.columns}
    if not means:
        return pd.DataFrame()
    order = sorted(means, key=means.get)
    best = order[0]
    best_vals = all_metrics[best][metric].values

    rows = []
    for rank, method in enumerate(order, start=1):
        vals = all_metrics[method][metric].values
        lo, hi = paired_bootstrap_ci(vals, best_vals)
        rows.append(
            {
                "metric": metric,
                "rank": rank,
                "method": method,
                "mean": means[method],
                "std": float(np.nanstd(vals, ddof=1)),
                "median": float(np.nanmedian(vals)),
                "best_method": best,
                "delta_vs_best": means[method] - means[best],
                "delta_ci95_lo": lo,
                "delta_ci95_hi": hi,
                "p_ttest_vs_best": paired_ttest(vals, best_vals) if method != best else np.nan,
                "p_wilcoxon_vs_best": wilcoxon_test(vals, best_vals) if method != best else np.nan,
                "cohens_d_vs_best": cohens_d_paired(vals, best_vals) if method != best else np.nan,
                # True when the CI on the paired difference excludes zero.
                "significantly_worse_than_best": bool(method != best and lo > 0.0),
            }
        )
    return pd.DataFrame(rows)


def _per_scenario_summary(per_trial: pd.DataFrame) -> pd.DataFrame:
    """Disaggregate by scenario.

    Pooled statistics mix six scenarios whose difficulty differs by more than
    the differences between methods, so the pooled standard deviation is
    mostly between-scenario variance and is not interpretable on its own.
    """
    if "scenario" not in per_trial.columns:
        return pd.DataFrame()
    cols = [c for c in KEY_METRICS if c in per_trial.columns]
    cols += [c for c in ("nees_mean", "nees_rot_mean", "nees_trans_mean")
             if c in per_trial.columns]
    g = per_trial.groupby(["scenario", "method"])[cols]
    out = g.agg(["mean", "std", "count"])
    out.columns = ["_".join(c) for c in out.columns]
    return out.reset_index()


def run_all_experiments(project):
    # Imported here rather than at module scope so that the classical filters
    # remain importable and testable without torch installed.
    from src.pinn.train import load_pinn
    from src.models.train import load_transformer

    test = _load_split(project.data_dir / "test.npz")

    pinn = load_pinn(project.model_dir / "pinn.pt", project.config["device"])
    trans = load_transformer(project.model_dir / "transformer.pt", project.config["device"])

    pred_dir = ensure_dir(project.output_dir / "predictions")
    # Stale prediction files would otherwise be picked up by the theory pass,
    # which globs this directory and would mix runs with different schemas.
    for stale in pred_dir.glob("*.npz"):
        stale.unlink()

    all_metrics: dict[str, pd.DataFrame] = {}

    section(f"Accuracy pass: {len(METHODS)} methods x {len(test)} trajectories")
    method_bar = progress(METHODS, desc="Evaluating methods", unit="method")
    for mi, method in enumerate(method_bar, start=1):
        method_bar.set_description_str(
            f"Method {mi}/{len(METHODS)} | Evaluating {method}")
        metrics_all: list[dict[str, Any]] = []

        for ti, traj in enumerate(progress(test, desc=f"    {method} trajectories",
                                           leave=False, unit="traj")):
            res = dispatch(method, traj, project, pinn=pinn, trans=trans,
                           collect_timing=False)

            if ti == 0:
                np.savez_compressed(
                    pred_dir / f"{method.replace('+', '_').replace('-', '_')}.npz",
                    time=traj.time,
                    truth=traj.states,
                    est=res.est,
                    nees=res.nees,
                    nees_rot=res.nees_rot,
                    nees_trans=res.nees_trans,
                    nis=res.nis,
                    innovations=res.innovations,
                    method=method,
                    scenario=str(traj.scenario),
                )

            m = dict(res.metrics)
            m["trial"] = ti
            m["scenario"] = str(traj.scenario)
            metrics_all.append(m)

        all_metrics[method] = pd.DataFrame(metrics_all)

    per_trial = pd.concat(
        [df.assign(method=method) for method, df in all_metrics.items()],
        ignore_index=True,
    )
    per_trial.to_csv(project.table_dir / "metrics_per_trial.csv", index=False)

    scenario_summary = _per_scenario_summary(per_trial)
    if not scenario_summary.empty:
        scenario_summary.to_csv(project.table_dir / "metrics_per_scenario.csv", index=False)

    # Ranking table with a paired test of every method against the rank-1
    # method, for every accuracy channel.
    rankings = pd.concat(
        [_ranking_table(all_metrics, metric) for metric in TESTED_METRICS],
        ignore_index=True,
    )
    rankings.to_csv(project.table_dir / "metric_rankings.csv", index=False)
    rankings.to_latex(project.table_dir / "metric_rankings.tex", index=False,
                      float_format="%.6g")

    base = all_metrics["EKF"].mean(numeric_only=True)
    rows: list[dict[str, Any]] = []

    for method, df in all_metrics.items():
        summary = df.mean(numeric_only=True).to_dict()
        summary["method"] = method
        summary["n_traj"] = len(test)

        for k, v in df.std(numeric_only=True).to_dict().items():
            summary[f"{k}_std"] = v
        for k, v in df.var(numeric_only=True).to_dict().items():
            summary[f"{k}_var"] = v

        for metric in KEY_METRICS:
            if metric in df.columns:
                lo, hi = confidence_interval(df[metric].values)
                summary[f"{metric}_ci95_lo"] = lo
                summary[f"{metric}_ci95_hi"] = hi
                if metric in base:
                    summary[f"{metric}_rel_impr_vs_ekf_pct"] = improvement(
                        base[metric], summary[metric]
                    )

        if method != "EKF":
            for metric in TESTED_METRICS:
                if metric in df.columns:
                    a = all_metrics["EKF"][metric].values
                    b = df[metric].values
                    summary[f"p_ttest_vs_ekf_{metric}"] = paired_ttest(a, b)
                    summary[f"p_wilcoxon_vs_ekf_{metric}"] = wilcoxon_test(a, b)
                    lo, hi = paired_bootstrap_ci(b, a)
                    summary[f"delta_vs_ekf_{metric}_ci95_lo"] = lo
                    summary[f"delta_vs_ekf_{metric}_ci95_hi"] = hi

        rows.append(summary)

    out = pd.DataFrame(rows)
    out.to_csv(project.table_dir / "metrics_summary.csv", index=False)
    out.to_latex(project.table_dir / "metrics_summary.tex", index=False,
                 float_format="%.6g")

    # Runtime is measured separately, after the accuracy pass.
    section("Runtime pass (interleaved repeats, leading repeat discarded)")
    run_runtime_benchmark(project, pinn=pinn, trans=trans)

    return rows
