from __future__ import annotations

"""Numerical support for the stability and observability claims.

Provides:
  1. Empirical local observability analysis: rank / condition number /
     minimum singular value of the discrete observability matrix
     O_k = [H_k; H_{k+1} F_k; ...] built from finite-difference Jacobians
     of the measurement map and the RK4 flow along test trajectories.
  2. Filter consistency check: fraction of NEES/NIS samples inside the
     95% chi-square bounds (supports a bounded-estimation-error claim:
     for a consistent filter, e_k' P_k^{-1} e_k bounded in probability
     implies ||e_k|| bounded whenever P_k is bounded).
  3. Bounded-error / ISS-style fit: fits ||e_k|| <= a * rho^k + b per
     trajectory and reports (a, rho, b); rho < 1 with small b is numerical
     evidence of exponential convergence to a bounded residual set.
  4. Lyapunov-candidate check: V_k = e_k' P_k^{-1} e_k (the NEES); reports
     the fraction of steps where E[V_k] stays below the chi-square bound
     and the empirical drift E[V_{k+1} - V_k | V_k > bound].

Run with: python main.py --mode theory
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, optimize

from src.dynamics.spacecraft import SpacecraftParams, rk4_step
from src.sensors.measurement_models import measurement_vector, measurement_dim
from src.evaluation.metrics import FULL_DOF, ROT_DOF, TRANS_DOF
from src.utils.math_utils import finite_difference_jacobian
from src.utils.progress import progress, section
from src.utils.quaternion import normalize_quaternion, quat_conjugate, quat_multiply, rotvec_from_quat


def _load_split(path: Path):
    data = np.load(path, allow_pickle=True)
    return list(data["trajectories"])


from src.evaluation.experiments import _env_at, _ctrl_at  # noqa: E402


def observability_analysis(project, n_traj: int = 5, horizon: int = 12) -> dict:
    """Empirical local observability Gramian along true trajectories."""
    params = project.spacecraft_params()
    include_rd = bool(project.config["synthetic"]["include_range_doppler"])
    dt = float(project.config["synthetic"]["dt"])
    test = _load_split(project.data_dir / "test.npz")[:n_traj]

    results = []
    for traj in progress(test, desc="  Observability: trajectories", unit="traj"):
        n = len(traj.time)
        for k0 in progress(np.linspace(0, n - horizon - 1, 4, dtype=int),
                           desc="    start points", leave=False, unit="point"):
            x = traj.states[k0].copy()
            env = _env_at(traj, k0)
            ctrl = _ctrl_at(traj, k0)

            def h(xx, env=env):
                return measurement_vector(xx, env, params, include_rd)

            blocks = []
            F_prod = np.eye(x.size)
            for j in progress(range(horizon), desc="      Jacobians / matrix product",
                              leave=False, unit="block"):
                kj = k0 + j
                envj = _env_at(traj, kj)
                ctrlj = _ctrl_at(traj, kj)
                H = finite_difference_jacobian(lambda xx: measurement_vector(xx, envj, params, include_rd), x)
                blocks.append(H @ F_prod)
                F = finite_difference_jacobian(
                    lambda xx: rk4_step(xx, traj.time[kj], dt, ctrlj, envj, params), x
                )
                F_prod = F @ F_prod
                x = rk4_step(x, traj.time[kj], dt, ctrlj, envj, params)
                x[:4] = normalize_quaternion(x[:4])

            O = np.vstack(blocks)
            sv = np.linalg.svd(O, compute_uv=False)
            tol = max(O.shape) * np.finfo(float).eps * sv[0]
            results.append(
                {
                    "scenario": str(getattr(traj, "scenario", "unknown")),
                    "k0": int(k0),
                    "rank": int(np.sum(sv > tol)),
                    "state_dim": int(x.size),
                    "min_singular_value": float(sv[-1]),
                    "condition_number": float(sv[0] / max(sv[-1], 1e-300)),
                }
            )

    df = pd.DataFrame(results)
    return {
        "n_evaluations": len(df),
        "full_rank_fraction": float(np.mean(df["rank"] == df["state_dim"])),
        "min_rank": int(df["rank"].min()),
        "state_dim": int(df["state_dim"].iloc[0]),
        "median_condition_number": float(df["condition_number"].median()),
        "median_min_singular_value": float(df["min_singular_value"].median()),
        "detail": results,
    }


def consistency_analysis(project) -> dict:
    """NEES/NIS chi-square coverage from saved predictions."""
    pred_dir = project.output_dir / "predictions"
    out = {}
    for f in progress(sorted(pred_dir.glob("*.npz")),
                      desc="  Consistency: chi-square coverage", unit="method"):
        d = np.load(f, allow_pickle=True)
        entry = {}
        for key, dof in (("nees", FULL_DOF), ("nees_rot", ROT_DOF), ("nees_trans", TRANS_DOF)):
            if key not in d:
                continue
            v = np.asarray(d[key], dtype=float)
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            lo, hi = stats.chi2.ppf([0.025, 0.975], dof)
            entry[f"{key}_mean"] = float(np.mean(v))
            entry[f"{key}_dof"] = dof
            entry[f"{key}_normalized"] = float(np.mean(v)) / float(dof)
            entry[f"{key}_within_95pct_bounds"] = float(np.mean((v >= lo) & (v <= hi)))
            entry[f"{key}_bounded_fraction"] = float(np.mean(v <= hi))
        if "nis" in d:
            nis = np.asarray(d["nis"], dtype=float)
            nis = nis[~np.isnan(nis)]
            if nis.size:
                dof = measurement_dim(bool(project.config["synthetic"]["include_range_doppler"]))
                lo, hi = stats.chi2.ppf([0.025, 0.975], dof)
                entry["nis_mean"] = float(np.mean(nis))
                entry["nis_dof"] = dof
                entry["nis_within_95pct_bounds"] = float(np.mean((nis >= lo) & (nis <= hi)))
        if entry:
            out[f.stem] = entry
    return out


def bounded_error_analysis(project) -> dict:
    """Fit ||e_k|| <= a rho^k + b for each method's saved trajectory."""
    pred_dir = project.output_dir / "predictions"
    out = {}
    for f in progress(sorted(pred_dir.glob("*.npz")),
                      desc="  Bounded-error fit", unit="method"):
        d = np.load(f, allow_pickle=True)
        if "truth" not in d or "est" not in d:
            continue
        truth = np.asarray(d["truth"], dtype=float)
        est = np.asarray(d["est"], dtype=float)
        n = len(truth)
        err = np.zeros(n)
        for k in progress(range(n), desc="    error norms", leave=False,
                          unit="step"):
            dq = quat_multiply(truth[k, :4], quat_conjugate(normalize_quaternion(est[k, :4])))
            att = rotvec_from_quat(dq)
            err[k] = np.linalg.norm(np.hstack([att, truth[k, 4:] - est[k, 4:]]))

        k_axis = np.arange(n, dtype=float)

        def model(k, a, rho, b):
            return a * np.power(np.clip(rho, 1e-6, 1.0), k) + b

        try:
            popt, _ = optimize.curve_fit(
                model, k_axis, err,
                p0=[max(err[0], 1e-6), 0.9, float(np.median(err[n // 2:]))],
                bounds=([0.0, 0.0, 0.0], [np.inf, 1.0, np.inf]),
                maxfev=20000,
            )
            a, rho, b = (float(v) for v in popt)
        except Exception:
            a, rho, b = float("nan"), float("nan"), float("nan")

        out[f.stem] = {
            "fit_a": a,
            "fit_rho": rho,
            "fit_b_asymptotic_bound": b,
            "sup_error": float(np.max(err)),
            "final_error": float(err[-1]),
            "steady_state_error_mean_last_25pct": float(np.mean(err[3 * n // 4:])),
            "error_is_bounded": bool(np.max(err) < 10.0 * max(err[0], 1e-9) + 10.0),
        }
    return out


def lyapunov_candidate_analysis(project) -> dict:
    """Empirical drift of V_k = e_k' P_k^{-1} e_k (stored as NEES)."""
    pred_dir = project.output_dir / "predictions"
    out = {}
    for f in progress(sorted(pred_dir.glob("*.npz")),
                      desc="  Lyapunov-candidate drift", unit="method"):
        d = np.load(f, allow_pickle=True)
        entry = {}
        for key, dof in (("nees", FULL_DOF), ("nees_rot", ROT_DOF), ("nees_trans", TRANS_DOF)):
            if key not in d:
                continue
            V = np.asarray(d[key], dtype=float)
            V = V[np.isfinite(V)]
            if V.size < 3:
                continue
            hi = stats.chi2.ppf(0.975, dof)
            dV = np.diff(V)
            above = V[:-1] > hi
            entry[key] = {
                "dof": dof,
                "V_mean": float(np.mean(V)),
                "V_sup": float(np.max(V)),
                "fraction_below_chi2_975": float(np.mean(V <= hi)),
                "mean_drift_when_above_bound": float(np.mean(dV[above])) if above.any() else 0.0,
                "negative_drift_above_bound": bool(above.any() and np.mean(dV[above]) < 0.0),
            }
        if entry:
            out[f.stem] = entry
    return out


def run_theory_analysis(project) -> dict:
    report = {}
    steps = [
        ("observability", "Step 1/4: local observability matrix (SVD, rank, condition)",
         observability_analysis),
        ("consistency_chi2", "Step 2/4: NEES / NIS chi-square consistency",
         consistency_analysis),
        ("bounded_error_fit", "Step 3/4: bounded-error / ISS fit  ||e_k|| <= a*rho^k + b",
         bounded_error_analysis),
        ("lyapunov_candidate", "Step 4/4: Lyapunov-candidate drift of V_k = e' P^-1 e",
         lyapunov_candidate_analysis),
    ]
    for key, message, func in steps:
        section(message)
        report[key] = func(project)

    out_path = project.output_dir / "theory_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Theory report written to {out_path}")
    obs = report["observability"]
    print(f"  Observability: rank {obs['min_rank']}/{obs['state_dim']} (min), "
          f"full-rank fraction {obs['full_rank_fraction']:.2f}")
    return report
