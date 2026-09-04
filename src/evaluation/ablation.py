from __future__ import annotations

"""Hyperparameter ablation.

One-at-a-time sweep around the baseline over PINN depth, width, window,
learning rate, loss weight and fusion scale.  Evaluated under the nominal
atmosphere on the standard test horizon, across every method the axis can
actually affect, reporting all accuracy channels plus the three NEES
partitions, disaggregated by scenario.

Run with:  python main.py --mode ablate
"""

import copy
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.experiments import METHODS, KEY_METRICS, _load_split, dispatch
from src.pinn.train import train_pinn, load_pinn
from src.models.train import load_transformer
from src.utils.profiling import repeat_timing, pin_threads
from src.utils.progress import progress, progress_bar, section
from src.utils.reproducibility import set_seed

DEFAULT_AXES = {
    "depth": [2, 4, 6],
    "hidden": [64, 128, 256],
    "window": [2, 4, 8],
    "lr": [3e-4, 1e-3, 3e-3],
    "lambda_norm": [0.0, 0.1, 1.0],
    "pinn_r_scale": [0.25, 1.0, 4.0, 16.0],
}

# Methods that consume the PINN weights.  The remaining methods cannot respond
# to a change in PINN architecture or training, so evaluating them once per
# axis value would only reproduce the same numbers at full cost.
PINN_METHODS = ["PINN-only", "PINN+EKF", "PINN+UKF", "PINN+MEKF", "PINN+UKF+MEKF"]

# Methods that fuse a learned prior through _fuse_pinn_prior, and so respond to
# pinn_r_scale.  PINN-only has no covariance and is excluded; Transformer+MEKF
# uses the same fusion path and is included.
FUSION_METHODS = ["PINN+EKF", "PINN+UKF", "PINN+MEKF", "PINN+UKF+MEKF",
                  "Transformer+MEKF"]

# Methods provably invariant to every axis here; run once as a reference row so
# the invariance can be checked rather than assumed.
INVARIANT_METHODS = [m for m in METHODS if m not in set(PINN_METHODS) | set(FUSION_METHODS)]

TRAINING_AXES = {"depth", "hidden", "window", "lr", "lambda_norm"}

METRIC_COLUMNS = KEY_METRICS + [
    "gyro_bias_rmse",
    "accel_bias_rmse",
    "disturbance_torque_rmse",
    "disturbance_accel_rmse",
    "nees_normalized",
    "nees_rot_normalized",
    "nees_trans_normalized",
    "nis_normalized",
]


def relevant_methods(axis: str) -> list[str]:
    """Methods whose output the axis can change."""
    if axis == "pinn_r_scale":
        return list(FUSION_METHODS)
    if axis in TRAINING_AXES:
        return list(PINN_METHODS)
    return list(METHODS)


def _estimate_cost(n_configs: int, seeds: int, n_test: int, steps: int,
                   avg_methods: float, s_per_step: float = 0.030) -> float:
    runs = n_configs * seeds * n_test * avg_methods
    return runs * steps * s_per_step / 3600.0


def _evaluate(project, pinn, trans, test, methods: Iterable[str]) -> list[dict[str, Any]]:
    """Accuracy pass: every requested method over every test trajectory."""
    rows: list[dict[str, Any]] = []
    for method in progress(list(methods), desc="      methods", leave=False,
                           unit="method"):
        for ti, traj in enumerate(progress(test, desc=f"        {method} trajectories",
                                           leave=False, unit="traj")):
            res = dispatch(method, traj, project, pinn=pinn, trans=trans,
                           collect_timing=False)
            row = {k: res.metrics.get(k, np.nan) for k in METRIC_COLUMNS}
            row.update({"method": method, "trial": ti,
                        "scenario": str(traj.scenario)})
            rows.append(row)
    return rows


def _runtime_pass(project, pinn, trans, traj, methods: Iterable[str],
                  repeats: int, discard_first: int, threads: int) -> list[dict[str, Any]]:
    """Runtime with methods interleaved inside the repeat loop.

    The leading repeat is discarded.  Without this the first configuration
    evaluated absorbs process warm-up (BLAS pool creation, torch lazy
    initialisation, allocator growth) and appears several times slower than
    configurations that merely happen to run later, which inverts the apparent
    ordering of an axis such as network depth.
    """
    methods = list(methods)
    n_steps = int(len(traj.time))
    samples: dict[str, list[float]] = {m: [] for m in methods}

    with pin_threads(threads):
        for rep in progress(range(repeats + discard_first),
                            desc="      runtime repeats", leave=False,
                            unit="repeat"):
            for method in methods:
                t0 = time.perf_counter()
                dispatch(method, traj, project, pinn=pinn, trans=trans,
                         collect_timing=False)
                samples[method].append((time.perf_counter() - t0) / max(n_steps, 1))

    out = []
    for method in methods:
        kept = np.asarray(samples[method][discard_first:], dtype=float)
        out.append({
            "method": method,
            "runtime_ms_mean": 1e3 * float(np.mean(kept)),
            "runtime_ms_std": 1e3 * (float(np.std(kept, ddof=1)) if kept.size > 1 else 0.0),
            "runtime_ms_min": 1e3 * float(np.min(kept)),
            "runtime_ms_median": 1e3 * float(np.median(kept)),
            "n_repeats_kept": int(kept.size),
            "n_repeats_discarded": int(discard_first),
        })
    return out


def _build_config(project, axis: str, value, epochs: int):
    """Return (cfg, proj) for one axis value, leaving the baseline untouched."""
    cfg = copy.deepcopy(project.config)
    cfg["training"]["pinn_epochs"] = epochs
    proj = copy.copy(project)

    if axis == "pinn_r_scale":
        proj.config = copy.deepcopy(project.config)
        proj.config.setdefault("fusion", {})["pinn_r_scale"] = float(value)
        return cfg, proj, False

    if axis == "lambda_norm":
        cfg["training"]["lambda_norm"] = float(value)
    elif axis == "lr":
        cfg["training"]["lr"] = float(value)
    else:
        cfg["training"][axis] = int(value)
    proj.config = cfg
    return cfg, proj, True


def run_ablation(project) -> pd.DataFrame:
    """Hyperparameter sensitivity under the nominal atmosphere."""
    abl = project.config.get("ablation", {})
    axes = {k: abl.get(k, v) for k, v in DEFAULT_AXES.items()}
    seeds = list(abl.get("seeds", [0, 1, 2]))
    n_test = int(abl.get("subset_test_trajectories", 20))
    n_train = int(abl.get("subset_train_trajectories", 40))
    epochs = int(abl.get("epochs", 15))
    repeats = int(abl.get("runtime_repeats", 3))
    discard_first = int(abl.get("runtime_discard_first", 1))
    threads = int(abl.get("pin_threads", 1))
    check_invariance = bool(abl.get("check_invariance", True))

    train = _load_split(project.data_dir / "train.npz")[:n_train]
    val = _load_split(project.data_dir / "val.npz")
    test = _load_split(project.data_dir / "test.npz")[:n_test]

    n_configs = sum(len(v) for v in axes.values())
    avg_methods = float(np.mean([len(relevant_methods(a)) for a, v in axes.items()
                                 for _ in v]))
    est = _estimate_cost(n_configs, len(seeds), n_test,
                         int(len(test[0].time)), avg_methods)
    print(f"[ablation] {n_configs} configurations x {len(seeds)} seeds, "
          f"{avg_methods:.1f} methods on average, {n_test} trajectories")
    print(f"[ablation] rough accuracy-pass estimate: {est:.1f} h "
          f"(set ablation.subset_test_trajectories lower to reduce)")

    baseline_pinn = load_pinn(project.model_dir / "pinn.pt", project.config["device"])
    trans = load_transformer(project.model_dir / "transformer.pt",
                             project.config["device"])

    # One untimed call so that process warm-up is not charged to whichever
    # configuration happens to be evaluated first.
    section("Warm-up call (untimed)")
    dispatch("PINN+UKF+MEKF", test[0], project, pinn=baseline_pinn, trans=trans)

    scratch = project.output_dir / "ablation_models"
    scratch.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    total_configs = sum(len(v) for v in axes.values())
    config_bar = progress_bar("Ablation configurations",
                              total=total_configs * len(seeds), unit="run")
    config_index = 0

    for axis, values in axes.items():
        methods = relevant_methods(axis)
        for value in values:
            config_index += 1
            for si, seed in enumerate(seeds, start=1):
                config_bar.set_description_str(
                    f"Configuration {config_index}/{total_configs} "
                    f"[{axis}={value}] seed {si}/{len(seeds)}")
                set_seed(int(seed))
                cfg, proj, needs_training = _build_config(project, axis, value, epochs)

                if needs_training:
                    pinn = train_pinn(
                        train, val, cfg,
                        scratch / f"{axis}_{value}_{seed}", cfg["device"],
                        progress_desc=f"      PINN [{axis}={value} seed={seed}]")
                else:
                    pinn = baseline_pinn

                new = _evaluate(proj, pinn, trans, test, methods)
                for r in new:
                    r.update({"axis": axis, "value": value, "seed": seed})
                rows.extend(new)

                if seed == seeds[0]:
                    for rt in _runtime_pass(proj, pinn, trans, test[0], methods,
                                            repeats, discard_first, threads):
                        rt.update({"axis": axis, "value": value, "seed": seed})
                        runtime_rows.append(rt)

                sub = pd.DataFrame(new)
                config_bar.write(
                    f"[ablation] {axis}={value} seed={seed}: "
                    f"att={sub['attitude_geodesic_rmse'].mean():.4f} "
                    f"pos={sub['position_rmse'].mean():.4f} "
                    f"vel={sub['velocity_rmse'].mean():.4e}")
                config_bar.update(1)

    # Reference row: methods that cannot respond to any axis.  Their spread
    # across configurations is a lower bound on measurement noise.
    config_bar.close()

    if check_invariance and INVARIANT_METHODS:
        section("Invariant-method reference pass")
        inv = _evaluate(project, baseline_pinn, trans, test, INVARIANT_METHODS)
        for r in inv:
            r.update({"axis": "invariant_reference", "value": "baseline",
                      "seed": seeds[0]})
        rows.extend(inv)

    df = pd.DataFrame(rows)
    project.table_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(project.table_dir / "ablation_results.csv", index=False)

    # NOTE: the DataFrame has a column literally named "axis". pandas'
    # DataFrameGroupBy.agg(list-of-funcs) internally probes `getattr(obj,
    # "axis", 0)`; when no real ".axis" attribute exists it falls back to
    # `obj["axis"]`, which collides with the *column* named "axis" on a
    # GroupBy that already has a column selection applied and raises
    # `IndexError: Column(s) [...] already selected` (seen with pandas
    # 3.0.x). Renaming the column around the aggregation call sidesteps the
    # collision without changing any on-disk column names.
    df_agg = df.rename(columns={"axis": "_axis"})
    agg = (df_agg.groupby(["_axis", "value", "method"])[METRIC_COLUMNS]
                 .agg(["mean", "std"]))
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index().rename(columns={"_axis": "axis"})
    agg.to_csv(project.table_dir / "ablation_summary.csv", index=False)

    per_scen = (df_agg.groupby(["_axis", "value", "method", "scenario"])[KEY_METRICS]
                      .agg(["mean", "std"]))
    per_scen.columns = ["_".join(c) for c in per_scen.columns]
    per_scen = per_scen.reset_index().rename(columns={"_axis": "axis"})
    per_scen.to_csv(project.table_dir / "ablation_per_scenario.csv", index=False)

    if runtime_rows:
        rt = pd.DataFrame(runtime_rows)
        rt.to_csv(project.table_dir / "ablation_runtime.csv", index=False)

    section("Generating ablation figures")
    _plot_axes(project, agg, axes)
    print(f"[ablation] written to {project.table_dir / 'ablation_results.csv'}")
    return df


def _plot_axes(project, agg: pd.DataFrame, axes: dict) -> None:
    """One panel per axis; a line per method so invariance is visible."""
    project.figure_dir.mkdir(parents=True, exist_ok=True)
    for axis in progress(list(axes), desc="  Accuracy figures", unit="axis"):
        sub = agg[agg["axis"] == axis]
        if sub.empty:
            continue
        for metric, label in [("attitude_geodesic_rmse", "Attitude geodesic RMSE [rad]"),
                              ("position_rmse", "Position RMSE [km]"),
                              ("velocity_rmse", "Velocity RMSE [km/s]"),
                              ("nees_normalized", "NEES / dof")]:
            col = f"{metric}_mean"
            if col not in sub.columns:
                continue
            fig, ax = plt.subplots(figsize=(6.5, 4))
            for method, g in sub.groupby("method", sort=False):
                g = g.copy()
                x = np.arange(len(g))
                ax.errorbar(x, g[col], yerr=g.get(f"{metric}_std"), marker="o",
                            capsize=3, lw=1.2, ms=4, label=method)
                ax.set_xticks(x)
                ax.set_xticklabels([str(v) for v in g["value"]])
            if metric == "nees_normalized":
                ax.axhline(1.0, color="k", ls="--", lw=0.9)
            ax.set_xlabel(axis)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(project.figure_dir / f"ablation_{axis}_{metric}.pdf")
            plt.close(fig)

    rt_path = project.table_dir / "ablation_runtime.csv"
    if not rt_path.exists():
        return
    rt = pd.read_csv(rt_path)
    for axis in progress(list(axes), desc="  Runtime figures", unit="axis"):
        sub = rt[rt["axis"] == axis]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for method, g in sub.groupby("method", sort=False):
            x = np.arange(len(g))
            ax.errorbar(x, g["runtime_ms_mean"], yerr=g["runtime_ms_std"],
                        marker="o", capsize=3, lw=1.2, ms=4, label=method)
            ax.plot(x, g["runtime_ms_min"], ls=":", lw=0.9, color="grey")
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in g["value"]])
        ax.set_xlabel(axis)
        ax.set_ylabel("Runtime per step [ms]")
        ax.set_title("solid = mean of kept repeats, dotted = minimum")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(project.figure_dir / f"ablation_{axis}_runtime.pdf")
        plt.close(fig)
