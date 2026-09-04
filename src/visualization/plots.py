from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.progress import progress, progress_bar, section


METHOD_STYLE = {
    "EKF": {"label": "EKF"},
    "Adaptive-EKF": {"label": "Adaptive-EKF"},
    "UKF": {"label": "UKF"},
    "MEKF": {"label": "MEKF"},
    "PINN-only": {"label": "PINN-only"},
    "Transformer-only": {"label": "Transformer-only"},
    "PINN+EKF": {"label": "PINN+EKF"},
    "PINN+UKF": {"label": "PINN+UKF"},
    "PINN+MEKF": {"label": "PINN+MEKF"},
    "Transformer+MEKF": {"label": "Transformer+MEKF"},
    "PINN+UKF+MEKF": {"label": "PINN+UKF+MEKF"},
}


def _load_npz_if_exists(path: Path) -> Optional[np.lib.npyio.NpzFile]:
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def _safe_save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _quat_geodesic_error(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    q_true = np.asarray(q_true, dtype=float)
    q_est = np.asarray(q_est, dtype=float)
    dot = np.abs(np.sum(q_true * q_est, axis=1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def _load_prediction(pred_dir: Path, name: str) -> Optional[dict]:
    f = _load_npz_if_exists(pred_dir / f"{name}.npz")
    if f is None:
        return None
    out = {}
    for k in f.files:
        out[k] = f[k]
    return out


def _plot_quaternion_norm(t: np.ndarray, truth: np.ndarray, methods: Dict[str, np.ndarray], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, np.linalg.norm(truth[:, :4], axis=1), label="Ground truth", linewidth=2)
    for name in ["EKF", "UKF", "MEKF", "PINN+UKF+MEKF"]:
        if name in methods:
            ax.plot(t, np.linalg.norm(methods[name][:, :4], axis=1), label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Quaternion norm")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _safe_save(fig, fig_dir / "quaternion_norm.pdf")


def _plot_position_error_time(t: np.ndarray, truth: np.ndarray, methods: Dict[str, np.ndarray], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["EKF", "UKF", "MEKF", "PINN+MEKF", "Transformer+MEKF", "PINN+UKF+MEKF"]:
        if name in methods:
            err = np.linalg.norm(methods[name][:, 7:10] - truth[:, 7:10], axis=1)
            ax.plot(t, err, label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position error [km]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _safe_save(fig, fig_dir / "position_error_time.pdf")


def _plot_velocity_error_time(t: np.ndarray, truth: np.ndarray, methods: Dict[str, np.ndarray], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["EKF", "UKF", "MEKF", "PINN+MEKF", "Transformer+MEKF", "PINN+UKF+MEKF"]:
        if name in methods:
            err = np.linalg.norm(methods[name][:, 10:13] - truth[:, 10:13], axis=1)
            ax.plot(t, err, label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity error [km/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _safe_save(fig, fig_dir / "velocity_error_time.pdf")


def _plot_bias_error_time(t: np.ndarray, truth: np.ndarray, methods: Dict[str, np.ndarray], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if "EKF" in methods:
        ax.plot(t, np.linalg.norm(methods["EKF"][:, 13:16] - truth[:, 13:16], axis=1), label="Gyro bias")
        ax.plot(t, np.linalg.norm(methods["EKF"][:, 16:19] - truth[:, 16:19], axis=1), label="Accel bias")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Bias error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _safe_save(fig, fig_dir / "bias_error_time.pdf")


def _plot_disturbance_error_time(t: np.ndarray, truth: np.ndarray, methods: Dict[str, np.ndarray], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if "EKF" in methods:
        ax.plot(t, np.linalg.norm(methods["EKF"][:, 19:22] - truth[:, 19:22], axis=1), label="Torque disturbance")
        ax.plot(t, np.linalg.norm(methods["EKF"][:, 22:25] - truth[:, 22:25], axis=1), label="Accel disturbance")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Disturbance error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _safe_save(fig, fig_dir / "disturbance_error_time.pdf")


def _plot_attitude_error_time(t: np.ndarray, truth: np.ndarray, methods: Dict[str, np.ndarray], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["EKF", "UKF", "MEKF", "PINN+EKF", "PINN+UKF", "PINN+MEKF", "Transformer+MEKF", "PINN+UKF+MEKF"]:
        if name in methods:
            e = _quat_geodesic_error(truth[:, :4], methods[name][:, :4])
            ax.plot(t, e, label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Attitude geodesic error [rad]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    _safe_save(fig, fig_dir / "attitude_error_time.pdf")


def _plot_3d_trajectory_comparison(truth: np.ndarray, methods: Dict[str, np.ndarray], fig_dir: Path) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(truth[:, 7], truth[:, 8], truth[:, 9], label="Ground truth", linewidth=2)

    for name in ["EKF", "MEKF", "PINN+MEKF", "Transformer+MEKF"]:
        if name in methods:
            est = methods[name]
            ax.plot(est[:, 7], est[:, 8], est[:, 9], label=name, alpha=0.9)

    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.legend()
    _safe_save(fig, fig_dir / "trajectory_3d_comparison.pdf")


def _plot_boxplots(per_trial: pd.DataFrame, fig_dir: Path) -> None:
    """Box plots over the Monte Carlo trials, one box per method.

    Consumes metrics_per_trial.csv; the one-row-per-method summary table
    would reduce each box to a single point.
    """
    box_metrics = [
        ("position_rmse", "Position RMSE [km]", "position_rmse_boxplot.pdf"),
        ("attitude_geodesic_rmse", "Attitude geodesic RMSE [rad]", "attitude_rmse_boxplot.pdf"),
        ("velocity_rmse", "Velocity RMSE [km/s]", "velocity_rmse_boxplot.pdf"),
        ("angular_rate_rmse", "Angular-rate RMSE [rad/s]", "angular_rate_rmse_boxplot.pdf"),
        ("runtime_per_step_s", "Runtime per step [s]", "runtime_boxplot.pdf"),
    ]
    methods = list(per_trial["method"].unique())
    for col, ylabel, fname in progress(box_metrics, desc="    box plots",
                                       leave=False, unit="fig"):
        if col not in per_trial.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        data = [per_trial.loc[per_trial["method"] == m, col].dropna().values for m in methods]
        # matplotlib >= 3.9 renamed this kwarg; try the current name first
        # and fall back so the figure still renders on older installs.
        try:
            ax.boxplot(data, tick_labels=methods, showfliers=True)
        except TypeError:
            ax.boxplot(data, labels=methods, showfliers=True)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        _safe_save(fig, fig_dir / fname)


def _plot_ci_bars(per_trial: pd.DataFrame, fig_dir: Path) -> None:
    """Mean with 95% confidence-interval error bars."""
    from scipy import stats as _st

    for col, ylabel, fname in progress([
        ("position_rmse", "Position RMSE [km]", "position_rmse_ci.pdf"),
        ("attitude_geodesic_rmse", "Attitude geodesic RMSE [rad]", "attitude_rmse_ci.pdf"),
        ("runtime_per_step_s", "Runtime per step [s]", "runtime_ci.pdf"),
    ], desc="    CI bars", leave=False, unit="fig"):
        if col not in per_trial.columns:
            continue
        methods, means, errs = [], [], []
        for m, g in per_trial.groupby("method", sort=False):
            v = g[col].dropna().values
            if v.size < 2:
                continue
            methods.append(m)
            means.append(np.mean(v))
            errs.append(_st.sem(v) * _st.t.ppf(0.975, len(v) - 1))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(methods, means, yerr=errs, capsize=5)
        ax.set_ylabel(ylabel + " (mean ± 95% CI)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        _safe_save(fig, fig_dir / fname)


def _plot_cdf_from_summary(metrics: pd.DataFrame, column: str, fig_dir: Path, fname: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, group in metrics.groupby("method"):
        if column not in group:
            continue
        vals = np.asarray(group[column].dropna(), dtype=float)
        if vals.size == 0:
            continue
        vals = np.sort(vals)
        cdf = np.arange(1, len(vals) + 1, dtype=float) / len(vals)
        ax.plot(vals, cdf, label=method)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(error ≤ x)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _safe_save(fig, fig_dir / fname)


def _plot_bar(metrics: pd.DataFrame, column: str, fig_dir: Path, fname: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(metrics["method"], metrics[column])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    _safe_save(fig, fig_dir / fname)


def _plot_radar(metrics: pd.DataFrame, fig_dir: Path) -> None:
    top = metrics.sort_values("position_rmse").head(5).copy()
    cols = [
        "attitude_geodesic_rmse",
        "position_rmse",
        "velocity_rmse",
        "runtime_per_step_s",
        "memory_usage_mb",
    ]
    vals = top[cols].to_numpy(dtype=float)

    # Normalize per column so metrics share a comparable scale.
    denom = np.maximum(np.nanmax(vals, axis=0) - np.nanmin(vals, axis=0), 1e-12)
    norm = (vals - np.nanmin(vals, axis=0)) / denom

    labels = ["Attitude", "Position", "Velocity", "Runtime", "Memory"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111, polar=True)

    for i, row in enumerate(norm):
        data = row.tolist()
        data += data[:1]
        ax.plot(angles, data, label=top.iloc[i]["method"])
        ax.fill(angles, data, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Radar comparison of top methods")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _safe_save(fig, fig_dir / "radar_comparison.pdf")


def _plot_pareto(metrics: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(metrics["runtime_per_step_s"], metrics["position_rmse"])
    for _, row in metrics.iterrows():
        ax.annotate(row["method"], (row["runtime_per_step_s"], row["position_rmse"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Runtime per step [s]")
    ax.set_ylabel("Position RMSE")
    ax.grid(True, alpha=0.3)
    _safe_save(fig, fig_dir / "pareto_front.pdf")


def _plot_innovation_traces(pred_dir: Path, fig_dir: Path) -> None:
    # If innovations are saved as .npz arrays, plot their norms and mean traces.
    for name in progress(["EKF", "UKF", "MEKF", "PINN_UKF_MEKF", "PINN_EKF",
                          "PINN_MEKF", "Transformer_MEKF"],
                         desc="    innovation traces", leave=False, unit="method"):
        f = _load_npz_if_exists(pred_dir / f"{name}.npz")
        if f is None or "innovations" not in f.files:
            continue

        innov = np.asarray(f["innovations"], dtype=float)
        if innov.ndim == 1:
            innov = innov[:, None]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.mean(innov, axis=1), label="Mean innovation")
        ax.set_xlabel("Step")
        ax.set_ylabel("Innovation mean")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _safe_save(fig, fig_dir / f"{name.lower()}_innovation_mean.pdf")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.var(innov, axis=1), label="Innovation variance")
        ax.set_xlabel("Step")
        ax.set_ylabel("Innovation variance")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _safe_save(fig, fig_dir / f"{name.lower()}_innovation_var.pdf")


def make_all_plots(project) -> None:
    fig_dir = project.figure_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = project.table_dir / "metrics_summary.csv"
    # Runtime and memory are produced by the separate benchmark pass and are
    # no longer columns of the accuracy summary; merge them back in so the
    # cost-versus-accuracy figures have both axes.
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics table: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    metrics = _merge_runtime(metrics, project.table_dir)
    pred_dir = project.output_dir / "predictions"

    ekf = _load_prediction(pred_dir, "EKF")
    ukf = _load_prediction(pred_dir, "UKF")
    mekf = _load_prediction(pred_dir, "MEKF")
    hybrid = _load_prediction(pred_dir, "PINN_UKF_MEKF")
    pinn_mekf = _load_prediction(pred_dir, "PINN_MEKF")
    trans_mekf = _load_prediction(pred_dir, "Transformer_MEKF")
    pinn_ekf = _load_prediction(pred_dir, "PINN_EKF")
    pinn_ukf = _load_prediction(pred_dir, "PINN_UKF")

    available = {
        "EKF": ekf["est"] if ekf is not None and "est" in ekf else None,
	"UKF": ukf["est"] if ukf is not None and "est" in ukf else None,
	"MEKF": mekf["est"] if mekf is not None and "est" in mekf else None,
	"PINN+UKF+MEKF": hybrid["est"] if hybrid is not None and "est" in hybrid else None,
	"PINN+MEKF": pinn_mekf["est"] if pinn_mekf is not None and "est" in pinn_mekf else None,
	"Transformer+MEKF": trans_mekf["est"] if trans_mekf is not None and "est" in trans_mekf else None,
	"PINN+EKF": pinn_ekf["est"] if pinn_ekf is not None and "est" in pinn_ekf else None,
	"PINN+UKF": pinn_ukf["est"] if pinn_ukf is not None and "est" in pinn_ukf else None,
    }

    # Use EKF as the representative trajectory source.
    if ekf is None or "time" not in ekf or "truth" not in ekf:
        raise FileNotFoundError("EKF prediction file must contain 'time' and 'truth' arrays.")

    t = np.asarray(ekf["time"], dtype=float)
    truth = np.asarray(ekf["truth"], dtype=float)

    # The figure sequence below is the same list of calls, in the same order,
    # as before; collecting it first only lets the loop report which figure
    # group is being produced and how many remain.
    per_trial_path = project.table_dir / "metrics_per_trial.csv"
    has_per_trial = per_trial_path.exists()
    has_runtime = "runtime_per_step_s" in metrics.columns
    per_trial = pd.read_csv(per_trial_path) if has_per_trial else None
    cdf_source = per_trial if has_per_trial else metrics

    tasks: List[Tuple[str, str, object]] = []

    # -- Core time-series plots
    tasks += [
        ("Time series", "quaternion_norm",
         lambda: _plot_quaternion_norm(t, truth, available, fig_dir)),
        ("Time series", "position_error_time",
         lambda: _plot_position_error_time(t, truth, available, fig_dir)),
        ("Time series", "velocity_error_time",
         lambda: _plot_velocity_error_time(t, truth, available, fig_dir)),
        ("Time series", "bias_error_time",
         lambda: _plot_bias_error_time(t, truth, available, fig_dir)),
        ("Time series", "disturbance_error_time",
         lambda: _plot_disturbance_error_time(t, truth, available, fig_dir)),
        ("Time series", "attitude_error_time",
         lambda: _plot_attitude_error_time(t, truth, available, fig_dir)),
    ]

    # -- Comparison bars
    tasks += [
        ("Comparison", "comparison_bar_position",
         lambda: _plot_bar(metrics, "position_rmse", fig_dir,
                           "comparison_bar_position.pdf", "Position RMSE")),
        ("Comparison", "comparison_bar_attitude",
         lambda: _plot_bar(metrics, "attitude_geodesic_rmse", fig_dir,
                           "comparison_bar_attitude.pdf", "Attitude geodesic RMSE")),
    ]
    if has_runtime:
        tasks += [
            ("Comparison", "runtime_comparison",
             lambda: _plot_bar(metrics, "runtime_per_step_s", fig_dir,
                               "runtime_comparison.pdf", "Runtime per step [s]")),
            ("Comparison", "runtime_repeats",
             lambda: _plot_runtime_repeats(project.table_dir, fig_dir)),
        ]

    # -- Publication-style plots
    tasks.append(("Publication", "trajectory_3d_comparison",
                  lambda: _plot_3d_trajectory_comparison(truth, available, fig_dir)))
    if has_per_trial:
        tasks += [
            ("Distribution", "boxplots", lambda: _plot_boxplots(per_trial, fig_dir)),
            ("Distribution", "ci_bars", lambda: _plot_ci_bars(per_trial, fig_dir)),
        ]
    tasks += [
        ("Distribution", "position_cdf",
         lambda: _plot_cdf_from_summary(cdf_source, "position_rmse", fig_dir,
                                        "position_cdf.pdf", "Position RMSE")),
        ("Distribution", "attitude_cdf",
         lambda: _plot_cdf_from_summary(cdf_source, "attitude_geodesic_rmse", fig_dir,
                                        "attitude_cdf.pdf", "Attitude geodesic RMSE")),
    ]
    if has_runtime:
        tasks.append(("Publication", "radar_comparison",
                      lambda: _plot_radar(metrics, fig_dir)))
    if has_runtime:
        tasks.append(("Publication", "pareto_front",
                      lambda: _plot_pareto(metrics, fig_dir)))

    # -- Per-scenario disaggregation and the paired velocity comparison
    if has_per_trial:
        tasks += [
            ("Per-scenario", "per_scenario",
             lambda: _plot_per_scenario(per_trial, fig_dir)),
            ("Per-scenario", "paired_difference_velocity",
             lambda: _plot_paired_difference(per_trial, fig_dir, "velocity_rmse",
                                             "Velocity RMSE [km/s]")),
            ("Per-scenario", "ranking_significance",
             lambda: _plot_ranking_significance(project.table_dir, fig_dir)),
        ]

    # -- Innovation traces if available
    tasks.append(("Innovations", "innovation_traces",
                  lambda: _plot_innovation_traces(pred_dir, fig_dir)))

    # -- NEES: full state and the two partitions, each against its own bounds
    tasks += [
        ("Consistency", "nees_partitions",
         lambda: _plot_nees_partitions(t, ekf, fig_dir)),
        ("Consistency", "nees_partition_bars",
         lambda: _plot_nees_partition_bars(project.table_dir, fig_dir)),
    ]

    if ekf is not None and "nis" in ekf:
        tasks.append(("Consistency", "nis_plot", lambda: _plot_nis(t, ekf, fig_dir)))

    total = len(tasks)
    section(f"{total} figure groups to generate")
    bar = progress_bar("Generating figures", total=total, unit="fig")
    last_category = None
    for i, (category, name, func) in enumerate(tasks, start=1):
        if category != last_category:
            bar.write(f"  Category: {category}")
            last_category = category
        bar.set_description_str(f"Generating figure group {i} of {total} [{name}]")
        func()
        bar.update(1)
    bar.close()
    section("Export complete")


def _plot_nis(t: np.ndarray, ekf: dict, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, ekf["nis"], label="NIS")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("NIS")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _safe_save(fig, fig_dir / "nis_plot.pdf")


# --------------------------------------------------------------------------
# Runtime (measured by the separate benchmark pass)
# --------------------------------------------------------------------------

def _merge_runtime(metrics: pd.DataFrame, table_dir: Path) -> pd.DataFrame:
    """Attach runtime/memory columns from runtime_breakdown.csv.

    These are no longer columns of metrics_summary.csv because runtime is
    measured in a separate repeated pass rather than inline with the accuracy
    run.  Figures that plot cost against accuracy need both, so they are
    joined here on method name.
    """
    path = table_dir / "runtime_breakdown.csv"
    if not path.exists():
        return metrics
    rt = pd.read_csv(path)
    keep = {"method": "method"}
    if "total_ms" in rt.columns:
        rt["runtime_per_step_s"] = rt["total_ms"] / 1e3
        keep["runtime_per_step_s"] = "runtime_per_step_s"
    for c in ("total_ms_min", "total_ms_std", "pinn_ms", "filter_ms", "memory_usage_mb"):
        if c in rt.columns:
            keep[c] = c
    return metrics.merge(rt[list(keep)], on="method", how="left")


def _plot_runtime_repeats(table_dir: Path, fig_dir: Path) -> None:
    """Per-repeat runtime samples, discarded warm-up marked separately.

    Plotting the individual repeats rather than only their mean makes any
    residual drift in machine load visible instead of hidden inside an
    error bar.
    """
    path = table_dir / "runtime_repeats.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    methods = list(dict.fromkeys(df["method"]))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, m in enumerate(methods):
        g = df[df["method"] == m]
        kept = g[~g["discarded"]]
        drop = g[g["discarded"]]
        ax.scatter(np.full(len(kept), i), kept["ms_per_step"], s=26,
                   color="tab:blue", label="kept" if i == 0 else None)
        ax.scatter(np.full(len(drop), i), drop["ms_per_step"], s=30, marker="x",
                   color="tab:red", label="discarded (warm-up)" if i == 0 else None)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Runtime per step [ms]")
    ax.set_title("Per-repeat runtime samples")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _safe_save(fig, fig_dir / "runtime_repeats.pdf")


# --------------------------------------------------------------------------
# NEES partitions
# --------------------------------------------------------------------------

def _plot_nees_partitions(t: np.ndarray, pred: Optional[dict], fig_dir: Path) -> None:
    """Full-state, rotational and translational NEES against chi-square bounds.

    Each panel uses its own degrees of freedom (24, 12, 12), so the bounds
    differ between panels and the three traces are not directly comparable in
    magnitude.
    """
    if pred is None:
        return
    from scipy import stats as _st

    panels = [
        ("nees", 24, "Full state (24 DoF)"),
        ("nees_rot", 12, "Rotational (12 DoF)"),
        ("nees_trans", 12, "Translational (12 DoF)"),
    ]
    available = [(k, d, lbl) for k, d, lbl in panels if k in pred]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(9, 3.0 * len(available)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (key, dof, title) in zip(axes, available):
        v = np.asarray(pred[key], dtype=float)
        lo, hi = _st.chi2.ppf([0.025, 0.975], dof)
        ax.plot(t[: v.size], v, lw=0.8, label="NEES")
        ax.axhline(dof, color="k", ls="-", lw=0.9, label="expected (= DoF)")
        ax.axhline(hi, color="r", ls="--", lw=0.9, label="95% bounds")
        ax.axhline(lo, color="r", ls="--", lw=0.9)
        finite = v[np.isfinite(v)]
        if finite.size:
            inside = float(np.mean((finite >= lo) & (finite <= hi)))
            ax.set_title(f"{title} — mean {finite.mean():.1f}, {inside:.0%} within bounds",
                         fontsize=10)
        ax.set_ylabel("NEES")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    _safe_save(fig, fig_dir / "nees_partitions.pdf")


def _plot_nees_partition_bars(table_dir: Path, fig_dir: Path) -> None:
    """Normalised NEES per method for each partition.

    Values are divided by their degrees of freedom so all three partitions
    share a common target of 1.0 and can sit on one axis.
    """
    path = table_dir / "metrics_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    cols = [("nees_normalized", "Full (24 DoF)"),
            ("nees_rot_normalized", "Rotational (12 DoF)"),
            ("nees_trans_normalized", "Translational (12 DoF)")]
    cols = [(c, lbl) for c, lbl in cols if c in df.columns]
    if not cols:
        return

    methods = df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (c, lbl) in enumerate(cols):
        ax.bar(x + i * width - 0.4 + width / 2, df[c].values, width, label=lbl)
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="consistent (= 1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("NEES / DoF")
    ax.set_title("Filter consistency by state partition")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    _safe_save(fig, fig_dir / "nees_partition_bars.pdf")


# --------------------------------------------------------------------------
# Scenario disaggregation and paired comparisons
# --------------------------------------------------------------------------

def _plot_per_scenario(per_trial: pd.DataFrame, fig_dir: Path) -> None:
    """Metrics split by scenario.

    Scenario difficulty varies by more than the spread between methods, so a
    pooled standard deviation is dominated by the scenario mix; disaggregating
    is what makes the between-method differences legible.
    """
    if "scenario" not in per_trial.columns:
        return
    for col, ylabel, fname in progress([
        ("position_rmse", "Position RMSE [km]", "position_rmse_by_scenario.pdf"),
        ("velocity_rmse", "Velocity RMSE [km/s]", "velocity_rmse_by_scenario.pdf"),
        ("attitude_geodesic_rmse", "Attitude RMSE [rad]", "attitude_rmse_by_scenario.pdf"),
    ], desc="    per-scenario figures", leave=False, unit="fig"):
        if col not in per_trial.columns:
            continue
        pivot = per_trial.pivot_table(index="scenario", columns="method",
                                      values=col, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(12, 4.5))
        pivot.plot(kind="bar", ax=ax, width=0.85)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Scenario")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7, ncol=3)
        _safe_save(fig, fig_dir / fname)


def _plot_paired_difference(per_trial: pd.DataFrame, fig_dir: Path,
                            metric: str, label: str) -> None:
    """Per-trial paired differences against the best method on `metric`.

    Trials are paired by trajectory, so the difference cancels the shared
    trajectory effect.  A bar chart of pooled means cannot show this because
    its error bars are dominated by between-trajectory variance.
    """
    if metric not in per_trial.columns or "trial" not in per_trial.columns:
        return
    wide = per_trial.pivot_table(index="trial", columns="method", values=metric)
    if wide.empty:
        return
    best = wide.mean().idxmin()
    diff = wide.subtract(wide[best], axis=0).drop(columns=[best])
    order = diff.mean().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    boxdata = [diff[m].dropna().values for m in order]
    # matplotlib >= 3.9 renamed this kwarg; try the current name first
    # and fall back so the figure still renders on older installs.
    try:
        ax.boxplot(boxdata, tick_labels=order, showfliers=False)
    except TypeError:
        ax.boxplot(boxdata, labels=order, showfliers=False)
    ax.axhline(0.0, color="r", ls="--", lw=1.0)
    ax.set_ylabel(f"{label}\ndifference vs {best}")
    ax.set_title(f"Paired per-trial difference in {label} (positive = worse than {best})")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    _safe_save(fig, fig_dir / f"{metric}_paired_difference.pdf")


def _plot_ranking_significance(table_dir: Path, fig_dir: Path) -> None:
    """Mean and paired 95% CI against the rank-1 method, per metric."""
    path = table_dir / "metric_rankings.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    metrics = list(dict.fromkeys(df["metric"]))
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3.2 * len(metrics)))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        sub = df[df["metric"] == metric].sort_values("rank")
        lo = sub["delta_vs_best"] - sub["delta_ci95_lo"]
        hi = sub["delta_ci95_hi"] - sub["delta_vs_best"]
        colors = ["tab:red" if w else "tab:blue"
                  for w in sub.get("significantly_worse_than_best", False)]
        ax.errorbar(range(len(sub)), sub["delta_vs_best"],
                    yerr=[lo.abs(), hi.abs()], fmt="none", ecolor="gray", capsize=3)
        ax.scatter(range(len(sub)), sub["delta_vs_best"], c=colors, zorder=3)
        ax.axhline(0.0, color="k", ls="--", lw=0.9)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["method"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Δ vs best")
        ax.set_title(f"{metric} (red = CI excludes zero)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    _safe_save(fig, fig_dir / "ranking_significance.pdf")


# --------------------------------------------------------------------------
# Historical storm experiment
# --------------------------------------------------------------------------

def _storm_predictions(pairing_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    pred_dir = pairing_dir / "predictions"
    if not pred_dir.exists():
        return out
    for f in sorted(pred_dir.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        out[str(d["method"])] = {k: d[k] for k in d.files}
    return out


def _plot_storm_drivers(pred: dict, fig_dir: Path, pairing: str) -> None:
    """Measured Ap and Dst alongside the resulting density."""
    if "ap" not in pred:
        return
    t_h = np.asarray(pred["time"], dtype=float) / 3600.0
    fig, ax1 = plt.subplots(figsize=(9, 4))

    rho = np.asarray(pred.get("rho_atm", []), dtype=float)
    if rho.size:
        ax1.plot(t_h[: rho.size], rho[: t_h.size], color="tab:blue", lw=1.2)
        ax1.set_ylabel("Atmospheric density [kg/m³]", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xlabel("Hours from 2015-03-17 00:00 UTC")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ap = np.asarray(pred["ap"], dtype=float)
    ax2.plot(t_h[: ap.size], ap[: t_h.size], color="tab:red", lw=1.2, ls="--", label="ap")
    if "dst" in pred:
        dst = np.asarray(pred["dst"], dtype=float)
        ax2.plot(t_h[: dst.size], dst[: t_h.size], color="tab:green", lw=1.2,
                 ls=":", label="Dst [nT]")
    ax2.set_ylabel("ap / Dst", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(fontsize=8, loc="upper left")

    ax1.set_title(f"St Patrick's Day storm drivers and density ({pairing})")
    _safe_save(fig, fig_dir / f"storm_drivers_{pairing}.pdf")


def _plot_storm_error_time(preds: Dict[str, dict], fig_dir: Path, pairing: str) -> None:
    """Position error against time for every method under storm forcing."""
    if not preds:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for method, d in preds.items():
        if "truth" not in d or "est" not in d:
            continue
        t_h = np.asarray(d["time"], dtype=float) / 3600.0
        err = np.linalg.norm(np.asarray(d["truth"])[:, 7:10]
                             - np.asarray(d["est"])[:, 7:10], axis=1)
        ax.plot(t_h, err, lw=0.9, label=method)
    ax.set_xlabel("Hours from 2015-03-17 00:00 UTC")
    ax.set_ylabel("Position error [km]")
    ax.set_yscale("log")
    ax.set_title(f"Position error during the storm ({pairing})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=3)
    _safe_save(fig, fig_dir / f"storm_position_error_{pairing}.pdf")


def _plot_storm_degradation(table_dir: Path, fig_dir: Path) -> None:
    """Percentage degradation from the matched to the mismatched pairing.

    This isolates the cost of an unmodelled density error: both bars use the
    same truth trajectories, differing only in the atmosphere the filters
    assume.
    """
    path = table_dir / "storm_degradation.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.endswith("_degradation_pct")]
    if not cols:
        return
    x = np.arange(len(df))
    width = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, c in enumerate(cols):
        ax.bar(x + i * width - 0.4 + width / 2, df[c].values, width,
               label=c.replace("_degradation_pct", ""))
    ax.axhline(0.0, color="k", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=45, ha="right")
    ax.set_ylabel("Degradation, mismatched vs matched [%]")
    ax.set_title("Cost of an unmodelled storm-time density error")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    _safe_save(fig, fig_dir / "storm_degradation.pdf")


def make_storm_plots(project) -> None:
    """Figures for the 17 March 2015 storm experiment."""
    storm_dir = Path(project.storm_dir)
    if not storm_dir.exists():
        return
    fig_dir = storm_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir = storm_dir / "tables"

    pairing_dirs = sorted(p for p in storm_dir.iterdir() if p.is_dir())
    for pairing_dir in progress(pairing_dirs, desc="Storm figures", unit="pairing"):
        pairing = pairing_dir.name
        if pairing in {"figures", "tables"}:
            continue
        preds = _storm_predictions(pairing_dir)
        if not preds:
            continue
        _plot_storm_error_time(preds, fig_dir, pairing)
        first = next(iter(preds.values()))
        _plot_storm_drivers(first, fig_dir, pairing)
        _plot_nees_partitions(np.asarray(first["time"], dtype=float),
                              preds.get("PINN+UKF+MEKF", first), fig_dir)
        src = fig_dir / "nees_partitions.pdf"
        if src.exists():
            src.rename(fig_dir / f"storm_nees_partitions_{pairing}.pdf")

    _plot_storm_degradation(table_dir, fig_dir)
    section(f"Storm figures written to {fig_dir}")
