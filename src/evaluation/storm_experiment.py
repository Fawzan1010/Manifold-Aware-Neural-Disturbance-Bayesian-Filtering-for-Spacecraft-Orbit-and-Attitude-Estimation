from __future__ import annotations

"""Historical geomagnetic storm evaluation: 17 March 2015.

Every estimator in ``METHODS`` is evaluated on trajectories whose truth
dynamics are driven by NRLMSISE-00 with measured F10.7 and Ap for the St
Patrick's Day storm.  The trained PINN and Transformer checkpoints are loaded
and used unchanged: nothing is retrained, no hyperparameter is retuned and no
coupling coefficient is adjusted.

Two pairings are evaluated:

``mismatched`` (primary)
    Truth uses NRLMSISE-00 while the filters keep the exponential model, so
    the storm enters as unmodelled density error.  This is the case that
    discriminates between estimators, because the disturbance-estimating
    methods are the ones that can absorb it into ``a_d``.

``matched``
    Truth and filters both use NRLMSISE-00, isolating accuracy under real
    forcing from the effect of model error.

Distribution note
-----------------
The networks' feature vector contains ``weather_index``, which they only ever
saw in roughly [0.1, 1.6].  Feeding raw Ap (0-236) would place the input far
outside that range and would measure extrapolation failure rather than
estimator quality.  ``weather_index_mode: normalized`` maps Ap onto the
trained range; ``ap_raw`` retains the unscaled value as a sensitivity check.
Both leave the feature dimension unchanged, so the checkpoints load as-is.
"""

import json
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.dynamics.simulator import Trajectory, simulate_trajectory, build_params
from src.evaluation.experiments import (
    METHODS,
    KEY_METRICS,
    TESTED_METRICS,
    dispatch,
    _per_scenario_summary,
    _ranking_table,
)
from src.evaluation.stats import paired_ttest, wilcoxon_test, paired_bootstrap_ci
from src.utils.frames import parse_epoch
from src.utils.progress import progress, section
from src.utils.reproducibility import ensure_dir, set_seed
from src.utils.space_weather import load_omni


def storm_weather_series(project, times_s: np.ndarray, epoch) -> dict[str, np.ndarray]:
    """Sample the OMNI record across the arc and derive the activity index."""
    scfg = project.config["storm_experiment"]
    acfg = project.config.get("atmosphere", {})
    omni = load_omni(str(acfg.get("space_weather_file",
                                  "data/space_weather/omni2_2015.dat")))
    epoch = parse_epoch(epoch)

    ap = np.zeros(times_s.size)
    f107 = np.zeros(times_s.size)
    dst = np.zeros(times_s.size)
    for i, t in progress(enumerate(times_s), desc="  Sampling OMNI record",
                         total=int(times_s.size), leave=False, unit="step"):
        rec = omni.step_hold(epoch + timedelta(seconds=float(t)))
        ap[i], f107[i], dst[i] = rec.ap, rec.f107, rec.dst

    mode = str(scfg.get("weather_index_mode", "normalized")).lower()
    if mode == "ap_raw":
        wi = ap.copy()
    else:
        scale = float(scfg.get("weather_index_ap_scale", 100.0))
        wmax = float(scfg.get("weather_index_max", 1.6))
        wi = np.clip(ap / scale, 0.0, wmax)

    return {"ap": ap, "f107": f107, "dst": dst, "weather_index": wi}


def _apply_storm_weather(traj: Trajectory, series: dict[str, np.ndarray]) -> Trajectory:
    """Overwrite the synthetic activity index with the measured one."""
    traj.env["weather_index"] = series["weather_index"]
    traj.env["ap"] = series["ap"]
    traj.env["f107"] = series["f107"]
    traj.env["dst"] = series["dst"]
    return traj


def generate_storm_trajectories(project, truth_atmosphere: str = "nrlmsise00"):
    """Simulate the storm arc with the requested truth atmosphere."""
    scfg = project.config["storm_experiment"]
    dt = float(scfg.get("dt", 1.0))
    horizon = int(float(scfg.get("duration_hours", 3.0)) * 3600.0 / dt)
    n_traj = int(scfg.get("n_trajectories", 2))
    epoch = scfg.get("start_utc", "2015-03-17T00:00:00Z")
    scenario = str(scfg.get("scenario", "historical_storm"))

    params = build_params(
        {**project.config, "simulation": {**project.config["simulation"],
                                          "epoch_utc": epoch}},
        atmosphere_override=truth_atmosphere,
    )
    noise_cfg = project.config["measurement_noise"]
    include_rd = bool(project.config["synthetic"]["include_range_doppler"])

    times = np.arange(horizon, dtype=float) * dt
    series = storm_weather_series(project, times, epoch)

    out: list[Trajectory] = []
    for i in progress(range(n_traj),
                      desc=f"  Storm trajectories [truth={truth_atmosphere}]",
                      unit="traj"):
        traj = simulate_trajectory(
            seed=int(project.config["seed"]) + 5000 + 17 * i,
            scenario=scenario,
            dt=dt,
            horizon=horizon,
            params=params,
            noise_cfg=noise_cfg,
            include_range_doppler=include_rd,
            epoch_utc=epoch,
            # The synthetic storm noise multiplier is disabled: the forcing
            # must come from the measured atmosphere, not a tuned constant.
            storm_noise=False,
        )
        out.append(_apply_storm_weather(traj, series))
    return out, series, params


def _run_pairing(project, pairing: str, pinn, trans) -> tuple[pd.DataFrame, dict]:
    """Evaluate every method under one truth/filter atmosphere pairing."""
    truth_model = "nrlmsise00"
    filter_model = "exponential" if pairing == "mismatched" else "nrlmsise00"

    trajectories, series, truth_params = generate_storm_trajectories(
        project, truth_atmosphere=truth_model
    )

    # The filters see the world through `filter_model`; overriding it on the
    # project means every estimator picks it up through spacecraft_params()
    # without any estimator code changing.
    original = project.config.get("atmosphere", {}).get("model", "exponential")
    project.config.setdefault("atmosphere", {})["model"] = filter_model
    project.config["simulation"]["epoch_utc"] = project.config["storm_experiment"].get(
        "start_utc", "2015-03-17T00:00:00Z"
    )

    pred_dir = ensure_dir(Path(project.storm_dir) / pairing / "predictions")
    for stale in pred_dir.glob("*.npz"):
        stale.unlink()

    rows: list[dict[str, Any]] = []
    per_method: dict[str, pd.DataFrame] = {}
    method_bar = progress(METHODS, desc=f"  [{pairing}] methods", unit="method")
    try:
        for mi, method in enumerate(method_bar, start=1):
            method_bar.set_description_str(
                f"  [{pairing}] Method {mi}/{len(METHODS)}: {method}")
            method_rows = []
            for ti, traj in enumerate(progress(
                    trajectories, desc=f"    {method} storm trajectories",
                    leave=False, unit="traj")):
                res = dispatch(method, traj, project, pinn=pinn, trans=trans,
                               collect_timing=False, show_progress=True)
                if ti == 0:
                    np.savez_compressed(
                        pred_dir / f"{method.replace('+', '_').replace('-', '_')}.npz",
                        time=traj.time, truth=traj.states, est=res.est,
                        nees=res.nees, nees_rot=res.nees_rot,
                        nees_trans=res.nees_trans, nis=res.nis,
                        ap=series["ap"], dst=series["dst"],
                        rho_atm=traj.env.get("rho_atm", np.array([])),
                        method=method, pairing=pairing,
                    )
                m = dict(res.metrics)
                m.update({"trial": ti, "method": method, "pairing": pairing,
                          "scenario": str(traj.scenario)})
                method_rows.append(m)
                rows.append(m)
            per_method[method] = pd.DataFrame(method_rows)
    finally:
        method_bar.close()
        project.config["atmosphere"]["model"] = original

    meta = {
        "pairing": pairing,
        "truth_atmosphere": truth_model,
        "filter_atmosphere": filter_model,
        "n_trajectories": len(trajectories),
        "horizon_steps": int(len(trajectories[0].time)),
        "dt_s": float(trajectories[0].time[1] - trajectories[0].time[0]),
        "ap_min": float(series["ap"].min()),
        "ap_max": float(series["ap"].max()),
        "dst_min_nT": float(series["dst"].min()),
        "f107_mean": float(series["f107"].mean()),
        "weather_index_mode": project.config["storm_experiment"].get(
            "weather_index_mode", "normalized"),
        "weather_index_range": [float(series["weather_index"].min()),
                                float(series["weather_index"].max())],
        "rho_atm_mean": float(np.mean(trajectories[0].env["rho_atm"])),
        "rho_atm_max": float(np.max(trajectories[0].env["rho_atm"])),
        "atmosphere_description": truth_params.atmosphere.describe(),
        "retrained": False,
        "retuned": False,
    }
    return pd.DataFrame(rows), (meta, per_method)


def run_storm_experiment(project) -> dict:
    """Entry point for ``python main.py --mode storm``."""
    set_seed(int(project.config["seed"]))
    scfg = project.config.get("storm_experiment", {})
    pairings = list(scfg.get("pairings", ["mismatched", "matched"]))

    storm_dir = ensure_dir(Path(project.storm_dir))
    table_dir = ensure_dir(storm_dir / "tables")

    from src.pinn.train import load_pinn
    from src.models.train import load_transformer

    pinn = load_pinn(project.model_dir / "pinn.pt", project.config["device"])
    trans = load_transformer(project.model_dir / "transformer.pt",
                             project.config["device"])

    frames = []
    report: dict[str, Any] = {"date": scfg.get("date", "2015-03-17"),
                              "pairings": {}}

    for pi, pairing in enumerate(progress(pairings, desc="Storm pairings",
                                          unit="pairing"), start=1):
        section(f"Pairing {pi}/{len(pairings)}: {pairing}")
        df, (meta, per_method) = _run_pairing(project, pairing, pinn, trans)
        frames.append(df)
        report["pairings"][pairing] = meta

        rankings = pd.concat(
            [_ranking_table(per_method, m)
             for m in progress(TESTED_METRICS, desc="    Ranking tables",
                               leave=False, unit="metric")],
            ignore_index=True,
        )
        rankings.to_csv(table_dir / f"storm_rankings_{pairing}.csv", index=False)

        summary = []
        for method, mdf in progress(list(per_method.items()),
                                    desc="    Significance vs EKF",
                                    leave=False, unit="method"):
            row = mdf.mean(numeric_only=True).to_dict()
            row["method"] = method
            row["pairing"] = pairing
            if method != "EKF":
                for metric in TESTED_METRICS:
                    if metric in mdf.columns:
                        a = per_method["EKF"][metric].values
                        b = mdf[metric].values
                        row[f"p_ttest_vs_ekf_{metric}"] = paired_ttest(a, b)
                        row[f"p_wilcoxon_vs_ekf_{metric}"] = wilcoxon_test(a, b)
            summary.append(row)
        pd.DataFrame(summary).to_csv(
            table_dir / f"storm_metrics_summary_{pairing}.csv", index=False)

    per_trial = pd.concat(frames, ignore_index=True)
    per_trial.to_csv(table_dir / "storm_metrics_per_trial.csv", index=False)

    # Degradation of each method relative to its own matched-pairing result
    # isolates the cost of the unmodelled density error.
    if {"mismatched", "matched"} <= set(per_trial["pairing"].unique()):
        deg = []
        for method in progress(METHODS, desc="  Degradation table",
                               leave=False, unit="method"):
            sub = per_trial[per_trial["method"] == method]
            mm = sub[sub["pairing"] == "mismatched"]
            ma = sub[sub["pairing"] == "matched"]
            row = {"method": method}
            for metric in KEY_METRICS:
                if metric in sub.columns and len(mm) and len(ma):
                    a, b = float(ma[metric].mean()), float(mm[metric].mean())
                    row[f"{metric}_matched"] = a
                    row[f"{metric}_mismatched"] = b
                    row[f"{metric}_degradation_pct"] = 100.0 * (b - a) / max(abs(a), 1e-15)
            deg.append(row)
        pd.DataFrame(deg).to_csv(table_dir / "storm_degradation.csv", index=False)

    (storm_dir / "storm_report.json").write_text(json.dumps(report, indent=2, default=str))

    primary = report["pairings"].get("mismatched", {})
    print(f"Storm experiment written to {storm_dir}")
    if primary:
        print(f"  17 Mar 2015: ap {primary['ap_min']:.0f}-{primary['ap_max']:.0f}, "
              f"Dst min {primary['dst_min_nT']:.0f} nT, "
              f"{primary['horizon_steps']} steps x {primary['n_trajectories']} trajectories")
        print("  models reused without retraining or retuning")
    return report
