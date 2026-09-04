from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.dynamics.spacecraft import (
    SpacecraftParams,
    control_profile,
    environmental_fields,
    rk4_step,
)
from src.sensors.measurement_models import (
    measurement_dim,
    measurement_noise_cov,
    measurement_vector,
)
from src.dynamics.atmosphere import make_atmosphere
from src.utils.progress import progress, progress_bar

# A per-timestep bar is only worth drawing for arcs long enough to take real
# time.  The 240-step benchmark horizon is sub-second and would just flood the
# console; the multi-thousand-step storm arcs are where progress matters.
PROGRESS_MIN_STEPS = 1000
from src.utils.quaternion import normalize_quaternion
from src.utils.state import pack_state


def build_params(cfg: dict, atmosphere_override: str | None = None) -> SpacecraftParams:
    """Construct SpacecraftParams with the configured atmosphere attached.

    The YAML ``simulation`` block is splatted into the dataclass, so the
    atmosphere settings live in their own top-level block and are injected
    here rather than being passed through as a string.
    """
    sim = {k: v for k, v in (cfg.get("simulation", {}) or {}).items()
           if k not in {"epoch_utc"}}
    params = SpacecraftParams(**sim)
    params.epoch_utc = cfg.get("simulation", {}).get("epoch_utc")
    params.atmosphere = make_atmosphere(cfg, override_model=atmosphere_override)
    return params


@dataclass
class Trajectory:
    scenario: str
    seed: int
    time: np.ndarray
    states: np.ndarray
    measurements: np.ndarray
    measurement_mask: np.ndarray
    controls: np.ndarray
    env: dict
    provenance: dict[str, Any]
    epoch_utc: str | None = None


def initial_state_for_scenario(
    rng: np.random.Generator,
    params: SpacecraftParams,
    scenario: str,
) -> np.ndarray:
    q0 = normalize_quaternion(
        np.array(
            [
                1.0,
                0.05 * rng.standard_normal(),
                0.05 * rng.standard_normal(),
                0.05 * rng.standard_normal(),
            ],
            dtype=float,
        )
    )
    w0 = np.array([0.02, -0.018, 0.015], dtype=float) + 0.005 * rng.standard_normal(3)
    r0 = np.array([params.radius_earth + 500.0, 0.0, 0.0], dtype=float) + 10.0 * rng.standard_normal(3)
    v0 = np.array([0.0, 7.6, 0.0], dtype=float) + 0.03 * rng.standard_normal(3)
    bg0 = 0.003 * rng.standard_normal(3)
    ba0 = 0.03 * rng.standard_normal(3)
    td0 = 4e-4 * rng.standard_normal(3)
    ad0 = 4e-5 * rng.standard_normal(3)

    if scenario == "biased_sensors":
        bg0 += np.array([0.04, -0.03, 0.02], dtype=float)
        ba0 += np.array([0.30, -0.18, 0.12], dtype=float)

    if scenario == "high_disturbance":
        td0 += np.array([1.5e-3, -1.2e-3, 9e-4], dtype=float)
        ad0 += np.array([8e-5, -6e-5, 4e-5], dtype=float)

    return pack_state(q0, w0, r0, v0, bg0, ba0, td0, ad0)


def simulate_trajectory(
    seed: int,
    scenario: str,
    dt: float,
    horizon: int,
    params: SpacecraftParams,
    noise_cfg: dict,
    include_range_doppler: bool = True,
    epoch_utc: str | None = None,
    storm_noise: bool = True,
    show_progress: bool = True,
) -> Trajectory:
    rng = np.random.default_rng(seed)
    x = initial_state_for_scenario(rng, params, scenario)

    time = np.arange(horizon, dtype=float) * dt
    n_meas = measurement_dim(include_range_doppler)

    states = np.zeros((horizon, 25), dtype=float)
    measurements = np.zeros((horizon, n_meas), dtype=float)
    masks = np.ones((horizon, n_meas), dtype=bool)
    controls = np.zeros((horizon, 6), dtype=float)
    envs = {
        "sun_vec_i": np.zeros((horizon, 3), dtype=float),
        "earth_vec_i": np.zeros((horizon, 3), dtype=float),
        "att_ref1_i": np.zeros((horizon, 3), dtype=float),
        "att_ref2_i": np.zeros((horizon, 3), dtype=float),
        "magnetic_field_i": np.zeros((horizon, 3), dtype=float),
        "weather_index": np.zeros(horizon, dtype=float),
        "rho_atm": np.zeros(horizon, dtype=float),
    }

    R = measurement_noise_cov(noise_cfg, include_range_doppler, star_tracker=bool(params.star_tracker))
    std = np.sqrt(np.diag(R))

    storm_steps = 0
    storm_torque_peak = 0.0
    storm_accel_peak = 0.0
    dropout_count = 0
    noise_scale_peak = 1.0

    steps = progress(enumerate(time), desc=f"  Propagating [{scenario}]",
                     total=horizon, leave=False, unit="step",
                     disable=not (show_progress and horizon >= PROGRESS_MIN_STEPS))
    for k, t in steps:
        env = environmental_fields(t, x, scenario, params)
        ctrl = control_profile(t, scenario)

        controls[k] = np.hstack([ctrl.torque_cmd, ctrl.accel_cmd])
        envs["sun_vec_i"][k] = env.sun_vec_i
        envs["earth_vec_i"][k] = env.earth_vec_i
        envs["att_ref1_i"][k] = env.att_ref1_i if env.att_ref1_i is not None else env.sun_vec_i
        envs["att_ref2_i"][k] = env.att_ref2_i if env.att_ref2_i is not None else env.earth_vec_i
        envs["magnetic_field_i"][k] = env.magnetic_field_i
        envs["weather_index"][k] = env.weather_index
        envs["rho_atm"][k] = env.rho_atm

        states[k] = x

        y = measurement_vector(x, env, params, include_range_doppler)
        noise = std * rng.standard_normal(y.shape[0])

        if scenario == "degraded_measurements":
            drop = rng.random(y.shape[0]) < 0.25
            masks[k, drop] = False
            dropout_count += int(drop.sum())

        if scenario == "biased_sensors":
            noise[:6] += np.array([0.008, -0.004, 0.006, 0.09, -0.05, 0.03], dtype=float)

        if scenario == "storm" and storm_noise:
            noise[:6] *= 1.8
            noise[0:3] += 1.2 * std[0:3] * np.sin(0.5 * t)
            noise[3:6] += 1.5 * std[3:6] * np.sin(0.3 * t)
            noise = noise * 1.5
            noise_scale_peak = max(noise_scale_peak, 1.5)

        measurements[k] = y + noise
        if not masks[k].all():
            measurements[k, ~masks[k]] = np.nan

        x = rk4_step(x, t, dt, ctrl, env, params)
        x[:4] = normalize_quaternion(x[:4])

        # Hidden disturbance random walk / storm excitation.
        x[19:22] += dt * (-0.005 * x[19:22] + 5e-4 * rng.standard_normal(3))
        x[22:25] += dt * (-0.005 * x[22:25] + 2e-4 * rng.standard_normal(3))

        if (scenario in {"storm", "high_disturbance"} and storm_noise
                and (20 < t < 40 or 100 < t < 130)):
            storm_steps += 1
            d_tau = np.array([3e-4, -4e-4, 2.5e-4], dtype=float)
            d_acc = np.array([4e-5, -3e-5, 2e-5], dtype=float)

            storm_torque_peak = max(storm_torque_peak, float(np.linalg.norm(d_tau)))
            storm_accel_peak = max(storm_accel_peak, float(np.linalg.norm(d_acc)))

            x[19:22] += d_tau
            x[22:25] += d_acc

        x[:4] = normalize_quaternion(x[:4])

    provenance = {
        "scenario": scenario,
        "seed": seed,
        "storm_steps": storm_steps,
        "storm_torque_peak": storm_torque_peak,
        "storm_accel_peak": storm_accel_peak,
        "dropout_count": dropout_count,
        "noise_scale_peak": noise_scale_peak,
        "epoch_utc": epoch_utc,
        "rho_atm_mean": float(np.mean(envs["rho_atm"])),
        "rho_atm_max": float(np.max(envs["rho_atm"])),
        "atmosphere_model": getattr(params.atmosphere, "name", "exponential"),
    }

    return Trajectory(
        scenario=scenario,
        seed=seed,
        time=time,
        states=states,
        measurements=measurements,
        measurement_mask=masks,
        controls=controls,
        env=envs,
        provenance=provenance,
        epoch_utc=epoch_utc,
    )


def generate_dataset(
    output_dir: Path,
    cfg: dict,
    split: str,
    n_traj: int,
    base_seed: int,
) -> list[Trajectory]:
    params = build_params(cfg)
    noise_cfg = cfg["measurement_noise"]
    dt = float(cfg["synthetic"]["dt"])
    horizon = int(cfg["synthetic"]["horizon"])
    scenarios = cfg["synthetic"]["scenarios"]
    include_range_doppler = bool(cfg["synthetic"]["include_range_doppler"])

    out: list[Trajectory] = []
    manifest_rows: list[dict[str, Any]] = []

    for i in progress(range(n_traj), desc=f"Generating trajectories [{split}]",
                      unit="traj"):
        scen = scenarios[i % len(scenarios)]
        traj = simulate_trajectory(
            seed=base_seed + 17 * i,
            scenario=scen,
            dt=dt,
            horizon=horizon,
            params=params,
            noise_cfg=noise_cfg,
            include_range_doppler=include_range_doppler,
            epoch_utc=cfg.get("simulation", {}).get("epoch_utc"),
        )
        out.append(traj)
        manifest_rows.append(traj.provenance)

    output_dir.mkdir(parents=True, exist_ok=True)
    with progress_bar(f"Saving dataset [{split}]", total=2, unit="file") as bar:
        np.savez_compressed(output_dir / f"{split}.npz",
                            trajectories=np.array(out, dtype=object))
        bar.update(1)
        pd.DataFrame(manifest_rows).to_csv(output_dir / f"{split}_manifest.csv",
                                           index=False)
        bar.update(1)

    return out