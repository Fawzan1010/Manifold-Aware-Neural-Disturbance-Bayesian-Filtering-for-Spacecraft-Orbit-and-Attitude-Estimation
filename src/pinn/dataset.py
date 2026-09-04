from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset

from src.dynamics.simulator import Trajectory
from src.utils.progress import progress


@dataclass
class Sample:
    x: np.ndarray
    y: np.ndarray
    target: np.ndarray
    next_state: np.ndarray


class ResidualDataset(Dataset):
    def __init__(self, trajectories: list[Trajectory], window: int, include_range_doppler: bool = True) -> None:
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for traj in progress(trajectories, desc="  Building residual windows",
                             leave=False, unit="traj"):
            meas = np.nan_to_num(traj.measurements, nan=0.0)
            for k in range(window, len(traj.time) - 1):
                hist = meas[k-window:k]
                cur = traj.states[k]
                ctrl = traj.controls[k]
                env = np.hstack([
                    traj.env['sun_vec_i'][k], traj.env['earth_vec_i'][k], traj.env['magnetic_field_i'][k], [traj.env['weather_index'][k]],
                ])
                x = np.hstack([cur, ctrl, env, hist.reshape(-1)])
                target = np.hstack([cur[19:22], cur[22:25]])
                self.samples.append((x.astype(np.float32), hist.astype(np.float32), target.astype(np.float32), traj.states[k+1].astype(np.float32)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, hist, target, nxt = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(hist), torch.from_numpy(target), torch.from_numpy(nxt)
