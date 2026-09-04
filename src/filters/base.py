from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.dynamics.spacecraft import SpacecraftParams
from src.sensors.measurement_models import measurement_vector
from src.utils.quaternion import normalize_quaternion
from src.utils.state import pack_state, unpack_state


@dataclass
class EstimationStep:
    x: np.ndarray
    P: np.ndarray
    y_pred: np.ndarray
    innovation: np.ndarray
    S: np.ndarray


class BaseFilter:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, params: SpacecraftParams, Q: np.ndarray, R: np.ndarray, include_range_doppler: bool = True):
        self.x = x0.astype(float).copy()
        self.x[:4] = normalize_quaternion(self.x[:4])
        self.P = P0.astype(float).copy()
        self.params = params
        self.Q = Q.astype(float).copy()
        self.R = R.astype(float).copy()
        self.include_range_doppler = include_range_doppler

    def step(self, *args, **kwargs) -> EstimationStep:
        raise NotImplementedError

    def predict_measurement(self, x: np.ndarray, env) -> np.ndarray:
        return measurement_vector(x, env, self.params, self.include_range_doppler)
