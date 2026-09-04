from __future__ import annotations

import numpy as np

from src.filters.base import BaseFilter, EstimationStep
from src.dynamics.spacecraft import rk4_step, control_profile
from src.utils.math_utils import finite_difference_jacobian
from src.utils.quaternion import normalize_quaternion
from src.sensors.measurement_models import measurement_vector
from src.utils.state import unpack_state


class AdaptiveEKF(BaseFilter):
    def step(self, t: float, dt: float, env, control, y: np.ndarray, mask: np.ndarray | None = None) -> EstimationStep:
        def f(xx):
            return rk4_step(xx, t, dt, control, env, self.params)
        x_pred = f(self.x)
        F = finite_difference_jacobian(f, self.x)
        # Default prediction using current Q
        P_pred = F @ self.P @ F.T + self.Q
        y_pred = measurement_vector(x_pred, env, self.params, self.include_range_doppler)
        if mask is None:
            mask = np.ones_like(y_pred, dtype=bool)
        idx = np.where(mask)[0]
        y_m = y[idx]
        yhat_m = y_pred[idx]
        def h(xx):
            return measurement_vector(xx, env, self.params, self.include_range_doppler)[idx]
        H = finite_difference_jacobian(h, x_pred)
        S = H @ P_pred @ H.T + self.R[np.ix_(idx, idx)]
        K = P_pred @ H.T @ np.linalg.inv(S)
        innov = y_m - yhat_m
        # ---------- Adaptive process-noise update ----------
        nis = float(innov.T @ np.linalg.inv(S) @ innov)
        m = len(innov)
        alpha = np.clip(nis / m, 1.0, 5.0)
        forget = 0.98
        self.Q = forget * self.Q + (1.0 - forget) * alpha * self.Q
        # keep Q symmetric
        self.Q = 0.5 * (self.Q + self.Q.T)
        # small numerical regularization
        self.Q += 1e-10 * np.eye(self.Q.shape[0])
        # -----------------------------------------------
        self.x = x_pred + K @ innov
        self.x[:4] = normalize_quaternion(self.x[:4])
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ self.R[np.ix_(idx, idx)] @ K.T
        return EstimationStep(self.x.copy(), self.P.copy(), y_pred, innov, S)
