from __future__ import annotations

import numpy as np

from src.filters.base import BaseFilter, EstimationStep
from src.dynamics.spacecraft import rk4_step
from src.sensors.measurement_models import measurement_vector
from src.utils.math_utils import safe_cholesky
from src.utils.quaternion import normalize_quaternion, quat_average


class UKF(BaseFilter):
    def __init__(self, *args, alpha: float = 0.25, beta: float = 2.0, kappa: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha; self.beta = beta; self.kappa = kappa

    def sigma_points(self, x, P):
        n = x.size
        lam = self.alpha**2 * (n + self.kappa) - n
        S = safe_cholesky((n + lam) * P)
        pts = [x]
        for i in range(n):
            pts.append(x + S[:, i])
            pts.append(x - S[:, i])
        Wm = np.full(2*n + 1, 1.0 / (2*(n + lam)))
        Wc = Wm.copy()
        Wm[0] = lam / (n + lam)
        Wc[0] = Wm[0] + (1 - self.alpha**2 + self.beta)
        return np.array(pts), Wm, Wc

    def step(self, t: float, dt: float, env, control, y: np.ndarray, mask: np.ndarray | None = None) -> EstimationStep:
        x = self.x.copy()
        P = self.P.copy()
        n = x.size
        sig, Wm, Wc = self.sigma_points(x, P)
        prop = []
        for s in sig:
            sp = rk4_step(s, t, dt, control, env, self.params)
            sp[:4] = normalize_quaternion(sp[:4])
            prop.append(sp)
        prop = np.array(prop)
        q_mean = quat_average(prop[:, :4], Wm)
        x_pred = np.average(prop, axis=0, weights=Wm)
        x_pred[:4] = q_mean
        X = prop - x_pred
        X[:, :4] = np.array([np.hstack([normalize_quaternion(p[:4]) - q_mean]) for p in prop])
        P_pred = self.Q.copy()
        for i in range(prop.shape[0]):
            P_pred += Wc[i] * np.outer(X[i], X[i])
        ysig = np.array([measurement_vector(s, env, self.params, self.include_range_doppler) for s in prop])
        y_pred = np.average(ysig, axis=0, weights=Wm)
        if mask is None:
            mask = np.ones_like(y_pred, dtype=bool)
        idx = np.where(mask)[0]
        ysig = ysig[:, idx]
        y_pred = y_pred[idx]
        Y = ysig - y_pred
        S = self.R[np.ix_(idx, idx)].copy()
        C = np.zeros((n, len(idx)))
        for i in range(sig.shape[0]):
            S += Wc[i] * np.outer(Y[i], Y[i])
            C += Wc[i] * np.outer(X[i], Y[i])
        K = C @ np.linalg.inv(S)
        innov = y[idx] - y_pred
        self.x = x_pred + K @ innov
        self.x[:4] = normalize_quaternion(self.x[:4])
        self.P = P_pred - K @ S @ K.T
        return EstimationStep(self.x.copy(), self.P.copy(), y_pred, innov, S)
