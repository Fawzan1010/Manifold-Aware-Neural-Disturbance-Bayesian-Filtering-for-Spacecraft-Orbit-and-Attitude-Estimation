from __future__ import annotations

import numpy as np

from src.filters.base import BaseFilter, EstimationStep
from src.dynamics.spacecraft import rk4_step
from src.sensors.measurement_models import measurement_vector
from src.utils.math_utils import finite_difference_jacobian
from src.utils.quaternion import (
    normalize_quaternion,
    quat_conjugate,
    quat_from_rotvec,
    quat_multiply,
    rotvec_from_quat,
)
from src.utils.state import STATE_DIM


class MEKF(BaseFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_dim = STATE_DIM - 1
        self.P = self.P[: self.error_dim, : self.error_dim]

    def inject(self, x: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Inject error-state delta into nominal state."""
        out = x.copy()

        rotvec = np.asarray(delta[:3], dtype=float)
        rot_norm = np.linalg.norm(rotvec)
        if rot_norm > 0.5:
            rotvec = rotvec * (0.5 / rot_norm)

        q = quat_multiply(quat_from_rotvec(rotvec), x[:4])
        out[:4] = normalize_quaternion(q)
        out[4:] = x[4:] + delta[3:]
        return out

    def _error_state(self, x_ref: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        """Return MEKF error-state between reference and test states."""
        dq = quat_multiply(x_test[:4], quat_conjugate(x_ref[:4]))
        rot_err = rotvec_from_quat(dq)
        other_err = x_test[4:] - x_ref[4:]
        return np.hstack([rot_err, other_err])

    def _propagate_error_jacobian(
        self,
        x_pred: np.ndarray,
        t: float,
        dt: float,
        control,
        env,
    ) -> np.ndarray:
        """Numerically approximate local error-state transition Jacobian."""
        zero = np.zeros(self.error_dim, dtype=float)

        def propagate(delta: np.ndarray) -> np.ndarray:
            x_pert = self.inject(x_pred, delta)
            x_prop = rk4_step(x_pert, t, dt, control, env, self.params)
            x_prop[:4] = normalize_quaternion(x_prop[:4])
            return self._error_state(x_pred, x_prop)

        F = finite_difference_jacobian(propagate, zero)
        return F

    def step(
        self,
        t: float,
        dt: float,
        env,
        control,
        y: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> EstimationStep:
        x_nom = self.x.copy()

        # Nominal propagation
        x_pred = rk4_step(x_nom, t, dt, control, env, self.params)
        x_pred[:4] = normalize_quaternion(x_pred[:4])

        # Local error-state propagation
        F = self._propagate_error_jacobian(x_nom, t, dt, control, env)
        Qe = self.Q[: self.error_dim, : self.error_dim]
        P_pred = F @ self.P @ F.T + Qe
        P_pred = 0.5 * (P_pred + P_pred.T)

        y_pred = measurement_vector(x_pred, env, self.params, self.include_range_doppler)
        if mask is None:
            mask = np.ones_like(y_pred, dtype=bool)

        idx = np.where(mask)[0]

        def h(delta: np.ndarray) -> np.ndarray:
            x_test = self.inject(x_pred, delta)
            return measurement_vector(x_test, env, self.params, self.include_range_doppler)[idx]

        H = finite_difference_jacobian(h, np.zeros(self.error_dim, dtype=float))
        Rm = self.R[np.ix_(idx, idx)]
        S = H @ P_pred @ H.T + Rm
        S = 0.5 * (S + S.T)

        # Robust solve instead of explicit inverse
        K = np.linalg.solve(S.T, (P_pred @ H.T).T).T

        innov = y[idx] - y_pred[idx]
        delta = K @ innov

        # Inject correction and renormalize quaternion
        self.x = self.inject(x_pred, delta)
        self.x[:4] = normalize_quaternion(self.x[:4])

        # Joseph stabilized covariance update
        I = np.eye(self.error_dim)
        self.P = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ Rm @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return EstimationStep(self.x.copy(), self.P.copy(), y_pred[idx], innov, S)