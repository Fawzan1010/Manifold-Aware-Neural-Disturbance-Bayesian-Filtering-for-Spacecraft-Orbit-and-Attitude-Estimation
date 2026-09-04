from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def finite_difference_jacobian(func, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(func(x), dtype=float).reshape(-1)
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps * (1.0 + abs(x[i]))
        fp = np.asarray(func(x + dx), dtype=float).reshape(-1)
        fm = np.asarray(func(x - dx), dtype=float).reshape(-1)
        J[:, i] = (fp - fm) / (2.0 * dx[i])
    return J


def safe_cholesky(P: np.ndarray, jitter: float = 1e-9) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    for k in range(8):
        try:
            return np.linalg.cholesky(P + (jitter * (10 ** k)) * np.eye(P.shape[0]))
        except np.linalg.LinAlgError:
            continue
    vals, vecs = np.linalg.eigh(P)
    vals = np.clip(vals, 1e-12, None)
    return vecs @ np.diag(np.sqrt(vals))


def mahalanobis(v: np.ndarray, S: np.ndarray) -> float:
    v = np.asarray(v).reshape(-1)
    return float(v.T @ np.linalg.solve(S, v))
