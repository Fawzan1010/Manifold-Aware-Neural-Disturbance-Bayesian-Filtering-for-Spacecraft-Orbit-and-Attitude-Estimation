from __future__ import annotations

import numpy as np


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n <= 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = q / n
    if q[0] < 0:
        q = -q
    return q


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = normalize_quaternion(q1)
    w2, x2, y2, z2 = normalize_quaternion(q2)
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def quat_from_rotvec(rv: np.ndarray) -> np.ndarray:
    rv = np.asarray(rv, dtype=float).reshape(3)
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return normalize_quaternion(np.array([1.0, 0.5*rv[0], 0.5*rv[1], 0.5*rv[2]]))
    axis = rv / th
    half = 0.5 * th
    return normalize_quaternion(np.hstack([np.cos(half), axis * np.sin(half)]))


def rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    if q[0] < 0:
        q = -q
    v = q[1:]
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return 2.0 * v
    return 2.0 * np.arctan2(nv, q[0]) * v / nv


def quat_geodesic_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    d = abs(np.dot(q1, q2))
    d = float(np.clip(d, -1.0, 1.0))
    return 2.0 * np.arccos(d)


def quat_average(quats: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    qs = np.asarray(quats, dtype=float)
    if weights is None:
        weights = np.ones(len(qs)) / len(qs)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
    ref = qs[0]
    A = np.zeros((4, 4))
    for q, w in zip(qs, weights):
        q = normalize_quaternion(q)
        if np.dot(q, ref) < 0:
            q = -q
        A += w * np.outer(q, q)
    vals, vecs = np.linalg.eigh(A)
    q = vecs[:, np.argmax(vals)]
    return normalize_quaternion(q)
