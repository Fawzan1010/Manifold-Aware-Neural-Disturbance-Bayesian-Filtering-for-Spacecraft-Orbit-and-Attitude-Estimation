from src.utils.quaternion import normalize_quaternion, quat_multiply, quat_geodesic_distance
import numpy as np


def test_quaternion_norm():
    q = normalize_quaternion(np.array([2.0, 0.0, 0.0, 0.0]))
    assert np.isclose(np.linalg.norm(q), 1.0)


def test_geodesic_zero():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.isclose(quat_geodesic_distance(q, q), 0.0)
