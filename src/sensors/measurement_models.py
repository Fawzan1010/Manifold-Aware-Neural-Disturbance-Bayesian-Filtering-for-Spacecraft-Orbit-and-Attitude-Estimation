from __future__ import annotations

import numpy as np

from src.dynamics.spacecraft import Environment, SpacecraftParams
from src.utils.quaternion import quat_to_dcm
from src.utils.state import unpack_state

MEAS_DIM = 18


def measurement_vector(x: np.ndarray, env: Environment, params: SpacecraftParams, include_range_doppler: bool = True) -> np.ndarray:
    d = unpack_state(x)
    q = d['q']
    w = d['w']
    r = d['r']
    v = d['v']
    bg = d['bg']
    ba = d['ba']
    td = d['td']
    ad = d['ad']
    Cbi = quat_to_dcm(q)
    a_body = Cbi @ (ad + np.array([0.0, 0.0, 0.0])) + ba
    att1 = getattr(env, 'att_ref1_i', None)
    att2 = getattr(env, 'att_ref2_i', None)
    if att1 is None:
        att1 = env.sun_vec_i
    if att2 is None:
        att2 = env.earth_vec_i
    y = np.hstack([
        w + bg,
        a_body,
        Cbi @ att1,
        Cbi @ att2,
        r,
        v,
    ])
    if include_range_doppler:
        station = np.array([6378.137 + 0.0, 0.0, 0.0])
        rho = np.linalg.norm(r - station)
        drho = np.dot((r - station) / (rho + 1e-12), v)
        y = np.hstack([y, rho, drho])
    return y


def measurement_dim(include_range_doppler: bool = True) -> int:
    return 20 if include_range_doppler else 18


def measurement_noise_cov(noise_cfg: dict, include_range_doppler: bool = True, star_tracker: bool = False) -> np.ndarray:
    vec_sigma = noise_cfg.get('vector_st', noise_cfg['vector']) if star_tracker else noise_cfg['vector']
    diag = [
        noise_cfg['gyro']] * 3 + [noise_cfg['accel']] * 3 + [vec_sigma] * 6 + [noise_cfg['gps_pos']] * 3 + [noise_cfg['gps_vel']] * 3
    if include_range_doppler:
        diag += [noise_cfg['range'], noise_cfg['doppler']]
    return np.diag(np.square(diag))


def split_measurements(y: np.ndarray, include_range_doppler: bool = True):
    parts = {'gyro': y[:3], 'accel': y[3:6], 'sun': y[6:9], 'earth': y[9:12], 'pos': y[12:15], 'vel': y[15:18]}
    if include_range_doppler:
        parts['range'] = y[18]
        parts['doppler'] = y[19]
    return parts
