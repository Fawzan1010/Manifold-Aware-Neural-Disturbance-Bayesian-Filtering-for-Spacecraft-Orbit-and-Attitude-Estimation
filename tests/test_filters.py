from src.utils.reproducibility import set_seed
from src.dynamics.spacecraft import SpacecraftParams
from src.dynamics.simulator import simulate_trajectory
from src.sensors.measurement_models import measurement_noise_cov
from src.filters.ekf import EKF
import numpy as np


def test_ekf_smoke():
    set_seed(1)
    params = SpacecraftParams()
    cfg = {'gyro':0.001,'accel':0.01,'vector':0.01,'gps_pos':0.02,'gps_vel':0.01,'range':0.02,'doppler':0.002}
    traj = simulate_trajectory(1, 'quiet', 1.0, 10, params, cfg, True)
    R = measurement_noise_cov(cfg, True)
    Q = np.eye(25) * 1e-5
    ekf = EKF(traj.states[0].copy(), np.eye(25)*1e-2, params, Q, R)
    env = type('Env', (), {'sun_vec_i': traj.env['sun_vec_i'][0], 'earth_vec_i': traj.env['earth_vec_i'][0], 'magnetic_field_i': traj.env['magnetic_field_i'][0], 'weather_index': float(traj.env['weather_index'][0])})
    ctrl = type('Ctrl', (), {'torque_cmd': traj.controls[0,:3], 'accel_cmd': traj.controls[0,3:]})
    step = ekf.step(0.0, 1.0, env, ctrl, traj.measurements[0], traj.measurement_mask[0])
    assert step.x.shape[0] == 25
