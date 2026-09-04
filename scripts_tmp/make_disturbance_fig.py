import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "."
FIGDIR = f"{BASE}/outputs/figures"
PRED = f"{BASE}/outputs/storm_17mar/matched/predictions"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

ekf = np.load(f"{PRED}/EKF.npz")
prop = np.load(f"{PRED}/PINN_UKF_MEKF.npz")

t = ekf["time"]
axis = 0
truth_tau = ekf["truth"][:, 19 + axis]
ekf_est_tau = ekf["est"][:, 19 + axis]
prop_est_tau = prop["est"][:, 19 + axis]

w = slice(0, 2000)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.plot(t[w], truth_tau[w], color="k", lw=1.1, label="True disturbance torque")
ax.plot(t[w], prop_est_tau[w], color="#4C72B0", lw=0.9, alpha=0.9, label="Proposed (fused) estimate")
ax.plot(t[w], ekf_est_tau[w], color="#C44E52", lw=0.7, alpha=0.8, label="EKF (Gauss-Markov) estimate")
ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$\tau_{d,x}$ [N$\cdot$m]")
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/disturbance_tracking.png", dpi=300, bbox_inches="tight")
plt.close(fig)

rmse_prop = np.sqrt(np.mean((truth_tau - prop_est_tau) ** 2))
rmse_ekf = np.sqrt(np.mean((truth_tau - ekf_est_tau) ** 2))
corr_prop = np.corrcoef(truth_tau, prop_est_tau)[0, 1]
corr_ekf = np.corrcoef(truth_tau, ekf_est_tau)[0, 1]
print("rmse_prop", rmse_prop, "rmse_ekf", rmse_ekf, "corr_prop", corr_prop, "corr_ekf", corr_ekf)
print("wrote disturbance_tracking.png")
