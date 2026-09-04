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

# ---------- 3. disturbance_tracking.png ----------
ekf = np.load(f"{PRED}/EKF.npz")
ndrn = np.load(f"{PRED}/PINN_only.npz")

t = ekf["time"]
axis = 0  # single illustrative axis
truth_tau = ekf["truth"][:, 19 + axis]
ekf_est_tau = ekf["est"][:, 19 + axis]
ndrn_est_tau = ndrn["est"][:, 19 + axis]

# plot a representative window (first 2000 s) for legibility
w = slice(0, 2000)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.plot(t[w], truth_tau[w], color="k", lw=1.1, label="True disturbance torque")
ax.plot(t[w], ndrn_est_tau[w], color="#4C72B0", lw=0.9, alpha=0.9, label="NDRN estimate")
ax.plot(t[w], ekf_est_tau[w], color="#C44E52", lw=0.7, alpha=0.8, label="EKF (Gauss-Markov) estimate")
ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$\tau_{d,x}$ [N$\cdot$m]")
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/disturbance_tracking.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote disturbance_tracking.png")

# ---------- 4. nees_consistency.png ----------
methods = ["EKF", "Adaptive_EKF", "MEKF", "PINN_UKF_MEKF"]
labels = {"EKF": "EKF", "Adaptive_EKF": "Adaptive-EKF", "MEKF": "MEKF", "PINN_UKF_MEKF": "Proposed"}
colors = {"EKF": "#C44E52", "Adaptive_EKF": "#DD8452", "MEKF": "#55A868", "PINN_UKF_MEKF": "#4C72B0"}

fig, ax = plt.subplots(figsize=(6.4, 3.2))
for m in methods:
    d = np.load(f"{PRED}/{m}.npz")
    nees_n = d["nees"] / 24.0
    t = d["time"]
    ax.plot(t, nees_n, color=colors[m], lw=0.8, alpha=0.85, label=labels[m])
ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="Ideal (=1)")
ax.set_yscale("log")
ax.set_xlabel("Time [s]")
ax.set_ylabel(r"Normalized full-state NEES ($n_x=24$)")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
fig.tight_layout()
fig.savefig(f"{FIGDIR}/nees_consistency.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote nees_consistency.png")
