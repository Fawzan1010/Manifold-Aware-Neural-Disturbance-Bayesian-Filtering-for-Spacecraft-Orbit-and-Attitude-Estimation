import csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "."
FIGDIR = f"{BASE}/outputs/figures"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

METHOD_ORDER = ["EKF","Adaptive-EKF","UKF","MEKF","PINN+EKF","PINN+UKF","PINN+MEKF","Transformer+MEKF","PINN+UKF+MEKF"]
DISPLAY_NAME = {
    "EKF":"EKF","Adaptive-EKF":"Adaptive-EKF","UKF":"UKF","MEKF":"MEKF",
    "PINN+EKF":"NDRN+EKF","PINN+UKF":"NDRN+UKF","PINN+MEKF":"NDRN+MEKF",
    "Transformer+MEKF":"Transformer+MEKF","PINN+UKF+MEKF":"Proposed",
}

# ---------- 1. grouped_bar_comparison.png ----------
rows = list(csv.DictReader(open(f"{BASE}/outputs/tables/metrics_summary.csv")))
by_method = {r["method"]: r for r in rows}
channels = [
    ("attitude_geodesic_rmse", "Attitude"),
    ("position_rmse", "Position"),
    ("velocity_rmse", "Velocity"),
    ("angular_rate_rmse", "Angular rate"),
]
data = {ch: np.array([float(by_method[m][col]) for m in METHOD_ORDER]) for col, ch in channels}
norm = {ch: data[ch] / data[ch].max() for ch in data}

fig, ax = plt.subplots(figsize=(7.6, 4.0))
n_methods = len(METHOD_ORDER)
n_ch = len(channels)
width = 0.8 / n_ch
x = np.arange(n_methods)
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for i, (col, ch) in enumerate(channels):
    ax.bar(x + i * width - 0.4 + width/2, norm[ch], width=width, label=ch, color=colors[i])
ax.set_xticks(x)
ax.set_xticklabels([DISPLAY_NAME[m] for m in METHOD_ORDER], rotation=35, ha="right")
ax.set_ylabel("Normalized error (lower is better)")
ax.set_ylim(0, 1.15)
ax.legend(frameon=False, ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.14))
fig.tight_layout()
fig.savefig(f"{FIGDIR}/grouped_bar_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote grouped_bar_comparison.png")

# ---------- 2. ablation_pinn_r_scale.pdf ----------
arows = list(csv.DictReader(open(f"{BASE}/outputs/tables/ablation_summary.csv")))
sub = [r for r in arows if r["axis"] == "pinn_r_scale" and r["method"] == "PINN+UKF+MEKF"]
sub.sort(key=lambda r: float(r["value"]))
rscale = [float(r["value"]) for r in sub]
att = [float(r["attitude_geodesic_rmse_mean"]) for r in sub]
att_std = [float(r["attitude_geodesic_rmse_std"]) for r in sub]
pos = [float(r["position_rmse_mean"]) for r in sub]
pos_std = [float(r["position_rmse_std"]) for r in sub]

fig, ax1 = plt.subplots(figsize=(4.8, 3.6))
color1 = "#4C72B0"
ax1.errorbar(rscale, att, yerr=att_std, marker="o", color=color1, capsize=3, label="Attitude RMSE")
ax1.set_xscale("log")
ax1.set_xlabel(r"$r_{\mathrm{scale}}$ (pseudo-measurement inverse trust)")
ax1.set_ylabel("Attitude RMSE [rad]", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_xticks(rscale)
ax1.set_xticklabels([str(v) for v in rscale])

ax2 = ax1.twinx()
color2 = "#DD8452"
ax2.errorbar(rscale, pos, yerr=pos_std, marker="s", color=color2, capsize=3, label="Position RMSE")
ax2.set_ylabel("Position RMSE [km]", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)
ax2.grid(False)

fig.tight_layout()
fig.savefig(f"{FIGDIR}/ablation_pinn_r_scale.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote ablation_pinn_r_scale.pdf")
