import csv
from collections import defaultdict
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

METHOD_ORDER = ["EKF","Adaptive-EKF","UKF","MEKF",
                "PINN+EKF","PINN+UKF","PINN+MEKF","Transformer+MEKF","PINN+UKF+MEKF"]
DISPLAY_NAME = {
    "EKF":"EKF","Adaptive-EKF":"Adaptive-EKF","UKF":"UKF","MEKF":"MEKF",
    "PINN-only":"NDRN-only","Transformer-only":"Transformer-only",
    "PINN+EKF":"NDRN+EKF","PINN+UKF":"NDRN+UKF","PINN+MEKF":"NDRN+MEKF",
    "Transformer+MEKF":"Transformer+MEKF","PINN+UKF+MEKF":"Proposed",
}

rows = list(csv.DictReader(open(f"{BASE}/outputs/tables/metrics_per_trial.csv")))
by_method = defaultdict(list)
for r in rows:
    by_method[r["method"]].append(r)

def boxplot_for(col, ylabel, fname, ylim=None):
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    data = [np.array([float(r[col]) for r in by_method[m]]) for m in METHOD_ORDER]
    bp = ax.boxplot(data, labels=[DISPLAY_NAME[m] for m in METHOD_ORDER],
                     showfliers=True, patch_artist=True,
                     boxprops=dict(facecolor="#4C72B0", alpha=0.55, linewidth=0.8),
                     medianprops=dict(color="#C44E52", linewidth=1.3),
                     whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
                     flierprops=dict(marker="o", markersize=2.5, alpha=0.4))
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/{fname}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)

# Attitude: clip y-range to keep filter-based methods legible (as original did with an implicit crop)
boxplot_for("attitude_geodesic_rmse", "Attitude geodesic RMSE [rad]", "attitude_rmse_boxplot.pdf")
# Position: log-scale needed since learning-only baselines are 3+ orders of magnitude off
fig, ax = plt.subplots(figsize=(6.6, 3.2))
data = [np.array([float(r["position_rmse"]) for r in by_method[m]]) for m in METHOD_ORDER]
ax.boxplot(data, labels=[DISPLAY_NAME[m] for m in METHOD_ORDER], showfliers=True, patch_artist=True,
           boxprops=dict(facecolor="#DD8452", alpha=0.55, linewidth=0.8),
           medianprops=dict(color="#C44E52", linewidth=1.3),
           whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
           flierprops=dict(marker="o", markersize=2.5, alpha=0.4))
ax.set_yscale("log")
ax.set_ylabel("Position RMSE [km] (log scale)")
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/position_rmse_boxplot.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote position_rmse_boxplot.pdf")
