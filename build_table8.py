"""Rebuild the storm Table 8 restricted to arcs every method survived.

Usage:
    python build_table8.py outputs/storm                       # single event
    python build_table8.py outputs/storm_17mar:17Mar \
                            outputs/storm_22jun:22Jun \
                            outputs/storm_20dec:20Dec           # multi-event

Each positional arg is a storm output directory (containing
tables/storm_metrics_per_trial.csv), optionally suffixed with :LABEL to tag
the event. Arcs are identified as (event, trial). For each pairing
(mismatched/matched) separately, a method "survives" an arc if diverged is
False for that (method, event, trial). The common-survivor set for a pairing
is the intersection over all methods. Table entries average only over that
common set; arc counts and per-method divergence counts (out of the full
arc set for that pairing) are reported alongside.
"""
import sys
import pandas as pd
import numpy as np

KEY_METRICS = ["attitude_geodesic_rmse", "angular_rate_rmse", "nees_normalized"]

def load(args):
    frames = []
    for a in args:
        if ":" in a:
            path, label = a.split(":", 1)
        else:
            path, label = a, path if False else a
        df = pd.read_csv(f"{path}/tables/storm_metrics_per_trial.csv")
        df["event"] = label
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["arc_id"] = out["event"].astype(str) + "_" + out["trial"].astype(str)
    return out

def main():
    args = sys.argv[1:]
    if not args:
        args = ["outputs/storm:17Mar"]
    df = load(args)
    methods = df["method"].unique().tolist()

    for pairing in ["mismatched", "matched"]:
        sub = df[df["pairing"] == pairing]
        all_arcs = sorted(sub["arc_id"].unique())
        n_total = len(all_arcs)

        survived_by_method = {}
        for m in methods:
            msub = sub[sub["method"] == m]
            survived = set(msub.loc[~msub["diverged"].astype(bool), "arc_id"])
            survived_by_method[m] = survived

        common = set(all_arcs)
        for m in methods:
            common &= survived_by_method[m]
        common = sorted(common)

        print(f"\n=== Pairing: {pairing} | total arcs: {n_total} | common-survived arcs: {len(common)} ===")
        print("arcs:", all_arcs)
        print("common survivor set:", common)

        rows = []
        for m in methods:
            msub = sub[(sub["method"] == m) & (sub["arc_id"].isin(common))]
            n_div_total = int(sub[sub["method"] == m]["diverged"].astype(bool).sum())
            row = {"method": m, "n_arcs_common": len(msub),
                   "n_diverged_of_total": f"{n_div_total}/{n_total}"}
            for k in KEY_METRICS:
                row[k] = float(msub[k].mean()) if len(msub) else float("nan")
            rows.append(row)
        out = pd.DataFrame(rows)
        pd.set_option("display.width", 160)
        print(out.to_string(index=False))

if __name__ == "__main__":
    main()
