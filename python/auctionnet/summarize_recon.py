"""Summarize a reconciliation batch (job_recon_role28.sh outputs under results/designA/recon_v1/role<r>/).

Writes summary_P.csv (interval, width, truth coverage by run x rule x N, with width ratios rectangle/budget/tail) and
summary_O.csv (share, lognormal-truth and empirical-mass feasibility, removal relative to the bids-only region) and prints both.
Usage: python summarize_recon.py <recon_role_dir>
"""
import glob
import os
import sys

import pandas as pd

d = sys.argv[1]
P, O = [], []
for f in sorted(glob.glob(os.path.join(d, "*_regimeP.csv"))):
    tag = os.path.basename(f).replace("_regimeP.csv", "")
    p = pd.read_csv(f); p.insert(0, "run", tag); P.append(p)
for f in sorted(glob.glob(os.path.join(d, "*_regimeO.csv"))):
    tag = os.path.basename(f).replace("_regimeO.csv", "")
    o = pd.read_csv(f); o.insert(0, "run", tag); O.append(o)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 40)
if P:
    P = pd.concat(P)
    pw = P.pivot_table(index=["run", "N"], columns="rule", values=["lam_lo", "lam_hi", "width", "contains_truth"])
    pw.columns = [f"{a}_{b}" for a, b in pw.columns]; pw = pw.reset_index()
    for a, b in [("momentF", "jointF"), ("momentF", "tailF"), ("jointF", "tailF"), ("momentB", "jointB"), ("tailB", "tailBc"), ("momentB", "momentBc")]:
        if f"width_{a}" in pw and f"width_{b}" in pw:
            pw[f"ratio_{a}_over_{b}"] = pw[f"width_{a}"] / pw[f"width_{b}"]
    pw.to_csv(os.path.join(d, "summary_P.csv"), index=False)
    cols = ["run", "N"] + [c for c in pw.columns if c.startswith("width_") or c.startswith("ratio_") or c.startswith("contains_")]
    print("Regime P:\n", pw[cols].round(2).to_string(index=False))
if O:
    O = pd.concat(O)
    ow = O.pivot_table(index=["run", "N"], columns="rule", values=["share", "truth_feasible", "emp_feasible"])
    ow.columns = [f"{a}_{b}" for a, b in ow.columns]; ow = ow.reset_index()
    for r in ["momentF", "jointF", "tailF", "tailF_own", "momentB", "jointB", "tailB", "tailBc", "momentBc", "tailB_own"]:
        if f"share_{r}" in ow and "share_none" in ow:
            ow[f"removed_by_{r}"] = 1.0 - ow[f"share_{r}"] / ow["share_none"]
    ow.to_csv(os.path.join(d, "summary_O.csv"), index=False)
    cols = ["run", "N"] + [c for c in ow.columns if c.startswith("share_") or c.startswith("removed_") or c.startswith("truth_") or c.startswith("emp_")]
    print("\nRegime O:\n", ow[cols].round(3).to_string(index=False))
