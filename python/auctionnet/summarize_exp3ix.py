"""Summarize the EXP3-IX runs (run_exp3ix.py) for all roles: the certified bandit learner under tailB and momentB.

Reads results/designA/exp3ix_v1/role<r>/ and writes to results/designA/exp3ix_v1/:
  summary_P.csv         Regime P: interval, width and truth coverage by role, N and rule (tailB, momentB, empirical);
  summary_O.csv         Regime O: share of the grid and truth feasibility by role, N and rule (none, tailB, momentB),
                        with the fraction of the bids-only region removed;
  summary_coverage.csv  20 coverage seeds (coarse grid): truth coverage in P and O by rule, median width and share.
Also prints, for reference, the Hedge tailF widths from tailF_v1 (same roles and N) when present.
Usage: python summarize_exp3ix.py <designA_dir>
"""
import glob
import os
import sys

import pandas as pd

da = sys.argv[1]; v1 = os.path.join(da, "exp3ix_v1")
P, O, covP, covO = [], [], [], []
for rdir in sorted(glob.glob(os.path.join(v1, "role*"))):
    role = int(os.path.basename(rdir)[4:])
    p = pd.read_csv(os.path.join(rdir, "exp3ix_1e7_regimeP.csv")); p.insert(0, "role", role); P.append(p)
    o = pd.read_csv(os.path.join(rdir, "exp3ix_1e7_regimeO.csv")); o.insert(0, "role", role); O.append(o)
    cp = os.path.join(rdir, "coverage_exp3ix_1e7", "coverage_exp3ix_P.csv")
    if os.path.exists(cp):
        d = pd.read_csv(cp); d.insert(0, "role", role); covP.append(d)
        d = pd.read_csv(cp.replace("_P.csv", "_O.csv")); d.insert(0, "role", role); covO.append(d)
P = pd.concat(P); O = pd.concat(O)
pw = P[P.rule.isin(["tailB", "momentB", "empirical"])].pivot_table(index=["role", "N"], columns="rule", values=["lam_lo", "lam_hi", "width", "contains_truth"])
pw.columns = [f"{a}_{b}" for a, b in pw.columns]; pw = pw.reset_index()
pw.to_csv(os.path.join(v1, "summary_P.csv"), index=False)
ow = O[O.rule.isin(["none", "tailB", "momentB"])].pivot_table(index=["role", "N"], columns="rule", values=["share", "truth_feasible"])
ow.columns = [f"{a}_{b}" for a, b in ow.columns]; ow = ow.reset_index()
for rule in ["tailB", "momentB"]:
    if f"share_{rule}" in ow:
        ow[f"removed_by_{rule}"] = 1.0 - ow[f"share_{rule}"] / ow["share_none"]
ow.to_csv(os.path.join(v1, "summary_O.csv"), index=False)
pd.set_option("display.width", 240); pd.set_option("display.max_rows", 400)
print("Regime P (EXP3-IX to 1e7), by role and N:\n", pw.round(2).to_string())
print("\nRegime O (EXP3-IX to 1e7), by role and N:\n", ow.round(3).to_string())
tf = sorted(glob.glob(os.path.join(da, "tailF_v1", "role*", "hedge_1e8_regimeP.csv")))
if tf:
    rows = []
    for f in tf:
        role = int(os.path.basename(os.path.dirname(f))[4:]); d = pd.read_csv(f)
        d = d[(d.rule == "tailF") & (d.N <= 10_000_000)][["N", "width"]]; d.insert(0, "role", role); rows.append(d)
    h = pd.concat(rows).rename(columns={"width": "width_hedge_tailF"})
    cmp = pw[["role", "N", "width_tailB"]].merge(h, on=["role", "N"], how="left")
    cmp["ratio_tailB_over_tailF"] = cmp["width_tailB"] / cmp["width_hedge_tailF"]
    print("\nPrice of bandit feedback, Regime P width (EXP3-IX tailB / Hedge tailF):\n", cmp.round(2).to_string())
    cmp.to_csv(os.path.join(v1, "summary_feedback_price.csv"), index=False)
if covP:
    cP = pd.concat(covP); cO = pd.concat(covO)
    c = cP.groupby(["role", "N", "rule"]).agg(seeds=("contains_truth", "size"), coverage_P=("contains_truth", "mean"), width_med=("width", "median")).reset_index()
    c2 = cO.groupby(["role", "N", "rule"]).agg(coverage_O=("truth_feasible", "mean"), share_med=("share", "median")).reset_index()
    c = c.merge(c2, on=["role", "N", "rule"], how="left")
    c.to_csv(os.path.join(v1, "summary_coverage.csv"), index=False)
    print("\nCoverage over seeds (coarse grid), rules tailB/momentB:\n", c[c.rule.isin(["tailB", "momentB"])].round(3).to_string())
