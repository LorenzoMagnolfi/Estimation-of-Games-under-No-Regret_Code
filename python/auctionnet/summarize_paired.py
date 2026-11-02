"""Summarize the paired-environment row (run_paired.py) for all roles against the i.i.d. direct-tail results (tailF_v1).

Reads results/designA/paired_v1/role<r>/ and results/designA/tailF_v1/role<r>/ and writes to results/designA/paired_v1/:
  summary_P.csv         Regime P by role, N and rule: paired interval, width, truth coverage; the i.i.d. tailF width and the
                        ratio paired / iid for the tailF rule;
  summary_O.csv         Regime O by role, N and rule: share of the (mu, sigma) grid, feasibility of the lognormal truth and of
                        the empirical context masses (the paired DGP's truth), fraction of the bids-only region removed;
  summary_coverage.csv  20 coverage seeds (coarse grid, paired): truth coverage in P (median width) and in O (lognormal and
                        empirical feasibility, median share) by rule.
Usage: python summarize_paired.py <designA_dir>
"""
import glob
import os
import sys

import pandas as pd

da = sys.argv[1]
v1 = os.path.join(da, "paired_v1"); iid = os.path.join(da, "tailF_v1")
P, O, cP, cO = [], [], [], []
for rdir in sorted(glob.glob(os.path.join(v1, "role*"))):
    role = int(os.path.basename(rdir)[4:])
    p = pd.read_csv(os.path.join(rdir, "hedge_1e8_paired_regimeP.csv")); p.insert(0, "role", role); P.append(p)
    o = pd.read_csv(os.path.join(rdir, "hedge_1e8_paired_regimeO.csv")); o.insert(0, "role", role); O.append(o)
    cov = os.path.join(rdir, "coverage_hedge_1e7_paired")
    if os.path.exists(os.path.join(cov, "coverage_hedge_P.csv")):
        d = pd.read_csv(os.path.join(cov, "coverage_hedge_P.csv")); d.insert(0, "role", role); cP.append(d)
        d = pd.read_csv(os.path.join(cov, "coverage_hedge_O.csv")); d.insert(0, "role", role); cO.append(d)
P = pd.concat(P); O = pd.concat(O)
pw = P.pivot_table(index=["role", "N"], columns="rule", values=["lam_lo", "lam_hi", "width", "contains_truth"])
pw.columns = [f"{a}_{b}" for a, b in pw.columns]; pw = pw.reset_index()
rows = []
for rdir in sorted(glob.glob(os.path.join(iid, "role*"))):
    role = int(os.path.basename(rdir)[4:])
    q = pd.read_csv(os.path.join(rdir, "hedge_1e8_regimeP.csv")); q = q[q.rule == "tailF"][["N", "width", "lam_lo", "lam_hi"]]
    q.insert(0, "role", role); rows.append(q.rename(columns={"width": "width_tailF_iid", "lam_lo": "lam_lo_tailF_iid", "lam_hi": "lam_hi_tailF_iid"}))
if rows:
    pw = pw.merge(pd.concat(rows), on=["role", "N"], how="left")
    if "width_tailF" in pw:
        pw["width_ratio_paired_over_iid"] = pw["width_tailF"] / pw["width_tailF_iid"]
pw.to_csv(os.path.join(v1, "summary_P.csv"), index=False)
vals = ["share", "truth_feasible"] + (["emp_feasible"] if "emp_feasible" in O else [])
ow = O.pivot_table(index=["role", "N"], columns="rule", values=vals)
ow.columns = [f"{a}_{b}" for a, b in ow.columns]; ow = ow.reset_index()
for rule in ["markov", "tailF", "momentF"]:
    if f"share_{rule}" in ow and "share_none" in ow:
        ow[f"removed_by_{rule}"] = 1.0 - ow[f"share_{rule}"] / ow["share_none"]
ow.to_csv(os.path.join(v1, "summary_O.csv"), index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 40)
print("Regime P (paired Hedge run), by role and N:\n", pw.round(2).to_string())
print("\nRegime O (paired Hedge run), by role and N:\n", ow.round(3).to_string())
if cP:
    a = pd.concat(cP); b = pd.concat(cO)
    c = a.groupby(["role", "N", "rule"]).agg(seeds=("contains_truth", "size"), coverage_P=("contains_truth", "mean"), width_med=("width", "median")).reset_index()
    agg = dict(coverage_O_lognormal=("truth_feasible", "mean"), share_med=("share", "median"))
    if "emp_feasible" in b:
        agg["coverage_O_empirical"] = ("emp_feasible", "mean")
    c = c.merge(b.groupby(["role", "N", "rule"]).agg(**agg).reset_index(), on=["role", "N", "rule"])
    c.to_csv(os.path.join(v1, "summary_coverage.csv"), index=False)
    print("\nCoverage over seeds (paired, coarse grid):\n", c.round(3).to_string())
