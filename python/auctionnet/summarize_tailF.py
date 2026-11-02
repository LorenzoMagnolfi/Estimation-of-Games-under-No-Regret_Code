"""Summarize the direct-tail reruns (rerun_tailF.py) for all roles against the Markov benchmark.

Reads results/designA/tailF_v1/role<r>/ and writes to results/designA/tailF_v1/:
  summary_P.csv         Regime P: interval, width and truth coverage by role, N and rule (tailF vs markov), with the width ratio;
  summary_O.csv         Regime O: share of the grid and truth feasibility by role, N and rule (none, markov, tailF), with the
                        fraction of the bids-only region removed by each obedience rule;
  summary_coverage.csv  20 coverage seeds (coarse grid): truth coverage in P (and median width) and in O (and median share), tailF;
  tolerance_ratio.csv   per-context tolerance ratio markov / tailF in Regime P by role and N (from the eps_v* columns).
Usage: python summarize_tailF.py <designA_dir>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

v1 = os.path.join(sys.argv[1], "tailF_v1")
P, O, covP, covO, tol = [], [], [], [], []
for rdir in sorted(glob.glob(os.path.join(v1, "role*"))):
    role = int(os.path.basename(rdir)[4:])
    p = pd.read_csv(os.path.join(rdir, "hedge_1e8_regimeP.csv")); p.insert(0, "role", role); P.append(p)
    o = pd.read_csv(os.path.join(rdir, "hedge_1e8_regimeO.csv")); o.insert(0, "role", role); O.append(o)
    eps_cols = [c for c in p.columns if c.startswith("eps_v")]
    for N, g in p.groupby("N"):
        m = g[g.rule == "markov"][eps_cols].values.ravel(); t = g[g.rule == "tailF"][eps_cols].values.ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(t > 0, m / t, np.nan)
        tol.append(dict(role=role, N=N, **{f"ratio_v{j + 1}": r[j] for j in range(len(r))}))
    for f in glob.glob(os.path.join(rdir, "cov*_hedge_1e7_regimeP.csv")):
        s = os.path.basename(f).split("_")[0][3:]
        d = pd.read_csv(f); d.insert(0, "seed", s); d.insert(0, "role", role); covP.append(d)
        d = pd.read_csv(f.replace("_regimeP.csv", "_regimeO.csv")); d.insert(0, "seed", s); d.insert(0, "role", role); covO.append(d)
P = pd.concat(P); O = pd.concat(O)
pw = P.pivot_table(index=["role", "N"], columns="rule", values=["lam_lo", "lam_hi", "width", "contains_truth"])
pw.columns = [f"{a}_{b}" for a, b in pw.columns]
pw = pw.reset_index()
if "width_markov" in pw and "width_tailF" in pw:
    pw["width_ratio_markov_over_tailF"] = pw["width_markov"] / pw["width_tailF"]
pw.to_csv(os.path.join(v1, "summary_P.csv"), index=False)
ow = O.pivot_table(index=["role", "N"], columns="rule", values=["share", "truth_feasible"])
ow.columns = [f"{a}_{b}" for a, b in ow.columns]
ow = ow.reset_index()
for rule in ["markov", "tailF"]:
    if f"share_{rule}" in ow:
        ow[f"removed_by_{rule}"] = 1.0 - ow[f"share_{rule}"] / ow["share_none"]
ow.to_csv(os.path.join(v1, "summary_O.csv"), index=False)
pd.DataFrame(tol).to_csv(os.path.join(v1, "tolerance_ratio.csv"), index=False)
pd.set_option("display.width", 240); pd.set_option("display.max_rows", 400)
print("Regime P (main Hedge run), by role and N:\n", pw.round(2).to_string())
print("\nRegime O (main Hedge run), by role and N:\n", ow.round(3).to_string())
print("\nPer-context tolerance ratio markov / tailF (Regime P):\n", pd.DataFrame(tol).round(1).to_string())
if covP:
    cP = pd.concat(covP); cO = pd.concat(covO)
    c = cP.groupby(["role", "N", "rule"]).agg(seeds=("contains_truth", "size"), coverage_P=("contains_truth", "mean"), width_med=("width", "median"),
                                              lo_med=("lam_lo", "median"), hi_med=("lam_hi", "median")).reset_index()
    cO2 = cO.groupby(["role", "N", "rule"]).agg(coverage_O=("truth_feasible", "mean"), share_med=("share", "median")).reset_index()
    c = c.merge(cO2, on=["role", "N", "rule"])
    c.to_csv(os.path.join(v1, "summary_coverage.csv"), index=False)
    print("\nCoverage over seeds (tailF, coarse grid):\n", c.round(3).to_string())
