"""Summarize the corrected Regime O rerun (rerun_regimeO.py) for all roles.

Reads results/designA/regimeO_v2/role<r>/<tag>_regimeO.csv and writes three tables to results/designA/regimeO_v2/:
  summary_main.csv      main Hedge (1e8) and EXP3 (1e7) runs: share of the grid and truth feasibility by N and rule;
  summary_variants.csv  robustness variants and seeds at 1e7 and 1e8: shares for none, markov (corrected), markov_realized;
  summary_coverage.csv  20 coverage seeds (coarse grid): truth feasibility rate and median share by N and rule.
Usage: python summarize_regimeO_v2.py <designA_dir>
"""
import glob
import os
import sys

import pandas as pd

v2 = os.path.join(sys.argv[1], "regimeO_v2")
main, var, cov = [], [], []
for rdir in sorted(glob.glob(os.path.join(v2, "role*"))):
    role = int(os.path.basename(rdir)[4:])
    for f in glob.glob(os.path.join(rdir, "*_regimeO.csv")):
        tag = os.path.basename(f).replace("_regimeO.csv", "")
        d = pd.read_csv(f)
        d.insert(0, "tag", tag); d.insert(0, "role", role)
        if tag in ("hedge_1e8", "exp3_1e7"):
            main.append(d)
        elif tag.startswith("cov"):
            cov.append(d)
        else:
            var.append(d)
main = pd.concat(main); var = pd.concat(var); cov = pd.concat(cov)
main.to_csv(os.path.join(v2, "summary_main.csv"), index=False)
w = var[var.N.isin([10_000_000, 100_000_000])].pivot_table(index=["role", "tag", "N"], columns="rule", values="share").reset_index()
w["markov_over_none"] = w["markov"] / w["none"]; w["realized_over_none"] = w["markov_realized"] / w["none"]
w.to_csv(os.path.join(v2, "summary_variants.csv"), index=False)
c = cov.groupby(["role", "N", "rule"]).agg(seeds=("truth_feasible", "size"), coverage=("truth_feasible", "mean"),
                                           share_med=("share", "median")).reset_index()
c.to_csv(os.path.join(v2, "summary_coverage.csv"), index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)
m = main[main.N.isin([10_000_000, 100_000_000])].pivot_table(index=["role", "tag", "N"], columns="rule", values="share")
print("Main runs, share of the grid:\n", m.round(3).to_string())
print("\nTruth feasible (main runs, all N), by rule:\n", main.groupby("rule").truth_feasible.agg(["sum", "size"]).to_string())
print("\nVariants, markov / none (corrected) and markov_realized / none:\n", w.round(3).to_string())
print("\nCoverage over seeds (coarse grid):\n", c[c.rule.isin(["markov", "markov_realized", "none"])].round(3).to_string())
