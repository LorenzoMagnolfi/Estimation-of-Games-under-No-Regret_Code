"""Rule-based selection of focal roles and value scales from the pooled per-period footprints.

Rules (Design A register):
  lambda0 rule : per category, the pooled median over periods and roles of lambda*/CPA among low-regret agents (delta* < 0.10),
                 applied to the focal role's CPA target.
  focal role   : per category, among roles that do not exhaust their budget in the majority of periods, the role with the median
                 budget (ties: lower role index).
Also reports, per role: share of periods exhausting, class frequencies, median lambda*/CPA, median delta*.
Usage: python select_roles.py <footprint_csv_glob> <out_csv>
"""
import glob
import sys

import numpy as np
import pandas as pd

files = sorted(glob.glob(sys.argv[1])); out = sys.argv[2]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"{len(files)} periods, {len(df)} role-periods")
low = df[df.delta_star < 0.10]
lam_rule = low.groupby("category").lam_star_over_cpa.median()
lam_rule_all = low.lam_star_over_cpa.median()
print("\nlambda*/CPA among low-regret agents, median by category:\n", lam_rule.round(3).to_string(), f"\n  pooled: {lam_rule_all:.3f} (n = {len(low)})")
roles = df.groupby("role").agg(category=("category", "first"), budget_med=("budget", "median"), cpa_med=("cpa", "median"),
                               n_periods=("period", "nunique"), exhaust_share=("exhaust_step", lambda s: (s < 48).mean()),
                               delta_med=("delta_star", "median"), low_share=("delta_star", lambda s: (s < 0.10).mean()),
                               lam_cpa_med=("lam_star_over_cpa", "median"), alpha_cpa_med=("alpha_over_cpa", "median"),
                               win_med=("win_rate", "median"), pv_med=("mean_pvalue", "median"))
cls = df.groupby("role").cls.agg(lambda s: s.value_counts().idxmax())
roles["cls_mode"] = cls
roles["cls_stable"] = df.groupby("role").cls.agg(lambda s: s.value_counts().iloc[0] / len(s))
sel = []
for cat, g in roles.groupby("category"):
    elig = g[g.exhaust_share <= 0.5]
    if len(elig) == 0:
        elig = g
    med = elig.budget_med.median()
    pick = elig.iloc[(elig.budget_med - med).abs().argsort(kind="stable").values[0]]
    lam_c = lam_rule.get(cat, lam_rule_all)
    # lambda0 rule: the POOLED median of lambda*/CPA among low-regret agents (category medians rest on 10-25 role-periods
    # and are reported for sensitivity); applied to the focal role's CPA target
    sel.append(dict(category=int(cat), role=int(pick.name), budget=float(pick.budget_med), cpa=float(pick.cpa_med),
                    lam_rule_over_cpa=float(lam_rule_all), lam_rule_category=float(lam_c), lambda0=float(lam_rule_all * pick.cpa_med),
                    lambda0_category=float(lam_c * pick.cpa_med), exhaust_share=float(pick.exhaust_share),
                    cls_mode=pick.cls_mode, n_eligible=int(len(elig))))
sel = pd.DataFrame(sel)
roles.to_csv(out.replace(".csv", "_roles.csv")); sel.to_csv(out, index=False)
pd.set_option("display.width", 220)
print("\nroles (pooled):\n", roles.round(3).to_string())
print("\nselected focal roles:\n", sel.round(3).to_string())
