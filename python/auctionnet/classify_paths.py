"""Classify multiplier paths by their step-to-step signature (no strategy labels in the data).
PID-like: log changes concentrated at log(1.2)=0.182 and log(0.7)=-0.357 (and 0). Constant: no change.
Capped: multiplier equals 1.5 x CPA for most steps. Other: RL/LP-type paths.
Usage: python classify_paths.py <results_dir>
"""
import os, sys
import numpy as np, pandas as pd
res = sys.argv[1]
ap = pd.read_csv(os.path.join(res, "alpha_by_step_active.csv"), index_col=0)
adv = pd.read_csv(os.path.join(res, "advertisers_refined.csv"), index_col=0)
reg = pd.read_csv(os.path.join(res, "regret_lambda.csv")).set_index("advertiser")
rows = []
for i, r in ap.iterrows():
    a = r.values.astype(float); a = a[~np.isnan(a)]
    lc = np.diff(np.log(a[a > 0]))
    n = len(lc)
    if n == 0:
        rows.append(dict(advertiser=i, cls="single-step")); continue
    zero = np.mean(np.abs(lc) < 1e-6)
    pid = np.mean((np.abs(lc - np.log(1.2)) < 1e-3) | (np.abs(lc - np.log(0.7)) < 1e-3))
    cap = np.mean(np.abs(a / adv.loc[i, "cpa_target"] - 1.5) < 1e-6)
    if zero > 0.98: cls = "constant"
    elif pid + zero > 0.9 and pid > 0.05: cls = "PID-like"
    elif cap > 0.5: cls = "capped 1.5xCPA"
    else: cls = "other (LP/RL)"
    rows.append(dict(advertiser=i, cls=cls, share_zero=round(zero, 2), share_pid=round(pid, 2), share_cap=round(cap, 2),
                     n_steps=len(a), cv=round(a.std() / a.mean(), 3), nchanges=int((np.abs(lc) > 1e-6).sum())))
cl = pd.DataFrame(rows).set_index("advertiser").join(adv[["category", "cpa_target", "exhaust_step", "budget_used"]]).join(reg[["alpha_over_cpa", "lam_star_over_cpa", "delta_star"]])
cl.to_csv(os.path.join(res, "path_classes.csv"))
print(cl.round(3).to_string())
print("\ncounts:\n", cl.cls.value_counts().to_string())
print("\nby class: median delta*, share delta*<0.1, median exhaustion step, median alpha/CPA, median lambda*/CPA")
print(cl.groupby("cls").agg(n=("delta_star", "size"), med_delta=("delta_star", "median"), share_low=("delta_star", lambda s: (s < 0.1).mean()),
                            med_exhaust=("exhaust_step", "median"), med_alpha_cpa=("alpha_over_cpa", "median"), med_lam_cpa=("lam_star_over_cpa", "median")).round(3).to_string())
