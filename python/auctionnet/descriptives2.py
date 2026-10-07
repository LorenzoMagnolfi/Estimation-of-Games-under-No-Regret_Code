"""Refined descriptives: multipliers on active rows only, budget exhaustion, CPA on expected conversions.
Usage: python descriptives2.py <results_dir>
"""
import os
import sys

import numpy as np
import pandas as pd

res = sys.argv[1]
d = pd.read_parquet(os.path.join(res, "period_ranked.parquet"),
                    columns=["advertiserNumber", "advertiserCategoryIndex", "budget", "CPAConstraint", "timeStepIndex",
                             "remainingBudget", "oppid", "pValue", "bid", "xi", "adSlot", "cost", "isExposed",
                             "conversionAction", "rank"])
d["active"] = d.bid > 0
d["alpha"] = np.where((d.pValue > 0) & d.active, d.bid / np.maximum(d.pValue, 1e-12), np.nan)
d["paid"] = d.cost * d.isExposed
d["exp_conv"] = d.xi * d.isExposed * d.pValue

# exhaustion: first step at which the advertiser bids zero on every opportunity
act_step = d.groupby(["advertiserNumber", "timeStepIndex"]).active.mean().unstack()
exhaust = act_step.apply(lambda r: int(r.index[r == 0].min()) if (r == 0).any() else 48, axis=1)
adv = d.groupby("advertiserNumber").agg(
    category=("advertiserCategoryIndex", "first"), budget=("budget", "first"), cpa_target=("CPAConstraint", "first"),
    win_rate=("xi", "mean"), paid=("paid", "sum"), conv=("conversionAction", "sum"), exp_conv=("exp_conv", "sum"),
    alpha_med=("alpha", "median"), alpha_p10=("alpha", lambda s: s.quantile(.1)), alpha_p90=("alpha", lambda s: s.quantile(.9)))
adv["exhaust_step"] = exhaust
adv["budget_used"] = adv.paid / adv.budget
adv["cpa_realized"] = adv.paid / np.maximum(adv.conv, 1)
adv["cpa_expected"] = adv.paid / np.maximum(adv.exp_conv, 1e-9)
adv["cpa_exp_over_target"] = adv.cpa_expected / adv.cpa_target
adv["alpha_over_cpa"] = adv.alpha_med / adv.cpa_target
adv["alpha_spread"] = adv.alpha_p90 / adv.alpha_p10
adv.to_csv(os.path.join(res, "advertisers_refined.csv"))
print("=== advertisers (active rows) ===")
print(adv[["category", "budget", "cpa_target", "exhaust_step", "budget_used", "win_rate", "conv", "cpa_expected",
           "cpa_exp_over_target", "alpha_med", "alpha_over_cpa", "alpha_spread"]].round(2).to_string())
print("\nsummary:\n", adv[["exhaust_step", "budget_used", "win_rate", "conv", "cpa_exp_over_target", "alpha_over_cpa", "alpha_spread"]].describe().round(3).to_string())
print("advertisers exhausting before step 48:", int((adv.exhaust_step < 48).sum()), "; median exhaustion step among them:",
      adv.exhaust_step[adv.exhaust_step < 48].median())
print("share with expected CPA <= target:", round((adv.cpa_exp_over_target <= 1).mean(), 3),
      "; <= 1.1 x target:", round((adv.cpa_exp_over_target <= 1.1).mean(), 3))

# multiplier path on active steps
ap = d[d.active & (d.pValue > 0)].groupby(["advertiserNumber", "timeStepIndex"]).alpha.median().unstack()
ap.to_csv(os.path.join(res, "alpha_by_step_active.csv"))
lchg = np.log(ap).diff(axis=1)
print("\nmultiplier path (active steps): median |log change| per step", round(float(np.nanmedian(np.abs(lchg.values))), 4),
      "; share of steps with |log change|>0.1:", round(float(np.nanmean(np.abs(lchg.values) > 0.1)), 3),
      "; >0.3:", round(float(np.nanmean(np.abs(lchg.values) > 0.3)), 3))
rng_ratio = (ap.max(axis=1) / ap.min(axis=1))
print("within-advertiser max/min multiplier over active steps: median", round(rng_ratio.median(), 2), " 90th pct", round(rng_ratio.quantile(.9), 2))
# which advertisers keep a constant multiplier (OnlineLp-like) vs adjust (PID-like)
cv_path = ap.std(axis=1) / ap.mean(axis=1)
print("CV of the step-multiplier path across advertisers: quantiles", cv_path.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
# within-step dispersion of alpha (Abid-type strategies have alpha proportional to pValue)
g = d[d.active & (d.pValue > 0)].groupby(["advertiserNumber", "timeStepIndex"]).alpha
cvw = (g.std() / g.mean()).groupby(level=0).median()
print("advertisers with within-step multiplier CV > 0.05 (bid not proportional to pValue):", list(cvw[cvw > 0.05].index))

# rival competition faced: distribution of the 3rd-highest rival bid relative to own bid, for active rows
top = d[d["rank"] <= 4][["oppid", "rank", "bid"]].pivot(index="oppid", columns="rank", values="bid")
top.columns = [f"b{int(c)}" for c in top.columns]
print("\ntop-4 bids per opportunity, quantiles:\n", top.quantile([.1, .5, .9]).round(4).to_string())
print("mean share of opportunities in which an active advertiser's bid exceeds the 4th-highest bid (i.e., wins):",
      round(d[d.active].xi.mean(), 4))
# distribution of pValue conditional on winning vs not (selection on context)
print("mean pValue | win:", round(d[d.xi == 1].pValue.mean(), 5), " | active & lose:", round(d[d.active & (d.xi == 0)].pValue.mean(), 5))
print("exposure by slot (all winners):", d[d.xi == 1].groupby("adSlot").isExposed.mean().round(3).to_dict())
