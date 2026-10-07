"""Descriptives for one AuctionNet period (uses period_ranked.parquet from verify_auction.py).
Usage: python descriptives.py <results_dir>
Writes CSV tables to <results_dir> and prints a summary.
"""
import os
import sys

import numpy as np
import pandas as pd

res = sys.argv[1]
d = pd.read_parquet(os.path.join(res, "period_ranked.parquet"))
d["alpha"] = d.bid / np.maximum(d.pValue, 1e-12)
d["value_won"] = d.xi * d.isExposed * d.pValue

adv = d.groupby("advertiserNumber").agg(
    category=("advertiserCategoryIndex", "first"), budget=("budget", "first"), cpa_target=("CPAConstraint", "first"),
    n_opps=("oppid", "nunique"), win_rate=("xi", "mean"), exposed_rate=("isExposed", "mean"),
    total_cost=("cost", "sum"), conversions=("conversionAction", "sum"), exp_conv=("value_won", "sum"),
    mean_pvalue=("pValue", "mean"), med_alpha=("alpha", "median"), min_rem=("remainingBudget", "min"),
    last_step=("timeStepIndex", "max"))
adv["realized_cpa"] = adv.total_cost / np.maximum(adv.conversions, 1)
adv["cpa_over_target"] = adv.realized_cpa / adv.cpa_target
adv["budget_used"] = adv.total_cost / adv.budget
adv["alpha_over_cpa"] = adv.med_alpha / adv.cpa_target
slots = d[d.xi == 1].groupby(["advertiserNumber", "adSlot"]).size().unstack(fill_value=0)
slots.columns = [f"slot{c}" for c in slots.columns]
adv = adv.join(slots)
adv.to_csv(os.path.join(res, "advertisers.csv"))
print("=== advertisers ===")
print(adv[["category", "budget", "cpa_target", "win_rate", "budget_used", "realized_cpa", "cpa_over_target",
           "med_alpha", "alpha_over_cpa", "last_step"]].round(3).to_string())
print("\nsummary:\n", adv[["win_rate", "budget_used", "cpa_over_target", "alpha_over_cpa", "mean_pvalue", "med_alpha"]].describe().round(3).to_string())

ap = d[d.pValue > 0].groupby(["advertiserNumber", "timeStepIndex"]).alpha.median().unstack()
ap.to_csv(os.path.join(res, "alpha_by_step.csv"))
chg = np.log(ap).diff(axis=1)
print("\nalpha path: median |log change| per step:", round(float(np.nanmedian(np.abs(chg.values))), 4),
      "; share of steps with |log change|>0.1:", round(float(np.nanmean(np.abs(chg.values) > 0.1)), 3))
print("alpha at step 0 vs 47 (median across advertisers):", round(ap[0].median(), 3), round(ap[47].median(), 3))
print("alpha range within advertiser (max/min over steps): median", round((ap.max(axis=1) / ap.min(axis=1)).median(), 3))

pv_cat = d.groupby("advertiserCategoryIndex").pValue.describe(percentiles=[.1, .5, .9])
pv_cat.to_csv(os.path.join(res, "pvalue_by_category.csv"))
print("\npValue by category:\n", pv_cat.round(5).to_string())
pv_step = d.groupby("timeStepIndex").agg(n_opps=("oppid", "nunique"), mean_pv=("pValue", "mean"), lwc=("leastWinningCost", "median"))
pv_step.to_csv(os.path.join(res, "by_step.csv"))
print("\nby step (first/last 5):\n", pv_step.head(5).round(5).to_string(), "\n", pv_step.tail(5).round(5).to_string())
disp = d.groupby("advertiserNumber").pValue.agg(["mean", "std"])
disp["cv"] = disp["std"] / disp["mean"]
print("\nwithin-advertiser pValue CV: median", round(disp.cv.median(), 3), " range", round(disp.cv.min(), 3), round(disp.cv.max(), 3))
sub_ids = d.oppid.drop_duplicates().sample(20000, random_state=1)
wide = d[d.oppid.isin(sub_ids)].pivot(index="oppid", columns="advertiserNumber", values="pValue")
corr = wide.corr().values
off = corr[~np.eye(corr.shape[0], dtype=bool)]
print("cross-advertiser pValue correlation within opportunity: mean", round(float(np.nanmean(off)), 3), " 90th pct", round(float(np.nanquantile(off, .9)), 3))

lwc = d.drop_duplicates("oppid")[["oppid", "timeStepIndex", "leastWinningCost"]]
print("\nleastWinningCost quantiles:", lwc.leastWinningCost.quantile([.1, .5, .9]).round(4).to_dict())
print("highest bid quantiles:", d[d["rank"] == 1].bid.quantile([.1, .5, .9]).round(4).to_dict())
print("median alpha across advertisers: quantiles", adv.med_alpha.quantile([.1, .5, .9]).round(2).to_dict())
print("share of advertisers with realized CPA <= target:", round((adv.cpa_over_target <= 1).mean(), 3),
      "; with budget_used>0.95:", round((adv.budget_used > 0.95).mean(), 3))
