"""Verify the AuctionNet auction rule on one period parquet.

Checks: rows per advertiser and opportunity; top-3 allocation by bid; next-price payments
(slot j pays the (j+1)-th highest bid, floored at the reserve); leastWinningCost = 4th highest bid;
cost recorded only when exposed; bid = alpha * pValue with alpha constant within advertiser x step;
conversions only when exposed, and the conversion rate against the mean pValue of exposed impressions.
Prints the checks and writes them to <results_dir>/verify_<period>.json. The ranked parquet used by the
period-13 pilot is written only without --no-save.
Usage: python verify_auction.py <parquet_path> <results_dir> [--no-save]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from provenance import code_commit

pth, res = sys.argv[1], sys.argv[2]
save_ranked = "--no-save" not in sys.argv[3:]
os.makedirs(res, exist_ok=True)
period = os.path.splitext(os.path.basename(pth))[0]
m = dict(period=period, parquet=os.path.abspath(pth))
t0 = time.time()
df = pq.read_table(pth).to_pandas()
print("columns:", list(df.columns))
print("rows:", len(df), f"({time.time() - t0:.0f}s)")
for c in ["advertiserNumber", "advertiserCategoryIndex", "timeStepIndex", "pvIndex", "xi",
          "adSlot", "isExposed", "conversionAction", "isEnd"]:
    if c in df:
        df[c] = df[c].astype(np.int64)
df["oppid"] = df.timeStepIndex * 1_000_000 + df.pvIndex  # pvIndex restarts within each step
n_adv = df.advertiserNumber.nunique()
n_pv = df.oppid.nunique()
n_step = df.timeStepIndex.nunique()
print(f"advertisers={n_adv} opportunities={n_pv} steps={n_step} rows/opportunity={len(df) / n_pv:.2f}")
rows_per_adv = df.groupby("advertiserNumber").size()
print("rows per advertiser: min/median/max", rows_per_adv.min(), int(rows_per_adv.median()), rows_per_adv.max())
opp_per_step = df.groupby("timeStepIndex").pvIndex.nunique()
print("opportunities per step:\n", opp_per_step.describe().round(1).to_string())
m.update(rows=int(len(df)), advertisers=int(n_adv), opportunities=int(n_pv), steps=int(n_step), rows_per_opportunity=len(df) / n_pv,
         rows_per_adv_min=int(rows_per_adv.min()), rows_per_adv_max=int(rows_per_adv.max()),
         opp_per_step_min=int(opp_per_step.min()), opp_per_step_max=int(opp_per_step.max()))

# --- reconstruct the auction per opportunity ---
d = df.sort_values(["oppid", "bid"], ascending=[True, False]).copy()
del df
d["rank"] = d.groupby("oppid").cumcount() + 1
d["next_bid"] = d.groupby("oppid").bid.shift(-1)
reserve = 0.01
top = d[d["rank"] <= 3].copy()
top["pay_hat"] = np.maximum(top["next_bid"].fillna(0.0), reserve)
top["slot_hat"] = top["rank"]
m["alloc_share"] = float((top["adSlot"] == top["slot_hat"]).mean())
print(f"allocation check: share of top-3 rows whose recorded adSlot equals bid rank: {m['alloc_share']:.4f}")
m["winners_per_opp"] = float(d[d.xi == 1].groupby("oppid").size().mean())
m["xi1_slot0_rows"] = int(((d.xi == 1) & (d.adSlot == 0)).sum())
print("recorded winners per opportunity (mean):", round(m["winners_per_opp"], 3), "; xi==1 rows with adSlot==0:", m["xi1_slot0_rows"])
b4 = d[d["rank"] == 4][["oppid", "bid"]].rename(columns={"bid": "b4"})
tie = d[d["rank"] == 3].merge(b4, on="oppid")
m["tie_share_3v4"] = float(np.isclose(tie.bid, tie.b4).mean())
print(f"ties between 3rd and 4th bid: {m['tie_share_3v4']:.5f}")

exp = top[top.isExposed == 1]
rel = np.abs(exp["cost"] - exp["pay_hat"]) / np.maximum(exp["pay_hat"], 1e-9)
m.update(pay_median_rel_err=float(rel.median()), pay_share_exact=float((rel < 1e-6).mean()), n_exposed_winners=int(len(exp)))
print(f"payment check (exposed winners): median rel. error = {m['pay_median_rel_err']:.2e}; share within 1e-6: {m['pay_share_exact']:.4f}; n={len(exp)}")
notexp = top[top.isExposed == 0]
m.update(cost_not_exposed_mean=float(notexp["cost"].mean()), cost_not_exposed_share_zero=float((notexp["cost"] == 0).mean()),
         cost_losers_max=float(d[d.xi == 0]["cost"].max()))
print("cost when NOT exposed (winners): mean", round(m["cost_not_exposed_mean"], 6), " share zero:", round(m["cost_not_exposed_share_zero"], 4))
print("cost for losers: max", m["cost_losers_max"])
alt4 = exp.merge(b4, on="oppid", how="left")["b4"].values
for lab, key, alt in [("own bid", "pay_alt_own_bid", exp["bid"].values), ("4th highest", "pay_alt_4th", alt4)]:
    r2 = np.abs(exp["cost"].values - alt) / np.maximum(alt, 1e-9)
    m[key] = float(np.nanmean(r2 < 1e-6))
    print(f"   alt payment '{lab}': share within 1e-6 = {m[key]:.4f}")
ex = top.groupby("slot_hat").isExposed.mean()
m.update(expo_slot1=float(ex.get(1, np.nan)), expo_slot2=float(ex.get(2, np.nan)), expo_slot3=float(ex.get(3, np.nan)))
print("exposure rate by slot:", ex.round(4).to_dict())

chk = d[["oppid", "leastWinningCost"]].drop_duplicates("oppid").merge(b4, on="oppid", how="left")
lwc_hat = np.maximum(chk.b4.fillna(0), reserve)
rel4 = np.abs(chk.leastWinningCost - lwc_hat) / np.maximum(chk.leastWinningCost, 1e-9)
rel4raw = np.abs(chk.leastWinningCost - chk.b4) / np.maximum(chk.leastWinningCost, 1e-9)
m.update(lwc_share_exact=float((rel4 < 1e-6).mean()), lwc_share_exact_raw4=float((rel4raw < 1e-6).mean()),
         lwc_constant_share=float((d.groupby("oppid").leastWinningCost.nunique() == 1).mean()))
print(f"leastWinningCost vs max(4th bid, reserve): share within 1e-6 = {m['lwc_share_exact']:.4f}; vs raw 4th bid: {m['lwc_share_exact_raw4']:.4f}")
print("leastWinningCost constant within opportunity:", round(m["lwc_constant_share"], 4))
m.update(conv_rate_exposed=float(d[d.isExposed == 1].conversionAction.mean()), conv_rate_not_exposed=float(d[d.isExposed == 0].conversionAction.mean()),
         mean_pvalue_exposed=float(d[d.isExposed == 1].pValue.mean()))
print("conversion rate | exposed:", round(m["conv_rate_exposed"], 5), " | not exposed:", round(m["conv_rate_not_exposed"], 5),
      " mean pValue | exposed:", round(m["mean_pvalue_exposed"], 5))

d["alpha"] = d.bid / np.maximum(d.pValue, 1e-12)
g = d[d.pValue > 0].groupby(["advertiserNumber", "timeStepIndex"]).alpha
cv = g.std() / g.mean()
m.update(alpha_cv_median=float(cv.median()), alpha_cv_p90=float(cv.quantile(.9)), alpha_cv_share_lt_1e3=float((cv < 1e-3).mean()),
         share_bid_zero=float((d.bid == 0).mean()), share_pvalue_zero=float((d.pValue == 0).mean()))
print("within (advertiser,step) CV of bid/pValue: median", round(m["alpha_cv_median"], 4), " 90th pct", round(m["alpha_cv_p90"], 4),
      " share<1e-3:", round(m["alpha_cv_share_lt_1e3"], 4))
print("share of rows with bid==0:", round(m["share_bid_zero"], 4), " pValue==0:", round(m["share_pvalue_zero"], 4))
endtab = d.groupby("advertiserNumber").agg(last_step=("timeStepIndex", "max"), min_rem=("remainingBudget", "min"),
                                           budget=("budget", "first"), n_end=("isEnd", "sum"))
print("per-advertiser end-of-period stats:\n", endtab.describe().round(2).to_string())
if save_ranked:
    d.to_parquet(os.path.join(res, "period_ranked.parquet"), compression="zstd")
    print("saved ranked parquet", f"({time.time() - t0:.0f}s)")
commit, dirty = code_commit()
m.update(code_commit=commit, code_dirty=dirty, seconds=round(time.time() - t0))
with open(os.path.join(res, f"verify_{period}.json"), "w") as f:
    json.dump(m, f, indent=1)
print(f"wrote verify_{period}.json ({time.time() - t0:.0f}s)")
