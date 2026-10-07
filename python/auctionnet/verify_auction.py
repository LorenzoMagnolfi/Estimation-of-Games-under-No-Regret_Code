"""Verify the AuctionNet auction rule on one period parquet.

Checks: rows per advertiser and opportunity; top-3 allocation by bid; next-price payments
(slot j pays the (j+1)-th highest bid, floored at the reserve); leastWinningCost = 4th highest bid;
cost recorded only when exposed; bid = alpha * pValue with alpha constant within advertiser x step.
Usage: python verify_auction.py <parquet_path> <results_dir>
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

pth, res = sys.argv[1], sys.argv[2]
os.makedirs(res, exist_ok=True)
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
print("opportunities per step:\n", df.groupby("timeStepIndex").pvIndex.nunique().describe().round(1).to_string())

# --- reconstruct the auction per opportunity ---
d = df.sort_values(["oppid", "bid"], ascending=[True, False]).copy()
d["rank"] = d.groupby("oppid").cumcount() + 1
d["next_bid"] = d.groupby("oppid").bid.shift(-1)
reserve = 0.01
top = d[d["rank"] <= 3].copy()
top["pay_hat"] = np.maximum(top["next_bid"].fillna(0.0), reserve)
top["slot_hat"] = top["rank"]
print(f"allocation check: share of top-3 rows whose recorded adSlot equals bid rank: {(top['adSlot'] == top['slot_hat']).mean():.4f}")
print("recorded winners per opportunity (mean):", round(d[d.xi == 1].groupby("oppid").size().mean(), 3),
      "; xi==1 rows with adSlot==0:", int(((d.xi == 1) & (d.adSlot == 0)).sum()))
b4 = d[d["rank"] == 4][["oppid", "bid"]].rename(columns={"bid": "b4"})
tie = d[d["rank"] == 3].merge(b4, on="oppid")
print(f"ties between 3rd and 4th bid: {np.isclose(tie.bid, tie.b4).mean():.5f}")

exp = top[top.isExposed == 1]
rel = np.abs(exp["cost"] - exp["pay_hat"]) / np.maximum(exp["pay_hat"], 1e-9)
print(f"payment check (exposed winners): median rel. error = {rel.median():.2e}; share within 1e-6: {(rel < 1e-6).mean():.4f}; n={len(exp)}")
notexp = top[top.isExposed == 0]
print("cost when NOT exposed (winners): mean", round(notexp["cost"].mean(), 6), " share zero:", round((notexp["cost"] == 0).mean(), 4))
print("cost for losers: max", d[d.xi == 0]["cost"].max())
alt4 = exp.merge(b4, on="oppid", how="left")["b4"].values
for lab, alt in [("own bid", exp["bid"].values), ("4th highest", alt4)]:
    r2 = np.abs(exp["cost"].values - alt) / np.maximum(alt, 1e-9)
    print(f"   alt payment '{lab}': share within 1e-6 = {np.nanmean(r2 < 1e-6):.4f}")
print("exposure rate by slot:", top.groupby("slot_hat").isExposed.mean().round(4).to_dict())

chk = d[["oppid", "leastWinningCost"]].drop_duplicates("oppid").merge(b4, on="oppid", how="left")
lwc_hat = np.maximum(chk.b4.fillna(0), reserve)
rel4 = np.abs(chk.leastWinningCost - lwc_hat) / np.maximum(chk.leastWinningCost, 1e-9)
rel4raw = np.abs(chk.leastWinningCost - chk.b4) / np.maximum(chk.leastWinningCost, 1e-9)
print(f"leastWinningCost vs max(4th bid, reserve): share within 1e-6 = {(rel4 < 1e-6).mean():.4f}; vs raw 4th bid: {(rel4raw < 1e-6).mean():.4f}")
print("leastWinningCost constant within opportunity:", round((df.groupby("oppid").leastWinningCost.nunique() == 1).mean(), 4))
print("conversion rate | exposed:", round(d[d.isExposed == 1].conversionAction.mean(), 5),
      " | not exposed:", round(d[d.isExposed == 0].conversionAction.mean(), 5),
      " mean pValue | exposed:", round(d[d.isExposed == 1].pValue.mean(), 5))

d["alpha"] = d.bid / np.maximum(d.pValue, 1e-12)
g = d[d.pValue > 0].groupby(["advertiserNumber", "timeStepIndex"]).alpha
cv = g.std() / g.mean()
print("within (advertiser,step) CV of bid/pValue: median", round(cv.median(), 4), " 90th pct", round(cv.quantile(.9), 4),
      " share<1e-3:", round((cv < 1e-3).mean(), 4))
print("share of rows with bid==0:", round((d.bid == 0).mean(), 4), " pValue==0:", round((d.pValue == 0).mean(), 4))
endtab = d.groupby("advertiserNumber").agg(last_step=("timeStepIndex", "max"), min_rem=("remainingBudget", "min"),
                                           budget=("budget", "first"), n_end=("isEnd", "sum"))
print("per-advertiser end-of-period stats:\n", endtab.describe().round(2).to_string())
d.to_parquet(os.path.join(res, "period_ranked.parquet"), compression="zstd")
print("saved ranked parquet", f"({time.time() - t0:.0f}s)")
