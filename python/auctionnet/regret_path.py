"""Regret over the day: does average regret vanish along the path? Plus decision counts.

For each advertiser, at its min-regret lambda* (from regret_lambda.csv), compute per-auction payoffs in time order
for the actual bids and for every fixed multiplier on the grid. Report:
  - cumulative average regret (relative to the best fixed multiplier in hindsight over the same prefix) at the end of
    each of 6 blocks of 8 steps, i.e. R_N/N as N grows within the day;
  - block-wise regret (regret computed on each block alone, against the block's own best fixed multiplier);
  - number of distinct multipliers and multiplier changes over the day (decision epochs);
  - per-context (pValue quintile) counts of auctions and of distinct multiplier values.
Usage: python regret_path.py <results_dir>
"""
import os
import sys

import numpy as np
import pandas as pd

res = sys.argv[1]
E = np.array([1.0, 0.80, 0.48])
RES = 0.01
d = pd.read_parquet(os.path.join(res, "period_ranked.parquet"),
                    columns=["advertiserNumber", "timeStepIndex", "oppid", "pValue", "bid", "rank"])
reg = pd.read_csv(os.path.join(res, "regret_lambda.csv")).set_index("advertiser")
cls = pd.read_csv(os.path.join(res, "path_classes.csv"), index_col=0)["cls"]
top = d[d["rank"] <= 4][["oppid", "rank", "bid", "advertiserNumber"]]
tb = top.pivot(index="oppid", columns="rank", values="bid"); tb.columns = [f"b{int(c)}" for c in tb.columns]
ta = top.pivot(index="oppid", columns="rank", values="advertiserNumber"); ta.columns = [f"a{int(c)}" for c in ta.columns]
alphas = np.exp(np.linspace(np.log(20), np.log(400), 31))


def outcome(b, r1, r2, r3):
    s1 = b > r1
    s2 = (~s1) & (b > r2)
    s3 = (~s1) & (~s2) & (b > r3)
    price = np.maximum(np.where(s1, r1, np.where(s2, r2, np.where(s3, r3, 0.0))), RES)
    e = np.where(s1, E[0], np.where(s2, E[1], np.where(s3, E[2], 0.0)))
    return e, price


rows = []
for i in sorted(d.advertiserNumber.unique()):
    di = d[d.advertiserNumber == i].set_index("oppid").sort_index().join(tb).join(ta)
    B = di[["b1", "b2", "b3", "b4"]].values
    A = di[["a1", "a2", "a3", "a4"]].values
    Bs = -np.sort(-np.where(A == i, -1.0, B), axis=1)[:, :3]
    r1, r2, r3 = Bs[:, 0], Bs[:, 1], Bs[:, 2]
    v = di.pValue.values
    step = di.timeStepIndex.values
    act = (di.bid.values > 0) & (v > 0)
    v, r1, r2, r3, bact, step = v[act], r1[act], r2[act], r3[act], di.bid.values[act], step[act]
    lam = reg.loc[i, "lam_star"]
    e_a, p_a = outcome(bact, r1, r2, r3)
    u_a = e_a * (lam * v - p_a)
    U = np.zeros((len(alphas), len(v)))
    for j, a in enumerate(alphas):
        e, p = outcome(a * v, r1, r2, r3)
        U[j] = e * (lam * v - p)
    cum_a = np.cumsum(u_a)
    cum_U = np.cumsum(U, axis=1)
    n = len(v)
    # cumulative average regret at block ends (blocks of 8 steps)
    ends = [np.searchsorted(step, 8 * (k + 1), side="left") - 1 for k in range(6)]
    ends = [max(min(e_, n - 1), 0) for e_ in ends]
    cum_rel = []
    for e_ in ends:
        best = cum_U[:, e_].max()
        cum_rel.append((best - cum_a[e_]) / max(best, 1e-12))
    # block-wise regret against the block's own best fixed multiplier
    blk_rel = []
    prev = 0
    for e_ in ends:
        seg = slice(prev, e_ + 1)
        bestb = U[:, seg].sum(axis=1).max()
        blk_rel.append((bestb - u_a[seg].sum()) / max(abs(bestb), 1e-12))
        prev = e_ + 1
    # decision epochs
    alpha_path = pd.Series(bact / v).groupby(step).median()
    n_distinct = int(np.round(alpha_path, 6).nunique())
    n_changes = int((np.abs(np.diff(np.log(alpha_path.values))) > 1e-6).sum())
    # per-context counts
    q = np.quantile(v, [0.2, 0.4, 0.6, 0.8]); bucket = np.searchsorted(q, v)
    per_ctx = np.bincount(bucket, minlength=5)
    rows.append(dict(advertiser=i, cls=cls.get(i, "?"), n_active=n, n_steps_active=len(alpha_path), n_distinct_alpha=n_distinct,
                     n_alpha_changes=n_changes, min_ctx_count=int(per_ctx.min()),
                     cum_rel_b1=cum_rel[0], cum_rel_b2=cum_rel[1], cum_rel_b3=cum_rel[2], cum_rel_b4=cum_rel[3], cum_rel_b5=cum_rel[4], cum_rel_b6=cum_rel[5],
                     blk_rel_b1=blk_rel[0], blk_rel_b2=blk_rel[1], blk_rel_b3=blk_rel[2], blk_rel_b4=blk_rel[3], blk_rel_b5=blk_rel[4], blk_rel_b6=blk_rel[5]))
out = pd.DataFrame(rows).set_index("advertiser")
out.to_csv(os.path.join(res, "regret_path.csv"))
pd.set_option("display.width", 250)
print("decision epochs per day: distinct multipliers (median, min, max):", out.n_distinct_alpha.median(), out.n_distinct_alpha.min(), out.n_distinct_alpha.max(),
      "; multiplier changes:", out.n_alpha_changes.median(), out.n_alpha_changes.min(), out.n_alpha_changes.max())
print("active auctions per advertiser-day: median", int(out.n_active.median()), "; smallest context-quintile cell:", int(out.min_ctx_count.min()))
print("\ncumulative average relative regret at the end of blocks 1..6 (8 steps each), median by class:")
print(out.groupby("cls")[[f"cum_rel_b{k}" for k in range(1, 7)]].median().round(3).to_string())
print("\nblock-wise relative regret (each block vs its own best fixed multiplier), median by class:")
print(out.groupby("cls")[[f"blk_rel_b{k}" for k in range(1, 7)]].median().round(3).to_string())
print("\nshare of advertisers whose cumulative average regret at the end of the day is below the value at the end of block 1:",
      round(float((out.cum_rel_b6 < out.cum_rel_b1).mean()), 3))
print("\nper advertiser:")
print(out[["cls", "n_distinct_alpha", "n_alpha_changes", "cum_rel_b1", "cum_rel_b3", "cum_rel_b6", "blk_rel_b1", "blk_rel_b6"]].round(3).to_string())
