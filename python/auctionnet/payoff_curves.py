"""Counterfactual payoff curves in the bid multiplier for selected advertisers (building block for E1/E2).

For advertiser i and multiplier m, the counterfactual bid on opportunity n is b = m * pValue_n.
Rivals' top-3 bids r1 >= r2 >= r3 are taken from the recorded 48 bids (excluding i).
Slot 1 if b > r1 (pays r1), slot 2 if r2 < b <= r1 (pays r2), slot 3 if r3 < b <= r2 (pays r3); prices floored at 0.01.
Expected payoff per opportunity with value-per-conversion lambda: e_slot * (lambda * pValue - price),
with exposure rates e = (1, 0.80, 0.48) as measured in the data.
Usage: python payoff_curves.py <results_dir> <lambda_rule> [advertiser ids...]
lambda_rule: 'cpa' uses the advertiser's CPA target as value per conversion.
"""
import os
import sys

import numpy as np
import pandas as pd

res, lam_rule = sys.argv[1], sys.argv[2]
ids = [int(x) for x in sys.argv[3:]] or [0, 8, 36]
E = np.array([1.0, 0.80, 0.48])
RES = 0.01
d = pd.read_parquet(os.path.join(res, "period_ranked.parquet"),
                    columns=["advertiserNumber", "CPAConstraint", "timeStepIndex", "oppid", "pValue", "bid", "xi", "adSlot",
                             "cost", "isExposed", "rank"])
top = d[d["rank"] <= 4][["oppid", "rank", "bid", "advertiserNumber"]]
tb = top.pivot(index="oppid", columns="rank", values="bid")
ta = top.pivot(index="oppid", columns="rank", values="advertiserNumber")
tb.columns = [f"b{int(c)}" for c in tb.columns]
ta.columns = [f"a{int(c)}" for c in ta.columns]
mult = np.array([0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0])
rows = []
for i in ids:
    di = d[d.advertiserNumber == i].set_index("oppid").sort_index()
    di = di.join(tb).join(ta)
    lam = di.CPAConstraint.iloc[0] if lam_rule == "cpa" else float(lam_rule)
    # rivals' top-3 bids excluding advertiser i
    B = di[["b1", "b2", "b3", "b4"]].values
    A = di[["a1", "a2", "a3", "a4"]].values
    mask = (A == i)
    Bm = np.where(mask, -1.0, B)                     # drop own bid
    Bs = -np.sort(-Bm, axis=1)[:, :3]                # rivals' top-3
    r1, r2, r3 = Bs[:, 0], Bs[:, 1], Bs[:, 2]
    v = di.pValue.values
    active = di.bid.values > 0
    out = {}
    for m in mult:
        b = m * lam * v
        s1 = b > r1
        s2 = (~s1) & (b > r2)
        s3 = (~s1) & (~s2) & (b > r3)
        price = np.where(s1, r1, np.where(s2, r2, np.where(s3, r3, 0.0)))
        price = np.maximum(price, RES)
        e = np.where(s1, E[0], np.where(s2, E[1], np.where(s3, E[2], 0.0)))
        u = e * (lam * v - price)
        u = np.where(v > 0, u, 0.0)
        out[m] = (u[active].mean(), (s1 | s2 | s3)[active].mean(), (e * price)[active].mean(), (e * lam * v)[active].mean())
    # realized payoff at actual bids
    b = di.bid.values
    s1 = b > r1; s2 = (~s1) & (b > r2); s3 = (~s1) & (~s2) & (b > r3)
    price = np.maximum(np.where(s1, r1, np.where(s2, r2, np.where(s3, r3, 0.0))), RES)
    e = np.where(s1, E[0], np.where(s2, E[1], np.where(s3, E[2], 0.0)))
    u_act = (e * (lam * v - price))[active].mean()
    alpha_act = np.median((b / np.maximum(v, 1e-12))[active & (v > 0)]) / lam
    best_m = max(out, key=lambda k: out[k][0])
    print(f"\nadvertiser {i}: lambda={lam:.0f}, active opps={active.sum()}, actual median multiplier/lambda={alpha_act:.2f}, "
          f"actual mean payoff={u_act:.5f}, best fixed multiplier/lambda={best_m:.2f} with payoff={out[best_m][0]:.5f}, "
          f"regret/best={(out[best_m][0]-u_act)/max(out[best_m][0],1e-12):.3f}")
    print("   m/lambda   payoff    winrate   E[cost]   E[value]")
    for m in mult:
        print(f"   {m:6.2f}  {out[m][0]:9.5f}  {out[m][1]:7.3f}  {out[m][2]:8.4f}  {out[m][3]:8.4f}")
        rows.append(dict(advertiser=i, lam=lam, m=m, payoff=out[m][0], winrate=out[m][1], cost=out[m][2], value=out[m][3],
                         u_actual=u_act, alpha_actual=alpha_act))
pd.DataFrame(rows).to_csv(os.path.join(res, "payoff_curves.csv"), index=False)
