"""Certified-radius arithmetic for the AuctionNet traces (Theorem CR inputs), per advertiser and pValue quintile.

At lambda* (from regret_lambda.csv), for each context quintile v:
  K(v)      = max over observed states and grid bids of |u(b, theta) - u(b', theta)| (per-period payoff range)
  phi(v)    = context frequency (0.2 by construction of quintiles)
  surplus   = mean realized payoff of the actual bids in the quintile (unconditional scale = phi * surplus)
  regret    = empirical regret in the quintile against the best fixed multiplier
  eps_auct  = Hedge radius of Corollary 1 with N = number of auctions
  eps_batch = Hedge radius with N = number of decision epochs (steps with a bid), batch-average payoffs
with M_R = 5 contexts, alpha_R = 0.045, |A| = 31 grid multipliers.
Usage: python radius_check.py <results_dir>
"""
import os
import sys

import numpy as np
import pandas as pd

res = sys.argv[1]
E = np.array([1.0, 0.80, 0.48])
RES = 0.01
M_R, ALPHA_R, NA = 5, 0.045, 31
d = pd.read_parquet(os.path.join(res, "period_ranked.parquet"),
                    columns=["advertiserNumber", "timeStepIndex", "oppid", "pValue", "bid", "rank"])
reg = pd.read_csv(os.path.join(res, "regret_lambda.csv")).set_index("advertiser")
cls = pd.read_csv(os.path.join(res, "path_classes.csv"), index_col=0)["cls"]
top = d[d["rank"] <= 4][["oppid", "rank", "bid", "advertiserNumber"]]
tb = top.pivot(index="oppid", columns="rank", values="bid"); tb.columns = [f"b{int(c)}" for c in tb.columns]
ta = top.pivot(index="oppid", columns="rank", values="advertiserNumber"); ta.columns = [f"a{int(c)}" for c in ta.columns]
alphas = np.exp(np.linspace(np.log(20), np.log(400), NA))


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
    U = np.zeros((NA, len(v)))
    for j, a in enumerate(alphas):
        e, p = outcome(a * v, r1, r2, r3)
        U[j] = e * (lam * v - p)
    q = np.quantile(v, [0.2, 0.4, 0.6, 0.8]); bucket = np.searchsorted(q, v)
    n_steps = len(np.unique(step))
    for k in range(5):
        m = bucket == k
        Nk = int(m.sum())
        K = float((U[:, m].max(axis=0) - U[:, m].min(axis=0)).max())      # per-period range over grid bids
        surplus = float(u_a[m].mean())
        best = float(U[:, m].mean(axis=1).max())
        regret = max(best - surplus, 0.0)
        phi = Nk / len(v)
        eps_auct = (M_R / ALPHA_R) * K * (np.sqrt(2 * phi * np.log(NA) / len(v)) + np.log(NA) / len(v))
        eps_batch = (M_R / ALPHA_R) * K * (np.sqrt(2 * phi * np.log(NA) / n_steps) + np.log(NA) / n_steps)
        rows.append(dict(advertiser=i, cls=cls.get(i, "?"), quintile=k + 1, N_ctx=Nk, n_steps=n_steps, K=K, surplus=surplus,
                         best_fixed=best, regret=regret, uncond_scale=phi * abs(best), eps_auct=eps_auct, eps_batch=eps_batch,
                         ratio_auct=eps_auct / max(phi * abs(best), 1e-12), ratio_batch=eps_batch / max(phi * abs(best), 1e-12),
                         regret_over_eps_auct=phi * regret / max(eps_auct, 1e-12)))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(res, "radius_check.csv"), index=False)
pd.set_option("display.width", 250)
print("K / |best fixed payoff| per quintile (median over advertisers):")
print(out.assign(k_over_best=out.K / out.best_fixed.abs().clip(lower=1e-12)).groupby("quintile").k_over_best.median().round(1).to_string())
print("\nradius over unconditional payoff scale, median over advertisers (auction-count N vs decision-epoch N):")
print(out.groupby("quintile")[["ratio_auct", "ratio_batch"]].median().round(1).to_string())
print("\nsame, low-regret classes only (constant, capped):")
print(out[out.cls.isin(["constant", "capped 1.5xCPA"])].groupby("quintile")[["ratio_auct", "ratio_batch"]].median().round(1).to_string())
print("\nempirical regret (unconditional) over the auction-count radius, median by class and quintile:")
print(out.pivot_table(index="cls", columns="quintile", values="regret_over_eps_auct", aggfunc="median").round(3).to_string())
print("\nexample rows (advertiser 41, constant; advertiser 4, capped; advertiser 0, PID-like):")
print(out[out.advertiser.isin([41, 4, 0])][["advertiser", "cls", "quintile", "N_ctx", "n_steps", "K", "best_fixed", "regret", "eps_auct", "eps_batch"]].round(5).to_string())
