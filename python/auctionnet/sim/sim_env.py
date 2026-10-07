"""Design A environment: AuctionNet-calibrated stage game for one focal bidder against the recorded fringe.

Inputs: period_ranked.parquet (from verify_auction.py). Choices: source advertiser (contexts and rival triples),
value per conversion lambda0, multiplier grid (trimmed), context grid on the multiplier lattice.

Outputs (npz): context grid and masses psi0 (discretized lognormal fitted to the source advertiser's pValues),
rival triples (r1 >= r2 >= r3 among the other 47 bidders, all opportunities), payoff table ubar[alpha, v],
expected exposure expo[alpha, v], expected payment cost[alpha, v], per-period payoff range K[v],
the bid lattice (bid = alpha * v -> lattice index), and the (mu0, sigma0) truth.

Usage: python sim_env.py <results_dir> <out_npz> [source_adv=28] [lambda0=120]
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

E = np.array([1.0, 0.80, 0.48])
RES = 0.01
NM = 12                             # multiplier levels
MULT_LO, MULT_HI = 0.5, 1.5         # trimmed multiplier grid, relative to lambda0

res = sys.argv[1]
out = sys.argv[2]
src = int(sys.argv[3]) if len(sys.argv) > 3 else 28
lam0 = float(sys.argv[4]) if len(sys.argv) > 4 else 120.0
NV = int(sys.argv[5]) if len(sys.argv) > 5 else 7   # context levels (7 spans the 1st to 99th percentile at ratio rho^5)

d = pd.read_parquet(os.path.join(res, "period_ranked.parquet"),
                    columns=["advertiserNumber", "timeStepIndex", "oppid", "pValue", "bid", "rank"])
top = d[d["rank"] <= 4][["oppid", "rank", "bid", "advertiserNumber"]]
tb = top.pivot(index="oppid", columns="rank", values="bid"); tb.columns = [f"b{int(c)}" for c in tb.columns]
ta = top.pivot(index="oppid", columns="rank", values="advertiserNumber"); ta.columns = [f"a{int(c)}" for c in ta.columns]
di = d[d.advertiserNumber == src].set_index("oppid").sort_index().join(tb).join(ta)
B = di[["b1", "b2", "b3", "b4"]].values
A = di[["a1", "a2", "a3", "a4"]].values
R = -np.sort(-np.where(A == src, -1.0, B), axis=1)[:, :3]          # rival triples, all opportunities
R = R[np.all(np.isfinite(R), axis=1)]
v_src = di.pValue.values
v_src = v_src[v_src > 0]

# --- multiplier and context grids on a common log lattice ---
rho = (MULT_HI / MULT_LO) ** (1.0 / (NM - 1))                        # multiplier ratio
alphas = lam0 * MULT_LO * rho ** np.arange(NM)                        # 0.5 lambda0 ... 1.5 lambda0
rho_v = rho ** 5                                                      # context ratio = 5 lattice steps
logv = np.log(v_src)
mu_log, sd_log = logv.mean(), logv.std()
v_c = np.exp(mu_log)                                                  # center context at the geometric mean
vgrid = v_c * rho_v ** (np.arange(NV) - (NV - 1) / 2)                 # 5 levels, ratio rho_v
edges = np.concatenate([[-np.inf], (np.log(vgrid[1:]) + np.log(vgrid[:-1])) / 2, [np.inf]])
mu0, sig0 = mu_log, sd_log
psi0 = np.diff(norm.cdf(edges, loc=mu0, scale=sig0))
psi0 = psi0 / psi0.sum()
expo_idx = (np.arange(NM)[:, None] + 5 * np.arange(NV)[None, :])      # NM x NV lattice exponent
bid_index = expo_idx - expo_idx.min()
NB = int(bid_index.max()) + 1


def outcome(b, r):
    r1, r2, r3 = r[:, 0], r[:, 1], r[:, 2]
    s1 = b > r1
    s2 = (~s1) & (b > r2)
    s3 = (~s1) & (~s2) & (b > r3)
    price = np.maximum(np.where(s1, r1, np.where(s2, r2, np.where(s3, r3, 0.0))), RES)
    e = np.where(s1, E[0], np.where(s2, E[1], np.where(s3, E[2], 0.0)))
    return e, price


ubar = np.zeros((NM, NV)); K = np.zeros(NV); expo = np.zeros((NM, NV)); cost = np.zeros((NM, NV)); winp = np.zeros((NM, NV))
U_all = np.zeros((NM, len(R)))
for j, v in enumerate(vgrid):
    for k, a in enumerate(alphas):
        e, p = outcome(a * v, R)
        U_all[k] = e * (lam0 * v - p)
        ubar[k, j] = U_all[k].mean(); expo[k, j] = e.mean(); cost[k, j] = (e * p).mean(); winp[k, j] = (e > 0).mean()
    K[j] = float((U_all.max(axis=0) - U_all.min(axis=0)).max())
best = ubar.max(axis=0); kbest = ubar.argmax(axis=0)
np.savez(out, alphas=alphas, vgrid=vgrid, psi0=psi0, mu0=mu0, sig0=sig0, lam0=lam0, R=R, ubar=ubar, K=K, expo=expo, cost=cost,
         winp=winp, bid_index=bid_index, NB=NB, rho=rho, src=src, E=E, RES=RES, edges=edges)
print(f"source advertiser {src}: {len(v_src)} contexts, {len(R)} rival triples; lambda0 = {lam0}")
print(f"context grid (x1e4): {np.round(vgrid * 1e4, 3)}; masses psi0: {np.round(psi0, 3)}; (mu0, sigma0) = ({mu0:.3f}, {sig0:.3f}); ratio {rho_v:.3f}")
print(f"multiplier grid / lambda0: {np.round(alphas / lam0, 3)}; ratio {rho:.4f}; distinct bids on lattice: {NB}")
print("payoff table ubar (rows = multipliers 0.5..1.5 lambda0, cols = contexts low..high), x1e3:")
print(np.round(ubar * 1e3, 3))
print("best multiplier / lambda0 by context:", np.round(alphas[kbest] / lam0, 3), " best payoff x1e3:", np.round(best * 1e3, 3))
print("win prob at best:", np.round(winp[kbest, np.arange(NV)], 3), " K[v]:", np.round(K, 4), " K/best:", np.round(K / np.maximum(best, 1e-12), 1))
print("unconditional scale phi*best x1e3:", np.round(psi0 * best * 1e3, 4))
