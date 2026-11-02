"""Design A environment from pooled per-period env files (convert_all.py output), for one focal role.

Contexts: the focal role's predicted values pooled over the listed periods; grid of NV points on the multiplier lattice
(ratio rho^5), masses from a discretized lognormal fitted to the pooled log values (truth = (mu0, sigma0)).
Fringe: the focal role's recorded rival triples pooled over the listed periods (resampled uniformly by the learner).
Payoff table, exposure and payment tables, per-period payoff range K, and the bid lattice as in sim_env.py.
Usage: python sim_env2.py <out_npz> <role> <lambda0> <NV> <env_npz_1> [<env_npz_2> ...]
"""
import sys

import numpy as np
from scipy.stats import norm

E = np.array([1.0, 0.80, 0.48]); RES = 0.01
out, role, lam0, NV = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
# optional grid spec as the 5th argument: NM,MULT_LO,MULT_HI,CTX_STEPS (default 12,0.5,1.5,5)
NM, MULT_LO, MULT_HI, CTX_STEPS = 12, 0.5, 1.5, 5
rest = sys.argv[5:]
if rest and "," in rest[0]:
    NM, MULT_LO, MULT_HI, CTX_STEPS = int(rest[0].split(",")[0]), float(rest[0].split(",")[1]), float(rest[0].split(",")[2]), int(rest[0].split(",")[3])
    rest = rest[1:]
files = rest
v_all, R_all, periods, expo_slots = [], [], [], []
for f in files:
    e = np.load(f)
    ii = int(np.where(e["roles"] == role)[0][0])
    v = e["pvals"][ii].astype(np.float64); r = e["rivals"][ii].astype(np.float64)
    keep = (v > 0) & np.all(np.isfinite(r), axis=1)
    v_all.append(v[keep]); R_all.append(r[keep]); periods.append(int(e["period"])); expo_slots.append(e["expo_by_slot"])
v_src = np.concatenate(v_all); R = np.concatenate(R_all)
expo_meas = np.mean(np.array(expo_slots), axis=0)
rho = (MULT_HI / MULT_LO) ** (1.0 / (NM - 1))
alphas = lam0 * MULT_LO * rho ** np.arange(NM)
rho_v = rho ** CTX_STEPS
logv = np.log(v_src); mu0, sig0 = logv.mean(), logv.std()
vgrid = np.exp(mu0) * rho_v ** (np.arange(NV) - (NV - 1) / 2)
edges = np.concatenate([[-np.inf], (np.log(vgrid[1:]) + np.log(vgrid[:-1])) / 2, [np.inf]])
psi0 = np.diff(norm.cdf(edges, loc=mu0, scale=sig0)); psi0 /= psi0.sum()
emp_hist = np.histogram(logv, bins=edges)[0] / len(logv)
# per-opportunity context bin, aligned with R (same keep mask, same order): the paired environment (run_paired.py) draws an
# opportunity and keeps its own context and rival triple, so the context truth is emp_hist rather than the lognormal psi0
ctx_idx = np.digitize(logv, edges[1:-1]).astype(np.int8)
expo_idx = np.arange(NM)[:, None] + CTX_STEPS * np.arange(NV)[None, :]
bid_index = expo_idx - expo_idx.min(); NB = int(bid_index.max()) + 1


def outcome(b, r):
    r1, r2, r3 = r[:, 0], r[:, 1], r[:, 2]
    s1 = b > r1; s2 = (~s1) & (b > r2); s3 = (~s1) & (~s2) & (b > r3)
    price = np.maximum(np.where(s1, r1, np.where(s2, r2, np.where(s3, r3, 0.0))), RES)
    e = np.where(s1, E[0], np.where(s2, E[1], np.where(s3, E[2], 0.0)))
    return e, price


# use a fixed subsample of rival triples for the tables if the pool is very large (tables are means; 2e6 is plenty)
rng = np.random.default_rng(0)
Rt = R if len(R) <= 2_000_000 else R[rng.choice(len(R), 2_000_000, replace=False)]
ubar = np.zeros((NM, NV)); K = np.zeros(NV); expo = np.zeros((NM, NV)); cost = np.zeros((NM, NV)); winp = np.zeros((NM, NV))
U_all = np.zeros((NM, len(Rt)))
for j, v in enumerate(vgrid):
    for k, a in enumerate(alphas):
        e, p = outcome(a * v, Rt)
        U_all[k] = e * (lam0 * v - p)
        ubar[k, j] = U_all[k].mean(); expo[k, j] = e.mean(); cost[k, j] = (e * p).mean(); winp[k, j] = (e > 0).mean()
    K[j] = float((U_all.max(axis=0) - U_all.min(axis=0)).max())
best = ubar.max(axis=0); kbest = ubar.argmax(axis=0)
np.savez(out, alphas=alphas, vgrid=vgrid, psi0=psi0, emp_hist=emp_hist, mu0=mu0, sig0=sig0, lam0=lam0, R=R.astype(np.float32), ubar=ubar, K=K,
         expo=expo, cost=cost, winp=winp, bid_index=bid_index, NB=NB, rho=rho, src=role, E=E, RES=RES, edges=edges,
         periods=np.array(periods), expo_measured=expo_meas, n_contexts=len(v_src), ctx_steps=CTX_STEPS, mult_lo=MULT_LO, mult_hi=MULT_HI,
         ctx_idx=ctx_idx)
print(f"role {role}: periods {periods}; {len(v_src):,d} contexts, {len(R):,d} rival triples; lambda0 = {lam0}; measured exposure {np.round(expo_meas, 3)}")
print(f"context grid (x1e4): {np.round(vgrid * 1e4, 2)}; lognormal masses {np.round(psi0, 3)}; empirical masses {np.round(emp_hist, 3)}; (mu0, sigma0) = ({mu0:.3f}, {sig0:.3f})")
print("best multiplier / lambda0 by context:", np.round(alphas[kbest] / lam0, 3))
print("best payoff x1e3:", np.round(best * 1e3, 3), " win prob at best:", np.round(winp[kbest, np.arange(NV)], 3))
print("K[v]:", np.round(K, 4), " K/best:", np.round(K / np.maximum(best, 1e-12), 1), " profitable contexts:", int((best > 1e-9).sum()), "of", NV)
