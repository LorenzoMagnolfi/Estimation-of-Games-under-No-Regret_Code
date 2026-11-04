"""Exact support extrema and mean tables for a Design A environment (findings I1, I2, I3 of Niccolò's feedback of 4 November 2026).

For each context j (value v_j on the lattice) and every multiplier k, over the FULL stored rival pool R (the pool sim_run.py samples):
  - K_env[j]        the stored span (computed by sim_env2.py on a 2,000,000-row subsample);
  - K_full[j]       the exact largest within-rival cross-multiplier payoff span over the full pool;
  - umax[j]         the learner's upper shift, E1 * (lam0 v_j - RES), as used in sim_run.py (exp3 / exp3ix);
  - g_max[j], g_min[j]   exact global payoff extrema over every multiplier and every rival triple;
  - loss_max[j]     the largest normalized loss (umax - g) / K_env the learner could have formed, and the share of
                    (multiplier, rival) pairs with loss above one; the same with K_full in place of K_env;
  - the exact unconditional tables (ubar, expo, cost) over the full pool against the stored subsample tables;
  - if the environment carries ctx_idx (paired mode): the tables conditional on the context bin, i.e. over the rivals of
    the opportunities whose context is j, against the unconditional ones, for the context's own value v_j.
Writes a JSON with everything and prints a summary. Chunked; no learner is run.
Usage: python audit_env_role.py <env_npz> <out_json>
"""
import json
import sys

import numpy as np

env = np.load(sys.argv[1]); out = sys.argv[2]
alphas, vgrid, lam0, R, K_env, E, RES = env["alphas"], env["vgrid"], float(env["lam0"]), env["R"], env["K"], env["E"], float(env["RES"])
expo_env, cost_env, ubar_env = env["expo"], env["cost"], env["ubar"]
NM, NV, NR = len(alphas), len(vgrid), len(R)
ctx_idx = env["ctx_idx"] if "ctx_idx" in env.files else None
umax = np.maximum(E[0] * (lam0 * vgrid - RES), 0.0)       # sim_run.py: top slot at the reserve, floored at zero
CH = 500_000

K_full = np.zeros(NV); g_max = np.full(NV, -np.inf); g_min = np.full(NV, np.inf)
n_loss_gt1_env = np.zeros(NV); n_loss_gt1_full = np.zeros(NV); loss_max_env = np.zeros(NV); loss_max_full = np.zeros(NV)
S_u = np.zeros((NM, NV)); S_e = np.zeros((NM, NV)); S_c = np.zeros((NM, NV))
Sc_u = np.zeros((NM, NV)); Sc_e = np.zeros((NM, NV)); Sc_c = np.zeros((NM, NV)); n_c = np.zeros(NV)
# pass 1: full-pool spans, extrema, unconditional and conditional sums (K_full needed for the loss share with K_full: second pass)
for start in range(0, NR, CH):
    r = R[start:start + CH].astype(np.float64); n = len(r)
    r1, r2, r3 = r[:, 0][None, :], r[:, 1][None, :], r[:, 2][None, :]
    cidx = ctx_idx[start:start + CH] if ctx_idx is not None else None
    for j, v in enumerate(vgrid):
        b = alphas[:, None] * v
        s1 = b > r1; s2 = (~s1) & (b > r2); s3 = (~s1) & (~s2) & (b > r3)
        price = np.maximum(np.where(s1, r1, np.where(s2, r2, np.where(s3, r3, 0.0))), RES)
        e = np.where(s1, E[0], np.where(s2, E[1], np.where(s3, E[2], 0.0)))
        U = e * (lam0 * v - price)                               # NM x n
        K_full[j] = max(K_full[j], float((U.max(axis=0) - U.min(axis=0)).max()))
        g_max[j] = max(g_max[j], float(U.max())); g_min[j] = min(g_min[j], float(U.min()))
        loss_env = (umax[j] - U) / max(K_env[j], 1e-12)
        loss_max_env[j] = max(loss_max_env[j], float(loss_env.max())); n_loss_gt1_env[j] += float((loss_env > 1.0 + 1e-12).sum())
        S_u[:, j] += U.sum(axis=1); S_e[:, j] += e.sum(axis=1); S_c[:, j] += (e * price).sum(axis=1)
        if cidx is not None:
            m = cidx == j
            if m.any():
                Sc_u[:, j] += U[:, m].sum(axis=1); Sc_e[:, j] += e[:, m].sum(axis=1); Sc_c[:, j] += (e * price)[:, m].sum(axis=1); n_c[j] += m.sum()
# pass 2: loss share with the exact full-pool span
for start in range(0, NR, CH):
    r = R[start:start + CH].astype(np.float64)
    r1, r2, r3 = r[:, 0][None, :], r[:, 1][None, :], r[:, 2][None, :]
    for j, v in enumerate(vgrid):
        b = alphas[:, None] * v
        s1 = b > r1; s2 = (~s1) & (b > r2); s3 = (~s1) & (~s2) & (b > r3)
        price = np.maximum(np.where(s1, r1, np.where(s2, r2, np.where(s3, r3, 0.0))), RES)
        e = np.where(s1, E[0], np.where(s2, E[1], np.where(s3, E[2], 0.0)))
        U = e * (lam0 * v - price)
        loss_full = (umax[j] - U) / max(K_full[j], 1e-12)
        loss_max_full[j] = max(loss_max_full[j], float(loss_full.max())); n_loss_gt1_full[j] += float((loss_full > 1.0 + 1e-12).sum())
ubar_full, expo_full, cost_full = S_u / NR, S_e / NR, S_c / NR
res = dict(env=sys.argv[1], NR=int(NR), lam0=lam0, vgrid=vgrid.tolist(), alphas=alphas.tolist(),
           K_env=K_env.tolist(), K_full=K_full.tolist(), umax=umax.tolist(), g_max=g_max.tolist(), g_min=g_min.tolist(),
           width_exact=(umax - g_min).tolist(), width_over_K_env=((umax - g_min) / np.maximum(K_env, 1e-12)).tolist(),
           loss_max_env=loss_max_env.tolist(), share_loss_gt1_env=(n_loss_gt1_env / (NM * NR)).tolist(),
           loss_max_full=loss_max_full.tolist(), share_loss_gt1_full=(n_loss_gt1_full / (NM * NR)).tolist(),
           ubar_full=ubar_full.tolist(), expo_full=expo_full.tolist(), cost_full=cost_full.tolist(),
           ubar_env=ubar_env.tolist(), expo_env=expo_env.tolist(), cost_env=cost_env.tolist(),
           table_maxabs_diff=dict(ubar=float(np.abs(ubar_full - ubar_env).max()), expo=float(np.abs(expo_full - expo_env).max()), cost=float(np.abs(cost_full - cost_env).max())),
           table_maxrel_diff=dict(ubar=float((np.abs(ubar_full - ubar_env) / np.maximum(np.abs(ubar_full), 1e-9)).max()),
                                  expo=float((np.abs(expo_full - expo_env) / np.maximum(expo_full, 1e-9)).max()),
                                  cost=float((np.abs(cost_full - cost_env) / np.maximum(cost_full, 1e-9)).max())))
if ctx_idx is not None:
    nc = np.maximum(n_c, 1.0)
    ubar_c, expo_c, cost_c = Sc_u / nc, Sc_e / nc, Sc_c / nc
    res.update(n_context=n_c.tolist(), ubar_cond=ubar_c.tolist(), expo_cond=expo_c.tolist(), cost_cond=cost_c.tolist(),
               cond_vs_uncond_maxabs=dict(ubar=float(np.abs(ubar_c - ubar_full).max()), expo=float(np.abs(expo_c - expo_full).max()), cost=float(np.abs(cost_c - cost_full).max())),
               cond_vs_uncond_maxrel=dict(ubar=float((np.abs(ubar_c - ubar_full) / np.maximum(np.abs(ubar_full), 1e-9)).max()),
                                          expo=float((np.abs(expo_c - expo_full) / np.maximum(expo_full, 1e-9)).max()),
                                          cost=float((np.abs(cost_c - cost_full) / np.maximum(cost_full, 1e-9)).max())))
with open(out, "w") as fh:
    json.dump(res, fh, indent=1)
np.set_printoptions(precision=4, suppress=True, linewidth=160)
print(f"{sys.argv[1]}: {NR:,d} rival triples, lam0 {lam0}")
print("K_env        ", np.array(K_env)); print("K_full       ", K_full); print("umax (shift) ", umax); print("g_min        ", g_min); print("g_max        ", g_max)
print("exact width umax-g_min / K_env:", (umax - g_min) / np.maximum(K_env, 1e-12))
print("loss_max with K_env:", loss_max_env, " share of (mult, rival) pairs with loss > 1:", n_loss_gt1_env / (NM * NR))
print("loss_max with K_full:", loss_max_full, " share > 1:", n_loss_gt1_full / (NM * NR))
print("tables full vs stored: max abs", res["table_maxabs_diff"], " max rel", res["table_maxrel_diff"])
if ctx_idx is not None:
    print("paired: opportunities per context", n_c)
    print("conditional vs unconditional tables: max abs", res["cond_vs_uncond_maxabs"], " max rel", res["cond_vs_uncond_maxrel"])
    print("expo_cond / expo_full by (mult, ctx):"); print(np.round(expo_c / np.maximum(expo_full, 1e-12), 3))
print("AUDIT_ENV_DONE")
