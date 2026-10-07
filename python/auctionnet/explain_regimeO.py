"""Diagnostics that explain Regime O (contexts latent, lambda0 known) on the Design A runs.

For one environment (payoff table without the rival draws, as extracted to results/designA/tables/) and one Hedge run:
  1. per context j: best expected payoff, the worst regret available on the multiplier grid (per auction in context j),
     and the Markov tolerance per auction in context j at each checkpoint (eps_j / phi_j, eps_j as in sim_infer.py);
  2. for every candidate (mu, sigma) on the 41 x 38 grid of sim_infer.py and each requested checkpoint N, the smallest
     multiple kappa of the Markov tolerance profile at which the candidate is feasible:
        min kappa  s.t.  nu >= 0, bid marginal of nu = q_N, L1 consistency ball around psi(mu, sigma) of radius r_cons(N),
                         sum_k nu(k, j) [u(k', j) - u(k, j)] <= kappa * eps_j(N)  for all j, k'.
     kappa = inf means the candidate is infeasible even without obedience (the 'none' baseline); the Markov region of
     sim_infer.py is {kappa <= 1}; smaller kappa shows what a tighter tolerance would remove.
  3. the same Markov region with a candidate-specific tolerance, eps_j computed from the candidate's context masses
     psi_j(mu, sigma) instead of the realized shares phi_j, which the analyst does not observe when contexts are latent.
Writes <out>/<tag>_tolerance.csv, <tag>_kappa_<N>.npy, <tag>_candphi_<N>.npy, <tag>_summary.csv.
Profile (6th argument): 'realized' (default; the behavior of commit 8348836) scales the Markov profile computed from the
realized context shares, and item 3 uses the candidate's shares psi_j exactly. 'candidate' scales, for each candidate, the
corrected profile of sim_infer.py (since 2026-10-07), eps_j(N, psi_j(mu, sigma) + r_cons / 2), so {kappa <= 1} is the
corrected Markov region; item 3 is then replaced by the old rule (realized shares at kappa = 1), saved as
<tag>_markovrealized_<N>.npy and reported as 'markov_realized'.
Usage: python explain_regimeO.py <table_npz> <run_npz> <out_dir> <tag> [checkpoints, default 1e7,1e8] [profile=realized]
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import norm

tab, run, out, tag = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3], sys.argv[4]
checks = [int(float(x)) for x in (sys.argv[5] if len(sys.argv) > 5 else "1e7,1e8").split(",")]
profile = sys.argv[6] if len(sys.argv) > 6 else "realized"
assert profile in ("realized", "candidate"), profile
os.makedirs(out, exist_ok=True)
alphas, vgrid, psi0, lam0, K = tab["alphas"], tab["vgrid"], tab["psi0"], float(tab["lam0"]), tab["K"]
expo, cost, bid_index, NB, edges = tab["expo"], tab["cost"], tab["bid_index"], int(tab["NB"]), tab["edges"]
mu0, sig0 = float(tab["mu0"]), float(tab["sig0"])
NM, NV = len(alphas), len(vgrid)
ALPHA_R, ALPHA_C, M_R, lnA = 0.045, 0.005, NV, np.log(NM)
u = lam0 * expo * vgrid[None, :] - cost                      # u[k, j], population payoff table
du = u[:, None, :] - u[None, :, :]                            # du[kp, k, j] = u(kp, j) - u(k, j)
mus = mu0 + np.linspace(-1.0, 1.0, 41); sigs = np.linspace(0.15, 2.0, 38)   # the grid of sim_infer.py


def psi_of(mu, sig):
    p = np.diff(norm.cdf(edges, loc=mu, scale=sig)); return p / p.sum()


def eps_markov(N, phi):
    return (M_R / (ALPHA_R * N)) * K * (np.sqrt(2 * N * phi * lnA) + lnA)


nnu = NM * NV
idx = lambda k, j: k * NV + j
nvar = nnu + NV + 1                                           # nu, d (L1 slacks), kappa
A_eq = np.zeros((NB, nvar))
for k in range(NM):
    for j in range(NV):
        A_eq[bid_index[k, j], idx(k, j)] = 1.0
cons_rows = []
for j in range(NV):
    r = np.zeros(nvar); r[[idx(k, j) for k in range(NM)]] = 1.0; r[nnu + j] = -1.0; cons_rows.append(r)
    r = np.zeros(nvar); r[[idx(k, j) for k in range(NM)]] = -1.0; r[nnu + j] = -1.0; cons_rows.append(r)
r = np.zeros(nvar); r[nnu:nnu + NV] = 1.0; cons_rows.append(r)
cons_rows = np.array(cons_rows)
obed_base = np.zeros((NV * NM, nvar))
for j in range(NV):
    for kp in range(NM):
        for k in range(NM):
            obed_base[j * NM + kp, idx(k, j)] = du[kp, k, j]


def min_kappa(q, psi, eps, r_cons):
    A_ob = obed_base.copy(); A_ob[:, -1] = -np.repeat(eps, NM)
    A_ub = np.vstack([cons_rows, A_ob])
    b_ub = np.concatenate([np.repeat(psi, 2) * np.tile([1.0, -1.0], NV), [r_cons], np.zeros(NV * NM)])
    c = np.zeros(nvar); c[-1] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=q, bounds=[(0, None)] * nvar, method="highs")
    return res.fun if res.status == 0 else np.inf


def feasible(q, psi, eps, r_cons):
    A_ob = obed_base.copy(); A_ob[:, -1] = 0.0
    A_ub = np.vstack([cons_rows, A_ob])
    b_ub = np.concatenate([np.repeat(psi, 2) * np.tile([1.0, -1.0], NV), [r_cons], np.repeat(eps, NM)])
    res = linprog(np.zeros(nvar), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=q, bounds=[(0, None)] * nvar, method="highs")
    return res.status == 0


# 1. tolerance versus available regret, per context
best = u.max(axis=0); worst_regret = best - u.min(axis=0)
rows = []
for N in [int(x) for x in run["checks"]]:
    phi = run[f"Z_{N}"].sum(axis=0) / N
    tol_cond = eps_markov(N, phi) / np.maximum(phi, 1e-300)
    for j in range(NV):
        rows.append(dict(N=N, context=j + 1, v=vgrid[j], phi=phi[j], best_payoff=best[j], best_mult_over_lam=alphas[u[:, j].argmax()] / lam0,
                         worst_regret=worst_regret[j], K=K[j], tol_markov_cond=tol_cond[j], ratio_worst_to_tol=worst_regret[j] / tol_cond[j] if tol_cond[j] > 0 else np.nan))
pd.DataFrame(rows).to_csv(os.path.join(out, f"{tag}_tolerance.csv"), index=False)

# 2-3. kappa maps and candidate-specific tolerance
summ = []
for N in checks:
    Z = run[f"Z_{N}"]; phi = Z.sum(axis=0) / N
    q = np.zeros(NB)
    for k in range(NM):
        for j in range(NV):
            q[bid_index[k, j]] += Z[k, j] / N
    r_cons = np.sqrt(2 * np.log((2 ** NV - 2) / ALPHA_C) / N)
    eps = eps_markov(N, phi)
    if profile == "candidate":
        def epsc(psi):
            return eps_markov(N, np.minimum(1.0, psi + r_cons / 2))
        kap = np.array([[min_kappa(q, psi_of(mu, sg), epsc(psi_of(mu, sg)), r_cons) for sg in sigs] for mu in mus])
        candphi = np.array([[feasible(q, psi_of(mu, sg), eps, r_cons) for sg in sigs] for mu in mus])      # old rule
        np.save(os.path.join(out, f"{tag}_kappa_{N}.npy"), kap); np.save(os.path.join(out, f"{tag}_markovrealized_{N}.npy"), candphi)
        k_truth = min_kappa(q, psi0, epsc(psi0), r_cons)
        other_name, other_truth = "markov_realized", bool(feasible(q, psi0, eps, r_cons))
    else:
        kap = np.array([[min_kappa(q, psi_of(mu, sg), eps, r_cons) for sg in sigs] for mu in mus])
        candphi = np.array([[feasible(q, psi_of(mu, sg), eps_markov(N, psi_of(mu, sg)), r_cons) for sg in sigs] for mu in mus])
        np.save(os.path.join(out, f"{tag}_kappa_{N}.npy"), kap); np.save(os.path.join(out, f"{tag}_candphi_{N}.npy"), candphi)
        k_truth = min_kappa(q, psi0, eps, r_cons)
        other_name, other_truth = "markov_candidate_phi", bool(feasible(q, psi0, eps_markov(N, psi0), r_cons))
    none = np.isfinite(kap)
    for thr in [np.inf, 10.0, 3.16, 1.0, 1 / 22.2, 0.01]:
        reg = np.isfinite(kap) if np.isinf(thr) else (kap <= thr)      # kappa = inf marks candidates infeasible without obedience
        summ.append(dict(N=N, kappa_threshold=thr, share=reg.mean(), share_of_none=reg.sum() / max(none.sum(), 1), truth_kappa=k_truth,
                         truth_in=bool(k_truth <= thr)))
    summ.append(dict(N=N, kappa_threshold=other_name, share=candphi.mean(), share_of_none=candphi.sum() / max(none.sum(), 1),
                     truth_kappa=k_truth, truth_in=other_truth))
    print(f"{tag} [{profile}] N={N:,d}: none {none.mean():.3f}; markov {np.mean(kap <= 1):.3f}; {other_name} {candphi.mean():.3f}; "
          f"kappa<=1/22 {np.mean(kap <= 1 / 22.2):.3f}; truth kappa {k_truth:.4f}", flush=True)
pd.DataFrame(summ).to_csv(os.path.join(out, f"{tag}_summary.csv"), index=False)
np.savez(os.path.join(out, f"{tag}_grid.npz"), mus=mus, sigs=sigs, mu0=mu0, sig0=sig0)
