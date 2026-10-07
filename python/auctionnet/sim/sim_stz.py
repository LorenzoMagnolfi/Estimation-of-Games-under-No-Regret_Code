"""STZ-style comparison on the simulated play (Regime O, contexts latent, lambda0 known).

Emulates Syrgkanis, Tamer, and Ziani (2021, Sections 5.2.2 and 7.4): for each of several independent runs at a smaller horizon
(the 'subsamples'), compute the minimal scalar tolerance tau at which the identification LP is feasible for some candidate
(mu, sigma) on a coarse grid; take the 95 percent quantile of these minima as the tolerance; the estimated set is the set of
candidates feasible at that tolerance on the main run. The tolerance relaxes every obedience inequality and the consistency
equalities by tau. Two obedience types: 'bce' (STZ: deviation conditional on the recommended multiplier, pointwise in
(alpha, v)) and 'bcce' (ours: fixed deviation per context). Also reports the BCCE set at STZ's tolerance next to the
Markov-radius set from sim_infer for the same run.
Usage: python sim_stz.py <env_npz> <main_run_npz> <N_main> <coverage_dir> <N_sub> <out_prefix>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import norm

env = np.load(sys.argv[1]); main_run = np.load(sys.argv[2]); N_main = int(float(sys.argv[3]))
cov_dir = sys.argv[4]; N_sub = int(float(sys.argv[5])); pref = sys.argv[6]
alphas, vgrid, psi0, lam0, expo, cost, bid_index, NB = (env["alphas"], env["vgrid"], env["psi0"], float(env["lam0"]), env["expo"], env["cost"],
                                                      env["bid_index"], int(env["NB"]))
mu0, sig0, edges = float(env["mu0"]), float(env["sig0"]), env["edges"]
NM, NV = len(alphas), len(vgrid)
a_kj = expo * vgrid[None, :]; c_kj = cost
du = lam0 * (a_kj[:, None, :] - a_kj[None, :, :]) - (c_kj[:, None, :] - c_kj[None, :, :])   # du[kp, k, j] = u(kp,j) - u(k,j)
idx = lambda k, j: k * NV + j
nnu = NM * NV
mus = mu0 + np.linspace(-1.0, 1.0, 21); sigs = np.round(np.linspace(0.2, 2.0, 19), 3)


def psi_of(mu, sig):
    p = np.diff(norm.cdf(edges, loc=mu, scale=sig)); return p / p.sum()


def qN_of(Z, N):
    q = np.zeros(NB)
    for k in range(NM):
        for j in range(NV):
            q[bid_index[k, j]] += Z[k, j] / N
    return q


def matrices(obed):
    """Variables: nu (nnu), tau (1). Returns A_eq (bid marginal), A_ub rows for consistency (+-) and obedience, each with -tau."""
    A_eq = np.zeros((NB, nnu + 1))
    for k in range(NM):
        for j in range(NV):
            A_eq[bid_index[k, j], idx(k, j)] = 1.0
    rows = []
    for j in range(NV):                                   # m_j - psi_j <= tau ; psi_j - m_j <= tau
        r = np.zeros(nnu + 1); r[[idx(k, j) for k in range(NM)]] = 1.0; r[-1] = -1.0; rows.append(r)
        r = np.zeros(nnu + 1); r[[idx(k, j) for k in range(NM)]] = -1.0; r[-1] = -1.0; rows.append(r)
    if obed == "bcce":
        for j in range(NV):
            for kp in range(NM):
                r = np.zeros(nnu + 1)
                for k in range(NM):
                    r[idx(k, j)] = du[kp, k, j]
                r[-1] = -1.0; rows.append(r)
    else:                                                 # bce: pointwise in the recommended multiplier
        for j in range(NV):
            for k in range(NM):
                for kp in range(NM):
                    if kp == k:
                        continue
                    r = np.zeros(nnu + 1); r[idx(k, j)] = du[kp, k, j]; r[-1] = -1.0; rows.append(r)
    return A_eq, np.array(rows)


MATS = {o: matrices(o) for o in ["bce", "bcce"]}


def min_tau(qN, psi, obed):
    A_eq, A_ub = MATS[obed]
    b_ub = np.zeros(A_ub.shape[0]); b_ub[:2 * NV] = np.repeat(psi, 2) * np.tile([1.0, -1.0], NV)
    c = np.zeros(nnu + 1); c[-1] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=qN, bounds=[(0, None)] * (nnu + 1), method="highs")
    return res.fun if res.status == 0 else np.inf


def feasible_at(qN, psi, obed, tau):
    A_eq, A_ub = MATS[obed]
    A = A_ub[:, :nnu]; b = np.zeros(A_ub.shape[0]); b[:2 * NV] = np.repeat(psi, 2) * np.tile([1.0, -1.0], NV); b = b + tau
    res = linprog(np.zeros(nnu), A_ub=A, b_ub=b, A_eq=A_eq[:, :nnu], b_eq=qN, bounds=[(0, None)] * nnu, method="highs")
    return res.status == 0


rows = []
sub_runs = sorted(glob.glob(os.path.join(cov_dir, "run_*.npz")))
print(f"{len(sub_runs)} subsample runs at N = {N_sub:,d}; main run at N = {N_main:,d}")
for obed in ["bce", "bcce"]:
    taus = []
    for f in sub_runs:
        r = np.load(f); Z = r[f"Z_{N_sub}"]; q = qN_of(Z, N_sub)
        tmin = min(min_tau(q, psi_of(mu, sg), obed) for mu in mus for sg in sigs)
        taus.append(tmin)
    taus = np.array(taus); tau_star = float(np.quantile(taus, 0.95))
    qM = qN_of(main_run[f"Z_{N_main}"], N_main)
    musF = mu0 + np.linspace(-1.0, 1.0, 41); sigsF = np.linspace(0.15, 2.0, 38)          # fine grid for the main-run set
    feas = np.array([[feasible_at(qM, psi_of(mu, sg), obed, tau_star) for sg in sigsF] for mu in musF])
    truth_ok = feasible_at(qM, psi0, obed, tau_star)
    mu_r = (musF[feas.any(axis=1)].min(), musF[feas.any(axis=1)].max()) if feas.any() else (np.nan, np.nan)
    sg_r = (sigsF[feas.any(axis=0)].min(), sigsF[feas.any(axis=0)].max()) if feas.any() else (np.nan, np.nan)
    np.save(f"{pref}_{obed}_feas.npy", feas)
    rows.append(dict(obedience=obed, tau_star=tau_star, tau_min_median=float(np.median(taus)), share=float(feas.mean()), mu_lo=mu_r[0], mu_hi=mu_r[1],
                     sig_lo=sg_r[0], sig_hi=sg_r[1], truth_feasible=truth_ok, N_main=N_main, N_sub=N_sub, n_sub=len(sub_runs)))
    print(f"{obed}: tau* (95% quantile of subsample minima) = {tau_star:.3e} (median {np.median(taus):.3e}); estimated set share {feas.mean():.3f}; "
          f"mu in [{mu_r[0]:.2f},{mu_r[1]:.2f}] (truth {mu0:.2f}); sigma in [{sg_r[0]:.2f},{sg_r[1]:.2f}] (truth {sig0:.2f}); truth feasible {truth_ok}")
pd.DataFrame(rows).to_csv(f"{pref}_stz.csv", index=False)
np.savez(f"{pref}_grid.npz", mus=mu0 + np.linspace(-1.0, 1.0, 41), sigs=np.linspace(0.15, 2.0, 38))
