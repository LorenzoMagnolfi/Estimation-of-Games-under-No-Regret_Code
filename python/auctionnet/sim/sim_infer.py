"""Design A inference on simulated play (sample-based obedience where contexts are observed).

Regime P (contexts observed): for each checkpoint N and slack rule, the set of lambda such that the recorded play satisfies the
relaxed coarse-obedience inequalities context by context, using the SAMPLE counterfactual sums recorded by the learner:
for context j and deviation k', lambda * (Sa[j,k'] - Ra[j]) - (Sc[j,k'] - Rc[j]) <= N * eps_j. Each pair gives a half-line in
lambda (payoffs are affine in lambda); intersect over deviations and contexts.
Regime O (contexts latent, lambda0 known): for each candidate (mu, sigma) of the discretized-lognormal context distribution, a
feasibility LP in nu(alpha, v) >= 0 with the bid-lattice marginal equal to q_N, an L1 consistency ball around psi(mu, sigma), and
relaxed obedience with the POPULATION payoff table (first-stage object, exact here because the fringe distribution is known).
Slack rules: exact (0); markov (Theorem CR with the Hedge envelope, M_R = NV, alpha_R = 0.045, alpha_C = 0.005);
empirical (the realized unconditional regret per context at lambda0, a diagnostic that contains the truth by construction).
The Markov tolerance of context j grows with the share phi_j of auctions in that context. In Regime P the contexts are
observed and phi_j is the realized share. In Regime O they are latent, so for a candidate (mu, sigma) the tolerance uses
the largest share consistent with the candidate on the consistency event, phi_j <= psi_j(mu, sigma) + r_cons / 2 (an L1
ball of radius r_cons moves a single coordinate by at most r_cons / 2); this is rule 'markov' in Regime O since 2026-10-07.
Rule 'markov_realized' (Regime O only) keeps the earlier tolerance with the realized shares, which the analyst does not
observe in that regime; it is reported as a diagnostic for comparison with the runs of 2026-09-23/24.
Usage: python sim_infer.py <env_npz> <run_npz> <out_prefix> [regimes=PO] [coarse] [checkpoints]
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import norm

env = np.load(sys.argv[1]); run = np.load(sys.argv[2]); pref = sys.argv[3]
regimes = sys.argv[4] if len(sys.argv) > 4 else "PO"
alphas, vgrid, psi0, lam0, K, expo, cost, bid_index, NB = (env["alphas"], env["vgrid"], env["psi0"], float(env["lam0"]), env["K"],
                                                          env["expo"], env["cost"], env["bid_index"], int(env["NB"]))
mu0, sig0, edges = float(env["mu0"]), float(env["sig0"]), env["edges"]
NM, NV = len(alphas), len(vgrid)
ALPHA_R, ALPHA_C = 0.045, 0.005
M_R = NV
lnA = np.log(NM)
a_kj = expo * vgrid[None, :]
c_kj = cost
checks = [int(x) for x in run["checks"]]
if len(sys.argv) > 6:                      # optional comma-separated subset of checkpoints
    want = set(int(float(x)) for x in sys.argv[6].split(","))
    checks = [c for c in checks if c in want]
TOL = 1e-12


LEARNER = str(run["learner"]) if "learner" in run.files else "hedge"


def markov_eps(N, phi):
    """Markov tolerance per auction (unconditional) for context shares phi."""
    if LEARNER == "exp3":          # Corollary 2 (bandit feedback): 2 sqrt(e-1) K sqrt(m |A| ln|A|)
        return (M_R / (ALPHA_R * N)) * 2 * np.sqrt(np.e - 1) * K * np.sqrt(N * phi * NM * lnA)
    return (M_R / (ALPHA_R * N)) * K * (np.sqrt(2 * N * phi * lnA) + lnA)


def slack(N, rule, Z, Sa, Sc, Ra, Rc):
    phi = Z.sum(axis=0) / N
    if rule == "exact":
        return np.zeros(NV)
    if rule == "none":             # no obedience restriction: the set identified by the bid marginal and consistency alone
        return np.full(NV, 1e6)
    if rule in ("markov", "markov_realized"):   # realized shares (legitimate in Regime P; diagnostic in Regime O)
        return markov_eps(N, phi)
    if rule == "empirical":
        S = lam0 * Sa - Sc; real = lam0 * Ra - Rc
        return np.maximum(S.max(axis=1) - real, 0.0) / N
    raise ValueError(rule)


def regime_P(N, Sa, Sc, Ra, Rc, eps):
    lo, hi = np.full(NV, -np.inf), np.full(NV, np.inf)
    for j in range(NV):
        for kp in range(NM):
            A = (Sa[j, kp] - Ra[j]) / N; C = (Sc[j, kp] - Rc[j]) / N
            if A > TOL:
                hi[j] = min(hi[j], (eps[j] + C) / A)
            elif A < -TOL:
                lo[j] = max(lo[j], (eps[j] + C) / A)
            elif C > eps[j] + TOL:
                lo[j], hi[j] = np.inf, -np.inf
    return lo, hi, max(lo), min(hi)


def psi_of(mu, sig):
    p = np.diff(norm.cdf(edges, loc=mu, scale=sig)); return p / p.sum()


def build_O_matrices():
    nvar = NM * NV + NV
    idx = lambda k, j: k * NV + j
    A_eq = np.zeros((NB, nvar))
    for k in range(NM):
        for j in range(NV):
            A_eq[bid_index[k, j], idx(k, j)] = 1.0
    A_ub = []
    for j in range(NV):
        row = np.zeros(nvar); row[[idx(k, j) for k in range(NM)]] = 1.0; row[NM * NV + j] = -1.0; A_ub.append(row)
        row = np.zeros(nvar); row[[idx(k, j) for k in range(NM)]] = -1.0; row[NM * NV + j] = -1.0; A_ub.append(row)
    row = np.zeros(nvar); row[NM * NV:] = 1.0; A_ub.append(row)
    for j in range(NV):
        for kp in range(NM):
            row = np.zeros(nvar)
            for k in range(NM):
                row[idx(k, j)] = lam0 * (a_kj[kp, j] - a_kj[k, j]) - (c_kj[kp, j] - c_kj[k, j])
            A_ub.append(row)
    return A_eq, np.array(A_ub), nvar


A_eq, A_ub, nvar = build_O_matrices()


def regime_O_feasible(qN, psi, eps, r_cons):
    b_ub = np.concatenate([np.repeat(psi, 2) * np.tile([1.0, -1.0], NV), [r_cons], np.repeat(eps, NM)])
    res = linprog(np.zeros(nvar), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=qN, bounds=[(0, None)] * nvar, method="highs")
    return res.status == 0


rowsP, rowsO = [], []
coarse = len(sys.argv) > 5 and sys.argv[5] == "coarse"
mus = mu0 + (np.linspace(-1.0, 1.0, 21) if coarse else np.linspace(-1.0, 1.0, 41))
sigs = np.linspace(0.15, 2.0, 38) if not coarse else np.round(np.linspace(0.2, 2.0, 19), 3)
for N in checks:
    Z, Sa, Sc, Ra, Rc = run[f"Z_{N}"], run[f"Sa_{N}"], run[f"Sc_{N}"], run[f"Ra_{N}"], run[f"Rc_{N}"]
    qN = np.zeros(NB)
    for k in range(NM):
        for j in range(NV):
            qN[bid_index[k, j]] += Z[k, j] / N
    r_cons = np.sqrt(2 * np.log((2 ** NV - 2) / ALPHA_C) / N)
    rules = ["exact", "markov", "empirical"] + (["none", "markov_realized"] if "O" in regimes else [])
    for rule in rules:
        eps = slack(N, rule, Z, Sa, Sc, Ra, Rc)
        if "P" in regimes and rule != "markov_realized":     # in Regime P the realized shares are observed: 'markov' already uses them
            lo, hi, L, H = regime_P(N, Sa, Sc, Ra, Rc, eps)
            rowsP.append(dict(N=N, rule=rule, lam_lo=L, lam_hi=H, width=H - L, contains_truth=(L - 1e-6 <= lam0 <= H + 1e-6),
                              **{f"lo_v{j + 1}": lo[j] for j in range(NV)}, **{f"hi_v{j + 1}": hi[j] for j in range(NV)},
                              **{f"eps_v{j + 1}": eps[j] for j in range(NV)}))
            print(f"P  N={N:>11,d} {rule:9s} lambda in [{L:9.2f}, {H:9.2f}] (truth {lam0:.0f}, {'in' if L - 1e-6 <= lam0 <= H + 1e-6 else 'OUT'}) "
                  f"per-context {[f'[{lo[j]:.0f},{hi[j]:.0f}]' for j in range(NV)]}", flush=True)
        if "O" in regimes:
            rc = 0.0 if rule == "exact" else r_cons

            def eps_for(psi):          # tolerance for a candidate with context masses psi (latent contexts)
                return markov_eps(N, np.minimum(1.0, psi + rc / 2)) if rule == "markov" else eps

            feas = np.zeros((len(mus), len(sigs)), dtype=bool)
            for a_, mu in enumerate(mus):
                for b_, sg in enumerate(sigs):
                    psi_c = psi_of(mu, sg)
                    feas[a_, b_] = regime_O_feasible(qN, psi_c, eps_for(psi_c), rc)
            share = feas.mean()
            if feas.any():
                mu_rng = (mus[feas.any(axis=1)].min(), mus[feas.any(axis=1)].max()); sg_rng = (sigs[feas.any(axis=0)].min(), sigs[feas.any(axis=0)].max())
            else:
                mu_rng = sg_rng = (np.nan, np.nan)
            truth_ok = regime_O_feasible(qN, psi0, eps_for(psi0), rc)
            rowsO.append(dict(N=N, rule=rule, share=share, mu_lo=mu_rng[0], mu_hi=mu_rng[1], sig_lo=sg_rng[0], sig_hi=sg_rng[1],
                              truth_feasible=truth_ok, r_cons=rc))
            np.save(f"{pref}_O_{rule}_{N}.npy", feas)
            print(f"O  N={N:>11,d} {rule:9s} feasible share {share:.3f}; mu in [{mu_rng[0]:.2f},{mu_rng[1]:.2f}] (truth {mu0:.2f}); "
                  f"sigma in [{sg_rng[0]:.2f},{sg_rng[1]:.2f}] (truth {sig0:.2f}); truth feasible: {truth_ok}", flush=True)
if rowsP:
    pd.DataFrame(rowsP).to_csv(f"{pref}_regimeP.csv", index=False)
if rowsO:
    pd.DataFrame(rowsO).to_csv(f"{pref}_regimeO.csv", index=False)
np.savez(f"{pref}_grid.npz", mus=mus, sigs=sigs)
