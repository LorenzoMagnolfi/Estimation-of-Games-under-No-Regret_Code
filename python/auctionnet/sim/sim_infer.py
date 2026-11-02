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

V05 rules (added 2026-11-02; Niccolò's V05 manuscript, Corollaries "Full-Feedback Inputs" and "Bandit Inputs"):
  tailF   (Hedge runs)  direct tail of the full-feedback family F: per context j with count m and exact payoff span K_j,
          H(m, delta) = K_j min{ m, c_F,j sqrt(m) + sqrt((m + 1) ln(J_N / delta)) }, J_N = 1 + ceil(log2 N), delta = alpha_R / M_R.
          The Hedge of sim_run.py is in family F (relative entropy to the uniform prior, step sqrt(ln|A| / (s - 1)) on payoffs
          divided by K_sub[j]); in the family's normalization its step multiplier lies in [rho_j, sqrt(2) rho_j] with
          rho_j = K_j / K_sub[j], so c_F,j = (1/rho_j + sqrt(2) rho_j / 4) sqrt(ln|A|).
  momentF (Hedge runs)  first-moment rectangle of family F at the expected count: (M_R / alpha_R) K_j (c_F,j + 1/4) sqrt(mu) / N.
  tailB   (exp3ix runs) direct tail of the stabilized bandit family B (exp3ix: bands a = 1, b = 1/2, so A = 3, b_- = 1/2,
          C_pr = ln|A|, L = K_j): H(m, delta) = L min{ m, A sqrt(d m C) + (sqrt 2 / (2 b_-)) sqrt(d m / C) ln(2 d J_N / delta) + ln(2/delta) }.
  momentB (exp3ix runs) first-moment rectangle of family B (calendar bound, keeps its sqrt(N) term):
          (M_R / alpha_R) (L / N) min{ mu, A sqrt(d C mu) + (ln d + 1) / (2 b_-) sqrt(d N / C) }.
The exact span K_j is recomputed over the full rival pool (the environment's K is a subsample maximum, not a support bound).
Regime P: m (and mu) are the observed counts. Regime O: tails use m = N min{1, psi_j + r_cons / 2} on the consistency event
(valid, weakly looser than the witness count), taken at delta / 2, plus the sample-vs-population comparison term
K_j sqrt(2 m ln(2|A| / delta)) at error delta / 2 (V05, "Integrating an Exogenous Competitive State"; the fresh independent
rival draws of sim_run.py satisfy its condition); first moments use the expected count N psi_j (V05, Q2).
Plain exp3 runs are outside family B: their only certified rule remains the historical 'markov' EXP3 envelope.
Usage: python sim_infer.py <env_npz> <run_npz> <out_prefix> [regimes=PO] [coarse|full] [checkpoints|all] [rules]
  rules: comma-separated subset of the rules applicable to the run's learner (default: all applicable).
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
if len(sys.argv) > 6 and sys.argv[6] not in ("", "all"):      # optional comma-separated subset of checkpoints
    want = set(int(float(x)) for x in sys.argv[6].split(","))
    checks = [c for c in checks if c in want]
RULES_WANTED = set(sys.argv[7].split(",")) if len(sys.argv) > 7 else None
TOL = 1e-12


LEARNER = str(run["learner"]) if "learner" in run.files else "hedge"


def markov_eps(N, phi):
    """Markov tolerance per auction (unconditional) for context shares phi (historical envelopes)."""
    if LEARNER == "exp3":          # Corollary 2 (bandit feedback): 2 sqrt(e-1) K sqrt(m |A| ln|A|)
        return (M_R / (ALPHA_R * N)) * 2 * np.sqrt(np.e - 1) * K * np.sqrt(N * phi * NM * lnA)
    return (M_R / (ALPHA_R * N)) * K * (np.sqrt(2 * N * phi * lnA) + lnA)


# ---- V05 rules: exact payoff span over the full rival pool and the family constants ----
K_FULL = None
C_F = None
A_B, BMINUS_B, C_B = 3.0, 0.5, lnA      # family B constants for exp3ix (a = 1, b = 1/2)


def exact_span():
    """Largest cross-multiplier payoff span per context over every recorded rival triple (chunked)."""
    R = env["R"]; E = env["E"]; RES = float(env["RES"])
    Kf = np.zeros(NV)
    CH = 500_000
    for start in range(0, len(R), CH):
        r = R[start:start + CH].astype(np.float64)
        r1, r2, r3 = r[:, 0][None, :], r[:, 1][None, :], r[:, 2][None, :]
        for j, v in enumerate(vgrid):
            b = alphas[:, None] * v
            s1 = b > r1
            s2 = (~s1) & (b > r2)
            s3 = (~s1) & (~s2) & (b > r3)
            price = np.maximum(np.where(s1, r1, np.where(s2, r2, np.where(s3, r3, 0.0))), RES)
            e = np.where(s1, E[0], np.where(s2, E[1], np.where(s3, E[2], 0.0)))
            U = e * (lam0 * v - price)
            Kf[j] = max(Kf[j], float((U.max(axis=0) - U.min(axis=0)).max()))
    return Kf


def v05_setup():
    global K_FULL, C_F
    if K_FULL is not None:
        return
    if LEARNER not in ("hedge", "exp3ix"):
        raise ValueError("the V05 rules are certified for the hedge (family F) and exp3ix (family B) learners only")
    K_FULL = np.maximum(exact_span(), K)          # the subsample maximum is a lower bound on the exact span
    Ksafe = np.maximum(K, 1e-12)
    rho = np.where(K_FULL > 0, K_FULL / Ksafe, 1.0)    # the Hedge learner's effective step multiplier band is [rho, sqrt(2) rho]
    C_F = (1.0 / rho + np.sqrt(2.0) * rho / 4.0) * np.sqrt(lnA)
    print(f"v05: K_full = {np.round(K_FULL, 5)}  K_env = {np.round(K, 5)}  rho = {np.round(rho, 4)}  c_F = {np.round(C_F, 3)}", flush=True)


def _JN(N):
    return 1 + int(np.ceil(np.log2(N)))


def rival_term(m, delta_G):
    """Sample-vs-population payoff comparison (Regime O), per context, cumulative units."""
    return K_FULL * np.sqrt(2.0 * m * np.log(NM / delta_G))


def tailF_eps(N, m, regime):
    v05_setup(); JN = _JN(N); delta = ALPHA_R / M_R; m = np.asarray(m, dtype=np.float64)
    if regime == "P":
        H = K_FULL * np.minimum(m, C_F * np.sqrt(m) + np.sqrt((m + 1.0) * np.log(JN / delta)))
    else:
        d2 = delta / 2.0
        H = np.minimum(K_FULL * m, K_FULL * (C_F * np.sqrt(m) + np.sqrt((m + 1.0) * np.log(JN / d2))) + rival_term(m, d2))
    return H / N


def tailB_eps(N, m, regime):
    v05_setup(); JN = _JN(N); delta = ALPHA_R / M_R; m = np.asarray(m, dtype=np.float64)
    d_ = delta if regime == "P" else delta / 2.0
    core = A_B * np.sqrt(NM * m * C_B) + (np.sqrt(2.0) / (2.0 * BMINUS_B)) * np.sqrt(NM * m / C_B) * np.log(2.0 * NM * JN / d_) + np.log(2.0 / d_)
    if regime == "P":
        H = K_FULL * np.minimum(m, core)
    else:
        H = np.minimum(K_FULL * m, K_FULL * core + rival_term(m, d_))
    return H / N


def momentF_eps(N, mu):
    v05_setup(); mu = np.asarray(mu, dtype=np.float64)
    b = K_FULL * np.minimum(mu, (C_F + 0.25) * np.sqrt(mu)) / N
    return (M_R / ALPHA_R) * b


def momentB_eps(N, mu):
    v05_setup(); mu = np.asarray(mu, dtype=np.float64)
    b = K_FULL * np.minimum(mu, A_B * np.sqrt(NM * C_B * mu) + (lnA + 1.0) / (2.0 * BMINUS_B) * np.sqrt(NM * N / C_B)) / N
    return (M_R / ALPHA_R) * b


def slack(N, rule, Z, Sa, Sc, Ra, Rc):
    phi = Z.sum(axis=0) / N
    cnt = Z.sum(axis=0)
    if rule == "exact":
        return np.zeros(NV)
    if rule == "none":             # no obedience restriction: the set identified by the bid marginal and consistency alone
        return np.full(NV, 1e6)
    if rule in ("markov", "markov_realized"):   # realized shares (legitimate in Regime P; diagnostic in Regime O)
        return markov_eps(N, phi)
    if rule == "empirical":
        S = lam0 * Sa - Sc; real = lam0 * Ra - Rc
        return np.maximum(S.max(axis=1) - real, 0.0) / N
    if rule == "tailF":            # Regime P: observed counts
        return tailF_eps(N, cnt, "P")
    if rule == "tailB":
        return tailB_eps(N, cnt, "P")
    if rule == "momentF":
        return momentF_eps(N, cnt)
    if rule == "momentB":
        return momentB_eps(N, cnt)
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


def applicable_rules():
    rules = ["exact", "empirical"] + (["none"] if "O" in regimes else [])
    if LEARNER == "hedge":
        rules += ["markov"] + (["markov_realized"] if "O" in regimes else []) + ["tailF", "momentF"]
    elif LEARNER == "exp3":
        rules += ["markov"] + (["markov_realized"] if "O" in regimes else [])
    elif LEARNER == "exp3ix":
        rules += ["tailB", "momentB"]
    if RULES_WANTED is not None:
        rules = [r for r in rules if r in RULES_WANTED]
    return rules


rowsP, rowsO = [], []
# empirical context masses (the truth of a paired-environment run; a second truth check in Regime O when present)
emp_hist = env["emp_hist"].astype(float) if "emp_hist" in env.files else None
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
    for rule in applicable_rules():
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
                if rule == "markov":
                    return markov_eps(N, np.minimum(1.0, psi + rc / 2))
                if rule == "tailF":
                    return tailF_eps(N, N * np.minimum(1.0, psi + rc / 2), "O")
                if rule == "tailB":
                    return tailB_eps(N, N * np.minimum(1.0, psi + rc / 2), "O")
                if rule == "momentF":
                    return momentF_eps(N, N * psi)
                if rule == "momentB":
                    return momentB_eps(N, N * psi)
                return eps

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
            emp_ok = regime_O_feasible(qN, emp_hist, eps_for(emp_hist), rc) if emp_hist is not None else np.nan
            rowsO.append(dict(N=N, rule=rule, share=share, mu_lo=mu_rng[0], mu_hi=mu_rng[1], sig_lo=sg_rng[0], sig_hi=sg_rng[1],
                              truth_feasible=truth_ok, emp_feasible=emp_ok, r_cons=rc))
            np.save(f"{pref}_O_{rule}_{N}.npy", feas)
            print(f"O  N={N:>11,d} {rule:9s} feasible share {share:.3f}; mu in [{mu_rng[0]:.2f},{mu_rng[1]:.2f}] (truth {mu0:.2f}); "
                  f"sigma in [{sg_rng[0]:.2f},{sg_rng[1]:.2f}] (truth {sig0:.2f}); truth feasible: {truth_ok}; empirical masses feasible: {emp_ok}", flush=True)
if rowsP:
    pd.DataFrame(rowsP).to_csv(f"{pref}_regimeP.csv", index=False)
if rowsO:
    pd.DataFrame(rowsO).to_csv(f"{pref}_regimeO.csv", index=False)
np.savez(f"{pref}_grid.npz", mus=mus, sigs=sigs)
