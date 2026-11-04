"""Design A inference, version 2 (reconciliation batch of 4 November 2026, after Niccolò's feedback on the two laboratories).

Same observations and regimes as sim_infer.py (Regime P: contexts observed, per-context intervals for the value per
conversion; Regime O: contexts latent, witness LP over (multiplier, context) masses with the L1 composition set), with:
  - the zero-slope sign repaired in Regime P (I5) and solver statuses separated from infeasibility in Regime O (I7d);
  - an optional tables JSON from audit_env_role.py: exact full-pool mean tables (I3), context-conditional tables for a
    paired environment (I2), the exact span K_full and the exact reward width L_exact = umax - g_min per context (I1);
  - the family-B rules use L = L_exact when the run was generated with that normalizer (run npz 'ixnorm' starts with
    'exact'), and refuse the certified family-B rules otherwise (losses outside [0, 1] are not in family B);
  - new rules: 'jointF' / 'jointB', the normalized joint first-moment budget of eq:s4jointbudget with equal weights
    (Regime P: LP in (lambda, r); Regime O: r variables added to the witness LP);
    'tailF_own' / 'tailB_own', the direct tail evaluated at the witness's own context counts N s_k(nu) (eq:s4directtail,
    Q14) through tangent-line relaxations of the concave count bound (an outer, therefore valid, LP; tighter than the
    separately inflated count ceiling as the number of tangents grows); Regime O only;
    'tailBc' / 'momentBc', the conditional-count family-B inputs of eq:s4Bconditional, to be used only where the
    conditioning premise is established (in this simulator the context schedule is drawn in advance, independent of the
    learner, and the learner's randomization is private; the flag is recorded in the output).
Usage: python sim_infer2.py <env_npz> <run_npz> <out_prefix> [regimes=PO] [grid=full|coarse] [checkpoints=all|c1,c2]
                            [rules=r1,r2,...] [tables=<audit_json>] [ntan=40]
Outputs <out_prefix>_regimeP.csv, _regimeO.csv, _O_<rule>_<N>.npy, _grid.npz, _solver_log.json; never overwrites sim_infer.py outputs
if a different prefix is used.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import norm

opts = {t.split("=", 1)[0]: t.split("=", 1)[1] for t in sys.argv[4:] if "=" in t}
env = np.load(sys.argv[1]); run = np.load(sys.argv[2]); pref = sys.argv[3]
regimes = opts.get("regimes", "PO")
coarse = opts.get("grid", "full") == "coarse"
alphas, vgrid, psi0, lam0, K, expo, cost, bid_index, NB = (env["alphas"], env["vgrid"], env["psi0"], float(env["lam0"]), env["K"],
                                                          env["expo"], env["cost"], env["bid_index"], int(env["NB"]))
mu0, sig0, edges = float(env["mu0"]), float(env["sig0"]), env["edges"]
NM, NV = len(alphas), len(vgrid)
ALPHA_R, ALPHA_C = 0.045, 0.005
M_R = NV
lnA = np.log(NM)
TOL = 1e-12
LEARNER = str(run["learner"]) if "learner" in run.files else "hedge"
MODE = str(run["mode"]) if "mode" in run.files else "iid"
IXNORM = str(run["ixnorm"]) if "ixnorm" in run.files else "env_K"
emp_hist = env["emp_hist"].astype(float) if "emp_hist" in env.files else None
paired_env = "ctx_idx" in env.files
NTAN = int(opts.get("ntan", "40"))

# ---- tables: stored (subsample, unconditional) or audited (exact full pool; conditional for a paired environment) ----
TABLES = "stored_subsample_unconditional"
K_FULL = None; L_EXACT = None
if "tables" in opts:
    with open(opts["tables"]) as fh:
        aud = json.load(fh)
    K_FULL = np.array(aud["K_full"], dtype=np.float64)
    L_EXACT = np.array(aud["width_exact"], dtype=np.float64)
    if paired_env and "expo_cond" in aud:
        expo = np.array(aud["expo_cond"]); cost = np.array(aud["cost_cond"]); TABLES = "audit_conditional_full_pool"
    else:
        expo = np.array(aud["expo_full"]); cost = np.array(aud["cost_full"]); TABLES = "audit_unconditional_full_pool"
a_kj = expo * vgrid[None, :]
c_kj = cost
checks = [int(x) for x in run["checks"]]
if opts.get("checkpoints", "all") != "all":
    want = set(int(float(x)) for x in opts["checkpoints"].split(","))
    checks = [c for c in checks if c in want]
RULES_WANTED = set(opts["rules"].split(",")) if "rules" in opts else None
SOLVER_LOG = []


# ---- family constants ----
def exact_span():
    R = env["R"]; E = env["E"]; RES = float(env["RES"])
    Kf = np.zeros(NV); CH = 500_000
    for start in range(0, len(R), CH):
        r = R[start:start + CH].astype(np.float64)
        r1, r2, r3 = r[:, 0][None, :], r[:, 1][None, :], r[:, 2][None, :]
        for j, v in enumerate(vgrid):
            b = alphas[:, None] * v
            s1 = b > r1; s2 = (~s1) & (b > r2); s3 = (~s1) & (~s2) & (b > r3)
            price = np.maximum(np.where(s1, r1, np.where(s2, r2, np.where(s3, r3, 0.0))), RES)
            e = np.where(s1, E[0], np.where(s2, E[1], np.where(s3, E[2], 0.0)))
            U = e * (lam0 * v - price)
            Kf[j] = max(Kf[j], float((U.max(axis=0) - U.min(axis=0)).max()))
    return Kf


if K_FULL is None:
    K_FULL = np.maximum(exact_span(), K)
Ksafe = np.maximum(K, 1e-12)
rho = np.where(K_FULL > 0, K_FULL / Ksafe, 1.0)                       # Hedge: the learner normalized by K, the bound by K_full
C_F = (1.0 / rho + np.sqrt(2.0) * rho / 4.0) * np.sqrt(lnA)          # step band [rho, sqrt(2) rho]
A_B, BMINUS_B, C_B = 3.0, 0.5, lnA                                   # family B: a = 1, b = 1/2
if LEARNER == "exp3ix":
    if IXNORM.startswith("exact") and L_EXACT is not None:
        L_B = L_EXACT
    elif IXNORM.startswith("exact"):
        L_B = run["L_exact"].astype(np.float64)
    else:
        L_B = None     # historical normalizer K: losses can exceed one, no certified family-B input (I1)
print(f"sim_infer2: learner {LEARNER}, mode {MODE}, ixnorm {IXNORM}, tables {TABLES}", flush=True)


def _JN(N):
    return 1 + int(np.ceil(np.log2(N)))


def rival_term(m, delta_G):
    return K_FULL * np.sqrt(2.0 * m * np.log(NM / delta_G))


def H_F(N, m, regime, delta=None):
    """Cumulative family-F direct-tail bound at count m (calendar version, eq:s4Finputs), with the rival term in Regime O."""
    JN = _JN(N); delta = ALPHA_R / M_R if delta is None else delta; m = np.asarray(m, dtype=np.float64)
    if regime == "P":
        return K_FULL * np.minimum(m, C_F * np.sqrt(m) + np.sqrt((m + 1.0) * np.log(JN / delta)))
    d2 = delta / 2.0
    return np.minimum(K_FULL * m, K_FULL * (C_F * np.sqrt(m) + np.sqrt((m + 1.0) * np.log(JN / d2))) + rival_term(m, d2))


def H_B(N, m, regime, cond=False):
    """Cumulative family-B direct-tail bound: calendar (eq:s4Btail) or conditional-count (eq:s4Bconditional)."""
    if L_B is None:
        raise ValueError("family-B rules need a run normalized by the exact reward width (ixnorm exact)")
    JN = _JN(N); delta = ALPHA_R / M_R; m = np.asarray(m, dtype=np.float64)
    d_ = delta if regime == "P" else delta / 2.0
    if cond:
        core = A_B * np.sqrt(NM * m * C_B) + (1.0 / (2.0 * BMINUS_B)) * np.sqrt(NM * m / C_B) * np.log(2.0 * NM / d_) + np.log(2.0 / d_)
    else:
        core = A_B * np.sqrt(NM * m * C_B) + (np.sqrt(2.0) / (2.0 * BMINUS_B)) * np.sqrt(NM * m / C_B) * np.log(2.0 * NM * JN / d_) + np.log(2.0 / d_)
    if regime == "P":
        return L_B * np.minimum(m, core)
    return np.minimum(L_B * m, L_B * core + rival_term(m, d_))


def b_F(N, mu):
    mu = np.asarray(mu, dtype=np.float64)
    return K_FULL * np.minimum(mu, (C_F + 0.25) * np.sqrt(mu)) / N


def b_B(N, mu, cond=False):
    if L_B is None:
        raise ValueError("family-B rules need a run normalized by the exact reward width (ixnorm exact)")
    mu = np.asarray(mu, dtype=np.float64)
    if cond:
        return L_B * np.minimum(mu, (A_B + (lnA + 1.0) / (2.0 * BMINUS_B * C_B)) * np.sqrt(NM * C_B * mu)) / N
    return L_B * np.minimum(mu, A_B * np.sqrt(NM * C_B * mu) + (lnA + 1.0) / (2.0 * BMINUS_B) * np.sqrt(NM * N / C_B)) / N


# ---- per-context allowances (Regime P: observed counts; Regime O: candidate-dependent, see eps_for) ----
def slack(N, rule, Z, Sa, Sc, Ra, Rc):
    cnt = Z.sum(axis=0)
    if rule == "exact":
        return np.zeros(NV)
    if rule == "none":
        return np.full(NV, 1e6)
    if rule == "empirical":
        S = lam0 * Sa - Sc; real = lam0 * Ra - Rc
        return np.maximum(S.max(axis=1) - real, 0.0) / N
    if rule == "tailF":
        return H_F(N, cnt, "P") / N
    if rule == "tailB":
        return H_B(N, cnt, "P") / N
    if rule == "tailBc":
        return H_B(N, cnt, "P", cond=True) / N
    if rule == "momentF":
        return (M_R / ALPHA_R) * b_F(N, cnt)
    if rule == "momentB":
        return (M_R / ALPHA_R) * b_B(N, cnt)
    if rule == "momentBc":
        return (M_R / ALPHA_R) * b_B(N, cnt, cond=True)
    if rule in ("jointF", "jointB"):
        return None       # handled by the joint-budget programs
    raise ValueError(rule)


def regime_P(N, Sa, Sc, Ra, Rc, eps):
    """Per-context intervals from lambda * A - C <= eps; zero slope leaves -C <= eps, infeasible iff C < -eps (I5 repaired)."""
    lo, hi = np.full(NV, -np.inf), np.full(NV, np.inf)
    for j in range(NV):
        for kp in range(NM):
            A = (Sa[j, kp] - Ra[j]) / N; C = (Sc[j, kp] - Rc[j]) / N
            if A > TOL:
                hi[j] = min(hi[j], (eps[j] + C) / A)
            elif A < -TOL:
                lo[j] = max(lo[j], (eps[j] + C) / A)
            elif C < -eps[j] - TOL:
                lo[j], hi[j] = np.inf, -np.inf
    return lo, hi, max(lo), min(hi)


def regime_P_joint(N, Sa, Sc, Ra, Rc, b):
    """Normalized joint first-moment budget (eq:s4jointbudget), equal weights on contexts with b_k > 0: the interval for
    lambda is min and max of lambda over (lambda, r >= 0) with lambda A_jk - C_jk <= r_j and sum_j w_j r_j / b_j <= 1/alpha_R."""
    J = np.where(b > 0)[0]
    if len(J) == 0:
        return regime_P(N, Sa, Sc, Ra, Rc, np.zeros(NV))[2:]
    w = np.zeros(NV); w[J] = 1.0 / len(J)
    nvar = 1 + NV
    A_ub, b_ub = [], []
    for j in range(NV):
        for kp in range(NM):
            A = (Sa[j, kp] - Ra[j]) / N; C = (Sc[j, kp] - Rc[j]) / N
            row = np.zeros(nvar); row[0] = A; row[1 + j] = -1.0; A_ub.append(row); b_ub.append(C)
    row = np.zeros(nvar); row[1:] = np.where(b > 0, w / np.maximum(b, 1e-300), 0.0); A_ub.append(row); b_ub.append(1.0 / ALPHA_R)
    bounds = [(None, None)] + [((0.0, 0.0) if b[j] <= 0 else (0.0, None)) for j in range(NV)]
    out = []
    for sign in (1.0, -1.0):
        c = np.zeros(nvar); c[0] = sign
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")
        if res.status == 0:
            out.append(sign * res.fun)
        elif res.status == 2:
            out.append(np.inf if sign > 0 else -np.inf)
        elif res.status == 3:      # unbounded in this direction
            out.append(-np.inf if sign > 0 else np.inf)
        else:
            SOLVER_LOG.append(dict(regime="P", rule="joint", N=int(N), status=int(res.status), message=str(res.message)))
            raise RuntimeError(f"Regime P joint-budget solve ended with status {res.status}: {res.message}")
    return out[0], out[1]


def psi_of(mu, sig):
    p = np.diff(norm.cdf(edges, loc=mu, scale=sig)); return p / p.sum()


# ---- Regime O witness programs ----
idx = lambda k, j: k * NV + j
NVAR0 = NM * NV + NV


def base_O():
    """Action matching (equalities), composition box |s_j - psi_j| <= t_j with sum t_j <= r, as in sim_infer.py."""
    A_eq = np.zeros((NB, NVAR0))
    for k in range(NM):
        for j in range(NV):
            A_eq[bid_index[k, j], idx(k, j)] = 1.0
    rows, rhs_fn = [], []
    for j in range(NV):
        row = np.zeros(NVAR0); row[[idx(k, j) for k in range(NM)]] = 1.0; row[NM * NV + j] = -1.0; rows.append(row)
        row = np.zeros(NVAR0); row[[idx(k, j) for k in range(NM)]] = -1.0; row[NM * NV + j] = -1.0; rows.append(row)
    row = np.zeros(NVAR0); row[NM * NV:] = 1.0; rows.append(row)
    return A_eq, np.array(rows)


A_EQ0, A_CONS = base_O()
GAMMA = np.zeros((NV, NM, NVAR0))        # obedience gain rows: Gamma_{j,kp}(nu) = sum_k nu(k,j) [u(kp,j) - u(k,j)]
for j in range(NV):
    for kp in range(NM):
        for k in range(NM):
            GAMMA[j, kp, idx(k, j)] = lam0 * (a_kj[kp, j] - a_kj[k, j]) - (c_kj[kp, j] - c_kj[k, j])
S_ROW = np.zeros((NV, NVAR0))             # s_j(nu) = sum_k nu(k, j)
for j in range(NV):
    S_ROW[j, [idx(k, j) for k in range(NM)]] = 1.0


def _solve(c, A_ub, b_ub, A_eq, b_eq, bounds, tag):
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status == 0:
        return True
    if res.status == 2:
        return False
    SOLVER_LOG.append(dict(regime="O", rule=tag, status=int(res.status), message=str(res.message)))
    raise RuntimeError(f"Regime O solve ({tag}) ended with status {res.status}: {res.message}")


def O_feasible_fixed(qN, psi, eps, r_cons, tag):
    """Fixed per-context allowances eps_j (rectangle, calendar tail at the count ceiling, exact, none)."""
    A_ub = np.vstack([A_CONS, GAMMA.reshape(NV * NM, NVAR0)])
    b_ub = np.concatenate([np.repeat(psi, 2) * np.tile([1.0, -1.0], NV), [r_cons], np.repeat(eps, NM)])
    return _solve(np.zeros(NVAR0), A_ub, b_ub, A_EQ0, qN, [(0, None)] * NVAR0, tag)


def O_feasible_joint(qN, psi, b, r_cons, tag):
    """Normalized joint first-moment budget: r_j variables, Gamma_{j,kp} <= r_j, sum_j w_j r_j / b_j <= 1/alpha_R."""
    J = np.where(b > 0)[0]
    nvar = NVAR0 + NV
    w = np.zeros(NV)
    if len(J):
        w[J] = 1.0 / len(J)
    rows, rhs = [], []
    rows.append(np.hstack([A_CONS, np.zeros((A_CONS.shape[0], NV))])); rhs.append(np.concatenate([np.repeat(psi, 2) * np.tile([1.0, -1.0], NV), [r_cons]]))
    G = np.zeros((NV * NM, nvar))
    for j in range(NV):
        for kp in range(NM):
            G[j * NM + kp, :NVAR0] = GAMMA[j, kp]; G[j * NM + kp, NVAR0 + j] = -1.0
    rows.append(G); rhs.append(np.zeros(NV * NM))
    row = np.zeros(nvar); row[NVAR0:] = np.where(b > 0, w / np.maximum(b, 1e-300), 0.0); rows.append(row[None, :]); rhs.append(np.array([1.0 / ALPHA_R]))
    A_eq = np.hstack([A_EQ0, np.zeros((NB, NV))])
    bounds = [(0, None)] * NVAR0 + [((0.0, 0.0) if b[j] <= 0 else (0.0, None)) for j in range(NV)]
    return _solve(np.zeros(nvar), np.vstack(rows), np.concatenate(rhs), A_eq, qN, bounds, tag)


def O_feasible_own(qN, psi, r_cons, N, Hfun, tag):
    """Direct tail at the witness's own counts: Gamma_{j,kp}(nu) <= H_j(N s_j(nu)) / N, relaxed by tangents of the concave
    count bound at NTAN log-spaced counts (each tangent overestimates a concave function, so the LP is an outer region);
    the cap K m is included as its own tangent (slope K through the origin)."""
    m_grid = np.unique(np.concatenate([[1.0], np.geomspace(1.0, N, NTAN)]))
    rows, rhs = [A_CONS], [np.concatenate([np.repeat(psi, 2) * np.tile([1.0, -1.0], NV), [r_cons]])]
    h = 1e-3
    for j in range(NV):
        # Hfun takes a count vector over contexts; evaluate context j at each tangent point with a constant count vector
        vals = np.array([Hfun(N, np.full(NV, m), "O")[j] for m in m_grid])
        dvals = np.array([(Hfun(N, np.full(NV, m * (1 + h)), "O")[j] - Hfun(N, np.full(NV, m * (1 - h)), "O")[j]) / (2 * h * m) for m in m_grid])
        for t, m_t in enumerate(m_grid):
            # Gamma(nu) - [H(m_t) + H'(m_t) (N s_j(nu) - m_t)] / N <= 0
            for kp in range(NM):
                row = GAMMA[j, kp] - dvals[t] * S_ROW[j]
                rows.append(row[None, :]); rhs.append(np.array([(vals[t] - dvals[t] * m_t) / N]))
        for kp in range(NM):     # deterministic cap: Gamma <= K_j s_j(nu)  (L for family B is inside Hfun via its own cap)
            cap = (K_FULL[j] if Hfun is H_F else L_B[j])
            row = GAMMA[j, kp] - cap * S_ROW[j]; rows.append(row[None, :]); rhs.append(np.array([0.0]))
    return _solve(np.zeros(NVAR0), np.vstack(rows), np.concatenate(rhs), A_EQ0, qN, [(0, None)] * NVAR0, tag)


def applicable_rules():
    rules = ["exact", "empirical"] + (["none"] if "O" in regimes else [])
    if LEARNER == "hedge":
        rules += ["tailF", "momentF", "jointF"] + (["tailF_own"] if "O" in regimes else [])
    elif LEARNER == "exp3ix":
        if L_B is not None:
            rules += ["tailB", "momentB", "jointB", "tailBc", "momentBc"] + (["tailB_own"] if "O" in regimes else [])
        else:
            print("exp3ix run with the historical normalizer K: certified family-B rules not applicable (I1); only exact/empirical/none", flush=True)
    if RULES_WANTED is not None:
        rules = [r for r in rules if r in RULES_WANTED]
    return rules


rowsP, rowsO = [], []
mus = mu0 + (np.linspace(-1.0, 1.0, 21) if coarse else np.linspace(-1.0, 1.0, 41))
sigs = np.linspace(0.15, 2.0, 38) if not coarse else np.round(np.linspace(0.2, 2.0, 19), 3)
for N in checks:
    Z, Sa, Sc, Ra, Rc = run[f"Z_{N}"], run[f"Sa_{N}"], run[f"Sc_{N}"], run[f"Ra_{N}"], run[f"Rc_{N}"]
    qN = np.zeros(NB)
    for k in range(NM):
        for j in range(NV):
            qN[bid_index[k, j]] += Z[k, j] / N
    r_cons = np.sqrt(2 * np.log((2 ** NV - 2) / ALPHA_C) / N)
    cnt = Z.sum(axis=0)
    for rule in applicable_rules():
        # ---------------- Regime P ----------------
        if "P" in regimes and rule not in ("none", "tailF_own", "tailB_own"):
            if rule in ("jointF", "jointB"):
                b = b_F(N, cnt) if rule == "jointF" else b_B(N, cnt)
                L, H = regime_P_joint(N, Sa, Sc, Ra, Rc, b)
                lo = hi = np.full(NV, np.nan); eps = b * (M_R / ALPHA_R)     # the matched rectangle, for the record
            else:
                eps = slack(N, rule, Z, Sa, Sc, Ra, Rc)
                lo, hi, L, H = regime_P(N, Sa, Sc, Ra, Rc, eps)
            rowsP.append(dict(N=N, rule=rule, lam_lo=L, lam_hi=H, width=H - L, contains_truth=(L - 1e-6 <= lam0 <= H + 1e-6), tables=TABLES, ixnorm=IXNORM,
                              **{f"lo_v{j + 1}": lo[j] for j in range(NV)}, **{f"hi_v{j + 1}": hi[j] for j in range(NV)},
                              **{f"eps_v{j + 1}": eps[j] for j in range(NV)}))
            print(f"P  N={N:>11,d} {rule:9s} lambda in [{L:9.2f}, {H:9.2f}] (truth {lam0:.0f}, {'in' if L - 1e-6 <= lam0 <= H + 1e-6 else 'OUT'})", flush=True)
        # ---------------- Regime O ----------------
        if "O" in regimes:
            rc = 0.0 if rule == "exact" else r_cons

            def feasible(psi):
                m_ceil = N * np.minimum(1.0, psi + rc / 2)
                if rule in ("exact", "none", "empirical"):
                    return O_feasible_fixed(qN, psi, slack(N, rule, Z, Sa, Sc, Ra, Rc), rc, rule)
                if rule == "tailF":
                    return O_feasible_fixed(qN, psi, H_F(N, m_ceil, "O") / N, rc, rule)
                if rule == "tailB":
                    return O_feasible_fixed(qN, psi, H_B(N, m_ceil, "O") / N, rc, rule)
                if rule == "tailBc":
                    return O_feasible_fixed(qN, psi, H_B(N, m_ceil, "O", cond=True) / N, rc, rule)
                if rule == "momentF":
                    return O_feasible_fixed(qN, psi, (M_R / ALPHA_R) * b_F(N, N * psi), rc, rule)
                if rule == "momentB":
                    return O_feasible_fixed(qN, psi, (M_R / ALPHA_R) * b_B(N, N * psi), rc, rule)
                if rule == "momentBc":
                    return O_feasible_fixed(qN, psi, (M_R / ALPHA_R) * b_B(N, N * psi, cond=True), rc, rule)
                if rule == "jointF":
                    return O_feasible_joint(qN, psi, b_F(N, N * psi), rc, rule)
                if rule == "jointB":
                    return O_feasible_joint(qN, psi, b_B(N, N * psi), rc, rule)
                if rule == "tailF_own":
                    return O_feasible_own(qN, psi, rc, N, H_F, rule)
                if rule == "tailB_own":
                    return O_feasible_own(qN, psi, rc, N, H_B, rule)
                raise ValueError(rule)

            feas = np.zeros((len(mus), len(sigs)), dtype=bool)
            for a_, mu in enumerate(mus):
                for b_, sg in enumerate(sigs):
                    feas[a_, b_] = feasible(psi_of(mu, sg))
            share = feas.mean()
            if feas.any():
                mu_rng = (mus[feas.any(axis=1)].min(), mus[feas.any(axis=1)].max()); sg_rng = (sigs[feas.any(axis=0)].min(), sigs[feas.any(axis=0)].max())
            else:
                mu_rng = sg_rng = (np.nan, np.nan)
            truth_ok = feasible(psi0)
            emp_ok = feasible(emp_hist) if emp_hist is not None else np.nan
            rowsO.append(dict(N=N, rule=rule, share=share, mu_lo=mu_rng[0], mu_hi=mu_rng[1], sig_lo=sg_rng[0], sig_hi=sg_rng[1],
                              truth_feasible=truth_ok, emp_feasible=emp_ok, r_cons=rc, tables=TABLES, ixnorm=IXNORM))
            np.save(f"{pref}_O_{rule}_{N}.npy", feas)
            print(f"O  N={N:>11,d} {rule:9s} feasible share {share:.3f}; mu in [{mu_rng[0]:.2f},{mu_rng[1]:.2f}] (truth {mu0:.2f}); "
                  f"sigma in [{sg_rng[0]:.2f},{sg_rng[1]:.2f}] (truth {sig0:.2f}); lognormal truth feasible: {truth_ok}; empirical masses feasible: {emp_ok}", flush=True)
if rowsP:
    pd.DataFrame(rowsP).to_csv(f"{pref}_regimeP.csv", index=False)
if rowsO:
    pd.DataFrame(rowsO).to_csv(f"{pref}_regimeO.csv", index=False)
np.savez(f"{pref}_grid.npz", mus=mus, sigs=sigs)
with open(f"{pref}_solver_log.json", "w") as fh:
    json.dump(dict(tables=TABLES, ixnorm=IXNORM, learner=LEARNER, mode=MODE, ntan=NTAN, K_full=K_FULL.tolist(),
                   L_exact=(None if L_EXACT is None else L_EXACT.tolist()), c_F=C_F.tolist(), solver_log=SOLVER_LOG), fh, indent=1)
print(f"sim_infer2 done: {len(SOLVER_LOG)} non-certified solver statuses (all raised)", flush=True)
