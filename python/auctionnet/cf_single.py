"""Single-learner counterfactual illustration for Design A: the focal advertiser's spend under a different auction rule.

Rivals are held at their recorded bids (partial equilibrium). For each rule, the expected exposure value a[k, j] = E[e] v_j
and expected payment c[k, j] = E[e * price] of every multiplier k in every context j are computed against a fixed subsample
of the focal role's pooled rival triples; the payoff at value lambda is lambda * a - c.
Rules: gsp (as recorded: next price, reserve 0.01), gsp_r05 and gsp_r10 (reserve 0.05 and 0.10 on every slot),
fp (first price on every slot), fp_top (first price on slot 1, next price on slots 2-3).
Outputs, per rule:
  truth      spend per auction when the learner best-responds at the true value lambda0 (the long run of Hedge);
  naive      spend if the advertiser kept its recorded GSP bidding (the Hedge distribution at N_MAIN) under the new rule;
  predicted  [min, max] spend over values lambda in the identified interval (Regime P, Markov) at 1e7 and 1e8, with exact
             best responses (long run) and with play within the Markov tolerance after N_cf auctions under the new rule
             (min / max of spend over contextual distributions with per-context regret at most eps_j, an LP per lambda).
Usage: python cf_single.py <env_npz> <run_npz> <regimeP_csv> <out_prefix> [n_rival_draws=2000000]
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from provenance import code_commit

env = np.load(sys.argv[1]); run = np.load(sys.argv[2]); regP = pd.read_csv(sys.argv[3]); pref = sys.argv[4]
n_draw = int(float(sys.argv[5])) if len(sys.argv) > 5 else 2_000_000
t0 = time.time()
alphas, vgrid, psi0, lam0, E, R = env["alphas"], env["vgrid"], env["psi0"], float(env["lam0"]), env["E"], env["R"]
NV = len(vgrid)
rng = np.random.default_rng(0)
Rt = R[rng.choice(len(R), min(n_draw, len(R)), replace=False)].astype(np.float64)
r1, r2, r3 = Rt[:, 0], Rt[:, 1], Rt[:, 2]
ACF = np.exp(np.linspace(np.log(5.0), np.log(260.0), 48))        # absolute multipliers for the counterfactual grid
ALPHA_R, M_R = 0.045, NV


def outcome(b, rule):
    res = {"gsp": 0.01, "gsp_r05": 0.05, "gsp_r10": 0.10, "fp": 0.01, "fp_top": 0.01}[rule]
    ok = b >= res
    s1 = ok & (b > r1); s2 = ok & ~s1 & (b > r2); s3 = ok & ~s1 & ~s2 & (b > r3)
    nxt = np.where(s1, r1, np.where(s2, r2, np.where(s3, r3, 0.0)))
    price = np.maximum(nxt, res)
    if rule == "fp":
        price = np.full_like(price, b)
    elif rule == "fp_top":
        price = np.where(s1, b, price)
    e = np.where(s1, E[0], np.where(s2, E[1], np.where(s3, E[2], 0.0)))
    return e, price


def tables(mults, rule):
    a = np.zeros((len(mults), NV)); c = np.zeros((len(mults), NV)); K = np.zeros(NV)
    for j, v in enumerate(vgrid):
        U = []
        for k, m in enumerate(mults):
            e, p = outcome(m * v, rule)
            a[k, j] = (e * v).mean(); c[k, j] = (e * p).mean(); U.append((e, p))
        # payoff range per auction at lambda0 (for the tolerance), as in sim_env2.py
        Uall = np.array([e * (lam0 * v - p) for e, p in U])
        K[j] = float((Uall.max(axis=0) - Uall.min(axis=0)).max())
    return a, c, K


def br_spend(a, c, lam):
    u = lam * a - c
    k = u.argmax(axis=0)
    return float(np.sum(psi0 * c[k, np.arange(NV)])), k


def tol_spend(a, c, K, lam, N):
    """min and max spend over contextual distributions with per-context regret <= Markov eps_j after N auctions."""
    nk = a.shape[0]; out = []
    eps_cond = (M_R / (ALPHA_R * N)) * K * (np.sqrt(2 * N * psi0 * np.log(nk)) + np.log(nk)) / psi0   # per auction in context j
    u = lam * a - c
    for sign in (1.0, -1.0):
        tot = 0.0
        for j in range(NV):                                            # contexts separate: one small LP each
            # variables: probabilities over multipliers in context j; regret vs every fixed deviation <= eps
            A_ub = np.array([u[kp, j] - u[:, j] for kp in range(nk)]); b_ub = np.full(nk, eps_cond[j])
            res = linprog(sign * c[:, j], A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, nk)), b_eq=[1.0], bounds=[(0, None)] * nk, method="highs")
            tot += psi0[j] * res.fun * sign
        out.append(tot)
    return out[0], out[1]


Nmain = int(run["checks"][-1]); Zf = run[f"Z_{Nmain}"].astype(float)
nu_rec = Zf / np.maximum(Zf.sum(axis=0, keepdims=True), 1)         # recorded GSP play: multiplier distribution per context
intervals = {int(r.N): (float(r.lam_lo), float(r.lam_hi)) for _, r in regP[regP.rule == "markov"].iterrows() if int(r.N) in (10_000_000, 100_000_000)}
rows = []
LGRID = np.linspace(1.0, 250.0, 250)                                # value grid for the spend curves
curves = {}
for rule in ["gsp", "gsp_r05", "gsp_r10", "fp", "fp_top"]:
    a, c, K = tables(ACF, rule)
    a12, c12, _ = tables(alphas, rule)
    truth, kstar = br_spend(a, c, lam0)
    naive = float(np.sum(psi0 * np.sum(nu_rec * c12, axis=0)))
    row = dict(rule=rule, truth=truth, truth_best_mult_over_lam=list(np.round(ACF[kstar] / lam0, 3)), naive=naive,
               truth_surplus=float(np.sum(psi0 * (lam0 * a - c)[kstar, np.arange(NV)])))
    curves[rule] = [br_spend(a, c, l)[0] for l in LGRID]
    for N, (lo, hi) in intervals.items():
        lams = np.linspace(max(lo, 1.0), hi, 120)
        sp = [br_spend(a, c, l)[0] for l in lams]
        row[f"pred_br_{N:.0e}"] = (min(sp), max(sp))
        for Ncf in (10_000_000, 100_000_000):
            mm = [tol_spend(a, c, K, l, Ncf) for l in lams[::6]]
            row[f"pred_tol{Ncf:.0e}_{N:.0e}"] = (min(m[0] for m in mm), max(m[1] for m in mm))
    if rule != "gsp":                                               # predicted change relative to the recorded rule
        for N, (lo, hi) in intervals.items():
            sel = (LGRID >= lo) & (LGRID <= hi)
            new, base = np.array(curves[rule])[sel], np.array(curves["gsp"])[sel]
            ok = base > 0
            row[f"pred_ratio_{N:.0e}"] = (float((new[ok] / base[ok]).min()), float((new[ok] / base[ok]).max())) if ok.any() else (np.nan, np.nan)
            row[f"pred_diff_{N:.0e}"] = (float((new - base).min()), float((new - base).max()))
        row["truth_ratio"] = truth / rows[0]["truth"]
    rows.append(row)
    print(rule, {k: (np.round(v, 6) if isinstance(v, float) else v) for k, v in row.items() if k != "rule"}, f"({time.time() - t0:.0f}s)", flush=True)
commit, dirty = code_commit()
with open(pref + "_cf_single.json", "w") as f:
    json.dump(dict(rows=rows, lam0=lam0, lam_grid=list(LGRID), curves=curves, intervals={str(k): v for k, v in intervals.items()}, n_rival_draws=len(Rt),
                   code_commit=commit, code_dirty=dirty, seconds=round(time.time() - t0)), f, indent=1)
print("CF_DONE", f"({time.time() - t0:.0f}s)")
