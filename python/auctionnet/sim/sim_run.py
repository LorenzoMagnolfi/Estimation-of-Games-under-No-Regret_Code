"""Design A learners: one focal bidder with a contextual learner (one copy per context) against the recorded fringe.

Learners:
  hedge : exponential weights with full feedback (payoff vector over all multipliers, given the realized context and
          rival triple), anytime step size eta_t = sqrt(ln NM / t) on payoffs normalized by K[v]. Because the environment
          is exogenous to the focal bidder, the Hedge path is a function of cumulative payoff vectors and is vectorized.
  exp3  : bandit feedback (only the chosen multiplier's realized payoff), importance-weighted estimates, anytime
          eta_t = sqrt(ln NM / (t NM)); sequential, compiled with numba.
  exp3ix: bandit feedback with implicit exploration (Neu 2015): the same loss normalization and step, estimates
          loss * 1{a} / (P[a] + gamma_t) with gamma_t = eta_t / 2. This is the stabilized bandit family B of the revised
          Section 4 (Niccolò's V05, 2026-11-02) with schedule bands a = 1, b = 1/2, prior uniform, so its certified
          inputs (rules tailB and momentB in sim_infer.py) apply; plain exp3 is outside that family.

Recorded at every checkpoint N (cumulative from period 1): Z counts over (multiplier, context); per context the
counterfactual sums over all visits Sa[k] = sum e_k v and Sc[k] = sum e_k price for every multiplier k; the realized
sums Ra = sum e_a v and Rc = sum e_a price; visit counts; win counts. Payoffs at any value lambda follow as
lambda * Sa - Sc and lambda * Ra - Rc, so both inference regimes and the regret diagnostics use sample quantities.
Usage: python sim_run.py <env_npz> <out_npz> <learner> <N_total> [seed=1] [mode=iid]
  mode 'iid': context drawn from the fitted lognormal masses psi0 and the rival triple drawn independently (Design A main).
  mode 'paired': one recorded opportunity drawn per auction, its own context bin (env ctx_idx) and its own rival triple
  kept together (the paired-environment robustness row; the context truth is then the empirical masses emp_hist).
"""
import sys
import time

import numpy as np

env = np.load(sys.argv[1])
out = sys.argv[2]
learner = sys.argv[3]
N_total = int(float(sys.argv[4]))
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
mode = sys.argv[6] if len(sys.argv) > 6 else "iid"
if mode not in ("iid", "paired"):
    raise SystemExit("mode must be iid or paired")
# Optional 7th argument: path to an audit JSON (audit_env_role.py) whose 'width_exact' gives, per context, the exact
# width umax - g_min of the reward support over every multiplier and every rival triple of the pool. With it, the bandit
# learners normalize losses by that width, so losses lie in [0, 1] as family B requires (finding I1, Niccolò, 4 November
# 2026); without it the historical normalizer K (within-rival cross-multiplier span) is kept, and losses can exceed one.
ixnorm = "env_K"
L_EXACT = None
if len(sys.argv) > 7 and sys.argv[7] not in ("", "env_K"):
    import json
    with open(sys.argv[7]) as fh:
        L_EXACT = np.array(json.load(fh)["width_exact"], dtype=np.float64)
    ixnorm = "exact:" + sys.argv[7]
CHECKS = sorted(set([int(x) for x in [1e5, 4.8e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9] if x <= N_total] + [N_total]))

alphas, vgrid, psi0, lam0, R, K, E, RES = (env["alphas"], env["vgrid"], env["psi0"], float(env["lam0"]), env["R"].astype(np.float64),
                                          env["K"], env["E"], float(env["RES"]))
NM, NV = len(alphas), len(vgrid)
Ksafe = np.maximum(K, 1e-12)          # contexts where every multiplier pays zero have K = 0
rng = np.random.default_rng(seed)
r1, r2, r3 = R[:, 0], R[:, 1], R[:, 2]
lnK = np.log(NM)


def outcome_vectors(j, ridx):
    """Exposure and price of every multiplier for context j against rival triples ridx: (NM, n) each."""
    v = vgrid[j]
    b = alphas[:, None] * v
    R1, R2, R3 = r1[ridx][None, :], r2[ridx][None, :], r3[ridx][None, :]
    s1 = b > R1
    s2 = (~s1) & (b > R2)
    s3 = (~s1) & (~s2) & (b > R3)
    price = np.maximum(np.where(s1, R1, np.where(s2, R2, np.where(s3, R3, 0.0))), RES)
    e = np.where(s1, E[0], np.where(s2, E[1], np.where(s3, E[2], 0.0)))
    return e, price


t0 = time.time()
if mode == "paired":
    if "ctx_idx" not in env.files:
        raise SystemExit("paired mode needs an environment with ctx_idx (rebuild with sim_env2.py)")
    ridx = rng.integers(0, len(R), size=N_total, dtype=np.int32)
    ctx = env["ctx_idx"][ridx].astype(np.int8)
else:
    ctx = rng.choice(NV, size=N_total, p=psi0).astype(np.int8)
    ridx = rng.integers(0, len(R), size=N_total, dtype=np.int32)
results = {}
Sa = np.zeros((NV, NM)); Sc = np.zeros((NV, NM))
Ra = np.zeros(NV); Rc = np.zeros(NV)
cnt = np.zeros(NV, dtype=np.int64); win_cnt = np.zeros(NV)
Z = np.zeros((NM, NV), dtype=np.int64)


def snapshot(Nc):
    results[Nc] = dict(Z=Z.copy(), Sa=Sa.copy(), Sc=Sc.copy(), Ra=Ra.copy(), Rc=Rc.copy(), cnt=cnt.copy(), win=win_cnt.copy())


if learner == "hedge":
    CHUNK = 2_000_000
    bounds = sorted(set(list(range(CHUNK, N_total, CHUNK)) + CHECKS))
    start = 0
    for stop in bounds:
        c = ctx[start:stop]; ri = ridx[start:stop]
        for j in range(NV):
            m = np.nonzero(c == j)[0]
            if len(m) == 0:
                continue
            e, p = outcome_vectors(j, ri[m])                     # NM x n_j
            v = vgrid[j]
            Ua = e * v; Uc = e * p                              # value and cost components
            U = lam0 * Ua - Uc
            n_j = len(m)
            t = cnt[j] + np.arange(n_j)
            S_prev = (lam0 * Sa[j] - Sc[j])[:, None] + np.concatenate([np.zeros((NM, 1)), np.cumsum(U, axis=1)[:, :-1]], axis=1)
            eta = np.sqrt(lnK / np.maximum(t, 1))
            logits = eta[None, :] * S_prev / Ksafe[j]
            logits -= logits.max(axis=0, keepdims=True)
            P = np.exp(logits); P /= P.sum(axis=0, keepdims=True)
            u = rng.random(n_j)
            a = np.minimum((np.cumsum(P, axis=0) < u[None, :]).sum(axis=0), NM - 1)
            ar = np.arange(n_j)
            Ra[j] += Ua[a, ar].sum(); Rc[j] += Uc[a, ar].sum()
            win_cnt[j] += (e[a, ar] > 0).sum()
            Sa[j] += Ua.sum(axis=1); Sc[j] += Uc.sum(axis=1)
            cnt[j] += n_j
            np.add.at(Z[:, j], a, 1)
        if stop in CHECKS:
            snapshot(stop)
            print(f"  checkpoint {stop:>12,d}  ({time.time() - t0:5.0f}s)", flush=True)
        start = stop
elif learner in ("exp3", "exp3ix"):
    from numba import njit

    @njit(cache=True)
    def run_exp3(ctx, ridx, r1, r2, r3, alphas, vgrid, Kv, lam0, E1, E2, E3, RESf, checks, seed_, ix):
        np.random.seed(seed_)
        NVl, NMl = len(vgrid), len(alphas)
        Sest = np.zeros((NVl, NMl)); Sa_ = np.zeros((NVl, NMl)); Sc_ = np.zeros((NVl, NMl))
        Ra_ = np.zeros(NVl); Rc_ = np.zeros(NVl); winv = np.zeros(NVl); cntv = np.zeros(NVl, dtype=np.int64)
        Zc = np.zeros((NMl, NVl), dtype=np.int64)
        nchk = len(checks)
        Z_out = np.zeros((nchk, NMl, NVl), dtype=np.int64); Sa_out = np.zeros((nchk, NVl, NMl)); Sc_out = np.zeros((nchk, NVl, NMl))
        Ra_out = np.zeros((nchk, NVl)); Rc_out = np.zeros((nchk, NVl)); cnt_out = np.zeros((nchk, NVl), dtype=np.int64); win_out = np.zeros((nchk, NVl))
        lnKl = np.log(NMl)
        P = np.zeros(NMl); ea = np.zeros(NMl); pa = np.zeros(NMl)
        umax = np.zeros(NVl)                      # per-context maximum one-period payoff (top slot at the reserve)
        for jj in range(NVl):
            umax[jj] = E1 * (lam0 * vgrid[jj] - RESf)
            if umax[jj] < 0.0:
                umax[jj] = 0.0
        ic = 0
        for n in range(len(ctx)):
            j = ctx[n]; i = ridx[n]; v = vgrid[j]
            t = cntv[j] + 1
            eta = np.sqrt(lnKl / (t * NMl))
            mx = -1e300
            for k in range(NMl):
                P[k] = eta * Sest[j, k]
                if P[k] > mx:
                    mx = P[k]
            s = 0.0
            for k in range(NMl):
                P[k] = np.exp(P[k] - mx); s += P[k]
            for k in range(NMl):
                P[k] /= s
            u = np.random.random(); a = NMl - 1; c = 0.0
            for k in range(NMl):
                c += P[k]
                if u < c:
                    a = k; break
            for k in range(NMl):
                b = alphas[k] * v
                if b > r1[i]:
                    e = E1; price = r1[i]
                elif b > r2[i]:
                    e = E2; price = r2[i]
                elif b > r3[i]:
                    e = E3; price = r3[i]
                else:
                    e = 0.0; price = 0.0
                if price < RESf:
                    price = RESf
                ea[k] = e * v; pa[k] = e * price
            g = lam0 * ea[a] - pa[a]
            # loss-based EXP3 on losses in [0, 1]: loss = (umax_j - u) / K_j, importance-weighted; Sest accumulates -loss_hat
            # exp3ix (ix == 1): implicit exploration, loss_hat = loss / (P[a] + gamma), gamma = eta / 2
            loss = (umax[j] - g) / Kv[j]
            if ix == 1:
                Sest[j, a] -= loss / (P[a] + 0.5 * eta)
            else:
                Sest[j, a] -= loss / P[a]
            for k in range(NMl):
                Sa_[j, k] += ea[k]; Sc_[j, k] += pa[k]
            Ra_[j] += ea[a]; Rc_[j] += pa[a]; cntv[j] += 1; Zc[a, j] += 1
            if ea[a] > 0:
                winv[j] += 1.0
            if ic < nchk and n + 1 == checks[ic]:
                Z_out[ic] = Zc; Sa_out[ic] = Sa_; Sc_out[ic] = Sc_; Ra_out[ic] = Ra_; Rc_out[ic] = Rc_; cnt_out[ic] = cntv; win_out[ic] = winv
                ic += 1
        return Z_out, Sa_out, Sc_out, Ra_out, Rc_out, cnt_out, win_out
    chk = np.array(CHECKS, dtype=np.int64)
    Knorm = np.maximum(L_EXACT, 1e-12) if L_EXACT is not None else Ksafe
    Zo, Sao, Sco, Rao, Rco, co, wo = run_exp3(ctx, ridx, r1, r2, r3, alphas, vgrid, Knorm, lam0, float(E[0]), float(E[1]), float(E[2]), RES, chk, seed + 7,
                                              1 if learner == "exp3ix" else 0)
    for ic, Nc in enumerate(chk):
        results[int(Nc)] = dict(Z=Zo[ic], Sa=Sao[ic], Sc=Sco[ic], Ra=Rao[ic], Rc=Rco[ic], cnt=co[ic], win=wo[ic])
    print(f"  {learner} done, {time.time() - t0:.0f}s")
else:
    raise SystemExit("learner must be hedge, exp3 or exp3ix")

np.savez(out, checks=np.array(sorted(results)), learner=learner, seed=seed, mode=mode, ixnorm=ixnorm,
         L_exact=(L_EXACT if L_EXACT is not None else np.zeros(0)),
         **{f"{key}_{Nc}": results[Nc][key] for Nc in results for key in ["Z", "Sa", "Sc", "Ra", "Rc", "cnt", "win"]})
print(f"\nlearner={learner} seed={seed} total={N_total:,d} ({time.time() - t0:.0f}s)")
for Nc in sorted(results):
    r = results[Nc]
    S = lam0 * r["Sa"] - r["Sc"]; real = lam0 * r["Ra"] - r["Rc"]
    best = S.max(axis=1); reg = np.maximum(best - real, 0.0)
    rel = np.where(best > 0, reg / np.maximum(best, 1e-15), np.nan)
    med = [alphas[np.argmax(np.cumsum(r["Z"][:, j]) >= r["Z"][:, j].sum() / 2)] / lam0 if r["Z"][:, j].sum() > 0 else np.nan for j in range(NV)]
    print(f"N={Nc:>11,d}: rel. regret by context {np.round(rel, 4)} | median mult/lambda0 {np.round(med, 3)} | "
          f"win rate {np.round(r['win'] / np.maximum(r['cnt'], 1), 3)} | uncond. regret x1e6 {np.round(reg / Nc * 1e6, 3)}")
