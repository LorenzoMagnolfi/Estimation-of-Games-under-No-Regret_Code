"""Feasibility probe: two-seller pricing lab (paper's game: logit alpha=-1/3, prices 4..8, costs {0,1.5,3,4.5,6},
psi = discretized N(3,1)). Learners: Hedge per context. Compare (a) i.i.d. costs, own-cost contexts (the paper's lab)
with (b) persistent costs (stay w.p. rho) and contexts (own cost, rival's last price). Report correlation of prices,
the BNE two-step estimate of (mu, sigma), and BCCE feasibility of the truth (LP)."""
import sys, time
import numpy as np
from scipy.optimize import linprog
P = np.arange(4., 9.); T = np.array([0, 1.5, 3, 4.5, 6]); al = -1/3; nA = nT = 5
def psi(mu, sd):
    w = np.exp(-0.5 * ((T - mu) / sd) ** 2); return w / w.sum()
E = np.exp(al * P)
S = E[:, None] / (1 + E[:, None] + E[None, :])            # S[a, b] sale prob of own price a vs rival b
U = S[None] * (P[None, :, None] - T[:, None, None])        # U[t, a, b]
K = (U.max(1) - U.min(1)).max()

def run(N, rho, informed, seed=1):
    rng = np.random.default_rng(seed); ps = psi(3, 1)
    nctx = nT * (nA if informed else 1)
    cum = np.zeros((2, nctx, nA)); cnt = np.zeros((2, nctx))
    t = rng.choice(nT, 2, p=ps); last = np.array([2, 2])
    out = np.zeros((N, 4), int)
    for n in range(N):
        a = np.zeros(2, int); ctx = np.zeros(2, int)
        for i in range(2):
            ctx[i] = t[i] * (nA if informed else 1) + (last[1 - i] if informed else 0)
            eta = np.sqrt(np.log(nA) / (cnt[i, ctx[i]] + 1)) / K
            z = eta * (cum[i, ctx[i]] - cum[i, ctx[i]].max()); p = np.exp(z); p /= p.sum()
            a[i] = rng.choice(nA, p=p)
        for i in range(2):
            cum[i, ctx[i]] += U[t[i], :, a[1 - i]]; cnt[i, ctx[i]] += 1
        out[n] = [a[0], a[1], t[0], t[1]]
        last = a.copy()
        for i in range(2):
            if rng.random() > rho: t[i] = rng.choice(nT, p=ps)
    return out[N // 10:]                                     # drop burn-in

def bne_two_step(q1, q2):
    """Beliefs = empirical marginal of rival prices; each cost type best-responds; choose (mu, sd) to match the
    observed own-price marginals (sum of squared differences over both sellers)."""
    best = None
    for mu in np.arange(1.5, 4.51, .025):
        for sd in np.arange(.4, 2.51, .025):
            w = psi(mu, sd); loss = 0
            for qo, qr in [(q1, q2), (q2, q1)]:
                br = (U @ qr).argmax(1)                       # best response per type
                qi = np.bincount(br, weights=w, minlength=nA); loss += ((qi - qo) ** 2).sum()
            if best is None or loss < best[0]: best = (loss, mu, sd)
    return best

def bcce_feasible(Q, mu, sd):
    """Exists Z(a1,a2,t1,t2) >= 0 with price marginal Q, type marginal psi x psi, coarse obedience per (i,t_i,a')."""
    w = psi(mu, sd); n = nA * nA * nT * nT
    idx = lambda a1, a2, t1, t2: ((a1 * nA + a2) * nT + t1) * nT + t2
    Aeq, beq = [], []
    for a1 in range(nA):
        for a2 in range(nA):
            r = np.zeros(n); r[[idx(a1, a2, t1, t2) for t1 in range(nT) for t2 in range(nT)]] = 1; Aeq.append(r); beq.append(Q[a1, a2])
    for t1 in range(nT):
        for t2 in range(nT):
            r = np.zeros(n); r[[idx(a1, a2, t1, t2) for a1 in range(nA) for a2 in range(nA)]] = 1; Aeq.append(r); beq.append(w[t1] * w[t2])
    Aub, bub = [], []
    for i in range(2):
        for ti in range(nT):
            for ap in range(nA):
                r = np.zeros(n)
                for a1 in range(nA):
                    for a2 in range(nA):
                        for tj in range(nT):
                            t1, t2 = (ti, tj) if i == 0 else (tj, ti)
                            own, riv = (a1, a2) if i == 0 else (a2, a1)
                            r[idx(a1, a2, t1, t2)] = U[ti, ap, riv] - U[ti, own, riv]
                Aub.append(r); bub.append(0.0)
    res = linprog(np.zeros(n), A_ub=np.array(Aub), b_ub=bub, A_eq=np.array(Aeq), b_eq=beq, bounds=(0, None), method='highs')
    return res.status == 0

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
for lab, rho, inf in [('paper lab: iid costs, own-cost contexts', 0.0, False),
                      ('persistent costs (rho=.9), own-cost contexts', 0.9, False),
                      ('persistent costs (rho=.9), contexts = own cost + rival last price', 0.9, True)]:
    t0 = time.time(); d = run(N, rho, inf)
    Q = np.zeros((nA, nA)); np.add.at(Q, (d[:, 0], d[:, 1]), 1); Q /= Q.sum()
    q1, q2 = Q.sum(1), Q.sum(0); x = P
    cov = (Q * np.outer(x - q1 @ x, x - q2 @ x)).sum(); corr = cov / np.sqrt((q1 @ (x - q1 @ x) ** 2) * (q2 @ (x - q2 @ x) ** 2))
    c2 = np.corrcoef(d[:, 0], d[:, 3])[0, 1]
    loss, mu, sd = bne_two_step(q1, q2)
    print(f'{lab}: corr(p1,p2)={corr:+.3f}, corr(p1,t2)={c2:+.3f}, mean price {q1 @ x:.2f}/{q2 @ x:.2f}; '
          f'BNE two-step (mu,sd)=({mu:.2f},{sd:.2f}) loss {loss:.4f}; BCCE truth feasible: {bcce_feasible(Q, 3, 1)}; '
          f'BCCE at BNE estimate feasible: {bcce_feasible(Q, mu, sd)}; {time.time() - t0:.0f}s', flush=True)
