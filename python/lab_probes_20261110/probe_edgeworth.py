"""Bertrand-Edgeworth probe: homogeneous good, unit mass of buyers with value above all prices, capacity k per seller.
Low-price seller sells k, high-price seller the residual 1-k, tie splits 1/2. Prices 4..8, costs {0,1.5,3,4.5,6},
psi = discretized N(3,1), i.i.d. across periods; Hedge per own-cost context (the paper's learner). Measures: correlation
of the two sellers' prices; total-variation distance of the joint from the product of marginals; within-type price
dispersion; minimal uniform slack making the truth BCCE-feasible vs the slack at other candidate (mu, sigma)."""
import sys
import numpy as np
from scipy.optimize import linprog
P = np.arange(4., 9.); T = np.array([0, 1.5, 3, 4.5, 6]); nA = nT = 5
k = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
def psi(mu, sd):
    w = np.exp(-0.5 * ((T - mu) / sd) ** 2); return w / w.sum()
Qs = np.where(P[:, None] < P[None, :], k, np.where(P[:, None] > P[None, :], 1 - k, 0.5))   # own sales [a, b]
U = Qs[None] * (P[None, :, None] - T[:, None, None])
K = (U.max(1) - U.min(1)).max()

def run(N, seed=1):
    rng = np.random.default_rng(seed); ps = psi(3, 1)
    cum = np.zeros((2, nT, nA)); cnt = np.zeros((2, nT)); out = np.zeros((N, 4), int)
    for n in range(N):
        t = rng.choice(nT, 2, p=ps); a = np.zeros(2, int)
        for i in range(2):
            eta = np.sqrt(np.log(nA) / (cnt[i, t[i]] + 1)) / K
            z = eta * (cum[i, t[i]] - cum[i, t[i]].max()); p = np.exp(z); p /= p.sum(); a[i] = rng.choice(nA, p=p)
        for i in range(2):
            cum[i, t[i]] += U[t[i], :, a[1 - i]]; cnt[i, t[i]] += 1
        out[n] = [a[0], a[1], t[0], t[1]]
    return out

def min_slack(Z4, mu, sd):
    """Smallest eps such that some Z(a1,a2,t1,t2) with the observed price marginal and type marginal psi x psi
    satisfies coarse obedience up to eps (expected-profit units)."""
    Q = Z4.sum((2, 3)); w = psi(mu, sd); n = nA * nA * nT * nT + 1
    idx = lambda a1, a2, t1, t2: ((a1 * nA + a2) * nT + t1) * nT + t2
    Aeq, beq, Aub, bub = [], [], [], []
    for a1 in range(nA):
        for a2 in range(nA):
            r = np.zeros(n); r[[idx(a1, a2, t1, t2) for t1 in range(nT) for t2 in range(nT)]] = 1; Aeq.append(r); beq.append(Q[a1, a2])
    for t1 in range(nT):
        for t2 in range(nT):
            r = np.zeros(n); r[[idx(a1, a2, t1, t2) for a1 in range(nA) for a2 in range(nA)]] = 1; Aeq.append(r); beq.append(w[t1] * w[t2])
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
                r[-1] = -1; Aub.append(r); bub.append(0.0)
    c = np.zeros(n); c[-1] = 1
    res = linprog(c, A_ub=np.array(Aub), b_ub=bub, A_eq=np.array(Aeq), b_eq=beq, bounds=[(0, None)] * n, method='highs')
    return res.fun if res.status == 0 else np.nan

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
d = run(N)
for lab, dd in [('first half', d[:N // 2]), ('second half', d[N // 2:])]:
    Z = np.zeros((nA, nA, nT, nT)); np.add.at(Z, (dd[:, 0], dd[:, 1], dd[:, 2], dd[:, 3]), 1); Z /= Z.sum()
    Q = Z.sum((2, 3)); q1, q2 = Q.sum(1), Q.sum(0)
    corr = np.corrcoef(P[dd[:, 0]], P[dd[:, 1]])[0, 1]; tv = .5 * np.abs(Q - np.outer(q1, q2)).sum()
    disp = np.mean([np.std(P[dd[dd[:, 2] == t, 0]]) for t in range(1, 4)])
    print(f'{lab}: corr(p1,p2) {corr:+.3f}; TV(joint, product) {tv:.3f}; mean within-type sd of price (types 1.5,3,4.5) {disp:.2f}; q1 {np.round(q1, 3)}')
    print('   joint price distribution:\n', np.round(Q, 3))
    grid = [(mu, sd) for mu in [2.0, 2.5, 3.0, 3.5, 4.0] for sd in [0.5, 1.0, 1.5]]
    sl = {g: min_slack(Z, *g) for g in grid}
    best = min(sl, key=sl.get)
    print(f'   min slack at truth (3,1): {sl[(3.0, 1.0)]:.4f}; smallest slack over grid at {best}: {sl[best]:.4f}; slacks:',
          {f'{g[0]},{g[1]}': round(v, 3) for g, v in sl.items()})
