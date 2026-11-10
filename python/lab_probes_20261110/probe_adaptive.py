import sys
import numpy as np
exec(open('probe_edgeworth.py').read().split("N = int(sys.argv[1])")[0])   # reuse game, psi, min_slack
def run_const(N, step, seed=1):
    rng = np.random.default_rng(seed); ps = psi(3, 1)
    cum = np.zeros((2, nT, nA)); out = np.zeros((N, 4), int)
    for n in range(N):
        t = rng.choice(nT, 2, p=ps); a = np.zeros(2, int)
        for i in range(2):
            z = (step / K) * (cum[i, t[i]] - cum[i, t[i]].max()); p = np.exp(z); p /= p.sum(); a[i] = rng.choice(nA, p=p)
        for i in range(2):
            cum[i, t[i]] = 0.999 * cum[i, t[i]] + U[t[i], :, a[1 - i]]       # recency weighting (memory about 1,000 own-type rounds)
        out[n] = [a[0], a[1], t[0], t[1]]
    return out
def bne_two_step(q1, q2):
    best = None
    for mu in np.arange(1.5, 4.51, .025):
        for sd in np.arange(.4, 2.51, .025):
            w = psi(mu, sd); loss = 0
            for qo, qr in [(q1, q2), (q2, q1)]:
                br = (U @ qr).argmax(1); qi = np.bincount(br, weights=w, minlength=nA); loss += ((qi - qo) ** 2).sum()
            if best is None or loss < best[0]: best = (loss, mu, sd)
    return best
N = int(sys.argv[1]); step = float(sys.argv[3])
d = run_const(N, step)[N // 2:]
Z = np.zeros((nA, nA, nT, nT)); np.add.at(Z, (d[:, 0], d[:, 1], d[:, 2], d[:, 3]), 1); Z /= Z.sum()
Q = Z.sum((2, 3)); q1, q2 = Q.sum(1), Q.sum(0)
lag = np.corrcoef(P[d[1:, 0]], P[d[:-1, 1]])[0, 1]
print(f'step {step}: corr(p1,p2) {np.corrcoef(P[d[:, 0]], P[d[:, 1]])[0, 1]:+.3f}; corr(p1_t, p2_t-1) {lag:+.3f}; TV(joint, product) {.5 * np.abs(Q - np.outer(q1, q2)).sum():.3f}; mean price {q1 @ P:.2f}')
print(np.round(Q, 3))
loss, mu, sd = bne_two_step(q1, q2); print(f'two-step equilibrium estimate (mu, sd) = ({mu:.2f}, {sd:.2f}), loss {loss:.4f}; truth (3, 1)')
grid = [(m, s) for m in [2.0, 2.5, 3.0, 3.5, 4.0] for s in [0.5, 1.0, 1.5]]
sl = {g: min_slack(Z, *g) for g in grid}
print('BCCE min slack (expected-profit units): truth', round(sl[(3.0, 1.0)], 4), '| zero-slack candidates', [g for g, v in sl.items() if v < 1e-6])
