import sys
import numpy as np
sys.argv = ['x', '60000', '0.7']
exec(open('probe_edgeworth.py').read().split("N = int(sys.argv[1])")[0])   # game (k = 0.7), psi, run (decreasing step)
src = open('probe_adaptive.py').read()
exec(src[src.index('def run_const'):src.index('def bne_two_step')])          # recency-weighted learner
def realized_regret(d):
    """Per seller and cost type: best fixed price in hindsight against the realized rival prices in that type's
    periods, minus realized profit; averaged over all periods (the paper's normalization) and as a share of profit."""
    out = []
    for i in range(2):
        own, riv, typ = d[:, i], d[:, 1 - i], d[:, 2 + i]
        tot_prof = U[typ, own, riv].mean()
        for t in range(nT):
            m = typ == t
            if m.sum() == 0: continue
            fact = U[t, own[m], riv[m]].sum()
            cf = U[t][:, riv[m]].sum(1)                     # payoff of each fixed price against realized rivals
            out.append((i, T[t], m.mean(), (cf.max() - fact) / len(d), P[cf.argmax()], tot_prof))
    return out
N = 60000
for lab, d in [('decreasing-step Hedge (family F)', run(N)[N // 2:]),
               ('recency-weighted, step 0.5', run_const(N, 0.5)[N // 2:]),
               ('recency-weighted, step 2.0', run_const(N, 2.0)[N // 2:])]:
    r = realized_regret(d)
    tot = sum(x[3] for x in r if x[0] == 0), sum(x[3] for x in r if x[0] == 1)
    prof = r[0][5], [x for x in r if x[0] == 1][0][5]
    print(f'{lab}: realized regret summed over cost types, per period: seller 1 {tot[0]:+.4f} ({100 * tot[0] / prof[0]:+.2f}% of profit), '
          f'seller 2 {tot[1]:+.4f} ({100 * tot[1] / prof[1]:+.2f}%)')
    for x in r:
        if x[0] == 0 and x[2] > .05:
            print(f'    seller 1, cost {x[1]}: share {x[2]:.3f}, regret {x[3]:+.4f}, best fixed price {x[4]:.0f}')
