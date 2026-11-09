"""Sanity check (not a proof): Q20-S tail and the expectation bound for Hedge, i.i.d. vectors, d=9."""
import math
import numpy as np
rng = np.random.default_rng(11)
d, reps, delta = 9, 4000, 0.025
gaps = np.linspace(0, 0.08, d)[::-1]                    # action 0 best; near-ties
worst_ratio_tail, worst_ratio_mean = 0, 0
for n in [20, 100, 500, 2000]:
    s = n + 1
    # Bernoulli payoffs in [0,1] with common span K=1
    G = (rng.random((reps, n, d)) < (0.5 + gaps / 2)).astype(float)
    mu = 0.5 + gaps / 2
    z = ((G.max(2, keepdims=True) + G.min(2, keepdims=True)) / 2 - G)  # K=1
    Lc = z.sum(1)
    eta = math.sqrt(math.log(d) / s)
    w = np.exp(-eta * (Lc - Lc.min(1, keepdims=True))); p = w / w.sum(1, keepdims=True)
    rho = mu.max() - p @ mu
    b = math.sqrt(math.log(d)) * math.sqrt(s) / n + math.sqrt(2 * math.log(2 * d / delta) / n)
    bE = math.sqrt(math.log(d)) * math.sqrt(s) / n + math.sqrt(2 * math.log(d) / n)
    print(f'n={n:5d}  P(rho>b)={np.mean(rho > b):.4f}  q99(rho)={np.quantile(rho, .99):.4f}  b={b:.4f}  E rho={rho.mean():.4f}  bE={bE:.4f}')
