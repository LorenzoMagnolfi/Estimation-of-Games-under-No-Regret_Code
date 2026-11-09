"""Q20 constants for GadgetPickup magnitudes (units of K per clock round)."""
import math
import numpy as np

d = 9
Breg, sig, c = math.log(d), 1.0, 1.0
dH = dM = 0.025

def clock(H, tau, label):
    s = np.arange(H + 1, H + tau + 1, dtype=float)
    n = s - 1
    ftrl = math.sqrt(Breg / sig) / c * np.sqrt(s) / n
    # Q20-S as written, fixed-window allocation delta_s = dH / tau
    conc_fix = np.sqrt(2 * np.log(2 * d * tau / dH) / n)
    # Q20-S as written, pi^2 allocation over all s >= 2
    conc_pi = np.sqrt(2 * np.log(2 * d * math.pi**2 * n**2 / (6 * dH)) / n)
    # sharpening 1: one-sided differences against a*, union over d-1, fixed allocation
    conc_fix1 = np.sqrt(2 * np.log((d - 1) * tau / dH) / n)
    # sharpening 2: Doob maximal (one event for all n in [H, H+tau-1]), one-sided, d-1
    N = H + tau - 1
    conc_doob = np.sqrt(2 * N * math.log((d - 1) / dH)) / n
    # expectation route: E rho_s <= K [ftrl + sqrt(2 ln d / n)]
    conc_E = np.sqrt(2 * math.log(d) / n)
    cap = lambda x: np.minimum(x, 1.0)
    out = {}
    out['ftrl_sum'] = ftrl.sum()
    out['S_fix'] = cap(ftrl + conc_fix).sum()
    out['S_pi'] = cap(ftrl + conc_pi).sum()
    out['S_fix1'] = cap(ftrl + conc_fix1).sum()
    out['S_doob'] = cap(ftrl + conc_doob).sum()
    out['E_sum'] = cap(ftrl + conc_E).sum()
    out['mart'] = math.sqrt(2 * tau * math.log(d / dM))
    out['split'] = math.sqrt(2 * tau * math.log(d / (dM / 2))) + math.sqrt(tau * math.log(1 / (dM / 2)) / 2)
    impl_start = 2 * math.sqrt(math.log(d) / H)
    impl_window = 2 * math.sqrt(math.log(d)) * 2 * (math.sqrt(H + tau) - math.sqrt(H)) / tau
    print(f'\n=== {label}: H={H}, tau={tau} ===')
    print(f'per-round b at s=H+1 (fix alloc): ftrl {ftrl[0]:.5f} conc {conc_fix[0]:.5f}; pi2 conc {conc_pi[0]:.5f}; at s=H+tau pi2 conc {conc_pi[-1]:.5f}')
    for k, v in out.items():
        print(f'{k:10s} window sum {v:12.3f}   per round {v / tau:.5f}')
    print(f'direct total per round (fix) {(out["S_fix"] + out["mart"]) / tau:.5f}; (pi2) {(out["S_pi"] + out["mart"]) / tau:.5f}; (doob sharp) {(out["S_doob"] + out["mart"]) / tau:.5f}')
    print(f'impl per round start {impl_start:.5f}; impl window-mode {impl_window:.5f}')
    print(f'ratios to impl_start: S_fix {out["S_fix"]/tau/impl_start:.2f}, S_pi {out["S_pi"]/tau/impl_start:.2f}, E {out["E_sum"]/tau/impl_start:.2f}, direct(fix) {(out["S_fix"]+out["mart"])/tau/impl_start:.2f}, direct(pi) {(out["S_pi"]+out["mart"])/tau/impl_start:.2f}')
    print(f'ln terms: ln(2d tau/dH)={math.log(2*d*tau/dH):.3f}, ln(d/dM)={math.log(d/dM):.3f}')
    return out

clock(49196, 10120, 'SKU-day clock')
Hday_exact = 7939 / (1642 / 164)
print('\nH_day exact', Hday_exact, 'cells/day', 10120 / 164)
clock(int(math.floor(Hday_exact)), 164, 'Day clock (floor H)')
clock(int(round(Hday_exact)), 164, 'Day clock (round H)')

# Counterexample: per-round realized-range normalization changes the limit action.
rng = np.random.default_rng(1)
T = 200000
st = rng.random(T) < 0.4
g = np.where(st[:, None], np.array([0.0, 1.0]), np.array([0.5, 0.0]))
rng_ = g.max(1) - g.min(1)
z = ((g.max(1) + g.min(1))[:, None] / 2 - g) / rng_[:, None]
L = np.cumsum(z, 0)
eta = math.sqrt(math.log(2) / T)
w = np.exp(-eta * (L[-1] - L[-1].min())); p = w / w.sum()
mu = g.mean(0)
print('\ncounterexample: mean payoffs', mu, 'final mixture', p, 'pseudo-regret', mu.max() - p @ mu)
