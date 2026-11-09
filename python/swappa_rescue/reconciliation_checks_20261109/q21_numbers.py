"""Q21 numbers: window regret bounds per window round, in units of the common span K.

Scratch script for 04_Q21_CONSTRUCTION.md. No data are read.
"""
import math

import numpy as np


def h(x):
    """Binary entropy in nats."""
    return -x * math.log(x) - (1 - x) * math.log(1 - x)


def fs_const(tau, d, eta, alpha):
    """Fixed share, constant (eta, alpha): interval bound in units of K (CBGLS 2012 Thm 2 specialised)."""
    return (math.log(d / alpha) + (tau - 1) * math.log(1 / (1 - alpha))) / eta + eta * tau / 8


def fs_tuned(tau0, d):
    """Return (eta, alpha) of CBGLS Corollary 2 tuning to design length tau0, and L(tau0)."""
    L = math.log(d) + tau0 * h(1 / tau0)
    return math.sqrt(8 * L / tau0), 1 / tau0, L


def rows(label, d, H, tau, delta=0.025, alpha_aw=0.05):
    T = H + tau
    out = {}
    eta, a, L = fs_tuned(tau, d)
    out['FS-c tuned to tau: exact L'] = math.sqrt(tau * L / 2) / tau
    out['  check via fs_const'] = fs_const(tau, d, eta, a) / tau
    out['FS-c tuned to tau: relaxed ln(e d tau)'] = math.sqrt(tau * math.log(math.e * d * tau) / 2) / tau
    conv = math.sqrt(tau * math.log(1 / delta) / 2) / tau
    out['conversion eq:s4mixedtail, fixed tau'] = conv
    out['FS-c total (exact L + conversion)'] = out['FS-c tuned to tau: exact L'] + conv
    out['FS-1/t, eta tuned to tau: ln(d T_life)'] = math.sqrt(tau * math.log(d * T) / 2) / tau
    out['FS-1/t total'] = out['FS-1/t, eta tuned to tau: ln(d T_life)'] + conv
    out['FS-1/t Adamskiy param: ln((d-1) T_life)'] = math.sqrt(tau * math.log((d - 1) * T) / 2) / tau
    out['CBGLS Cor 6 anytime (T = T_life)'] = (math.sqrt(2 * T * math.log(d * T)) + 3 * math.log(3 * d)) / tau
    c = math.sqrt(math.log(d) / 2)
    lead = 4 / (math.sqrt(2) - 1) * c * math.sqrt(tau)
    out['CBCE Thm 2 (Hedge black box)'] = (lead + 8 * math.sqrt(tau * (7 * math.log(T) + 5))) / tau
    out['SAOL Thm 1 (log2)'] = (lead + 40 * math.log2(T + 1) * math.sqrt(tau)) / tau
    t1 = H + 1
    lnq = math.log(d * t1 ** 2 * math.pi ** 2 / 6)
    lnB = math.log(2.5 + 1.5 * math.log(1 + T))
    out['AdaNormalHedge.TV derived (Thm 1 eq 2 + Thm 3)'] = math.sqrt(3 * tau * (lnq + lnB + 1)) / tau
    out['Uncertified AW, common K'] = math.sqrt(tau) * (math.sqrt(2 * math.log(d * T)) + math.sqrt(2 * math.log(d / alpha_aw))) / tau
    out['  AW first term only'] = math.sqrt(tau) * math.sqrt(2 * math.log(d * T)) / tau
    out['  AW second term only'] = math.sqrt(tau) * math.sqrt(2 * math.log(d / alpha_aw)) / tau
    print(f'\n=== {label}: d={d}, H={H:.1f}, tau={tau}, T_life={T:.1f}, eta*={eta:.5f}, alpha*={a:.6f}, L={L:.4f}')
    for k, v in out.items():
        print(f'{k:50s} {v:.4f}   (cumulative {v * tau:.1f})')
    return out


def mistune(d, tau, factors):
    print(f'\n--- mis-tuning, tau={tau}: per-round FS-c mixed bound when tuned to tau0 = f * tau')
    for f in factors:
        tau0 = max(2, round(f * tau))
        eta, a, _ = fs_tuned(tau0, d)
        print(f'f={f:<6} tau0={tau0:<7d} D/tau={fs_const(tau, d, eta, a) / tau:.4f}')


def prefix_hedge_example(d, H, tau, eta_rule='lifetime'):
    """Hedge on a prefix where action 0 is best, then action 1 best in the window. Unit span."""
    T = H + tau
    logw = np.zeros(d)
    reg = 0.0
    regfs = 0.0
    eta_fs, a_fs, _ = fs_tuned(tau, d)
    p_fs = np.full(d, 1 / d)
    for t in range(1, T + 1):
        g = np.zeros(d)
        if t <= H:
            g[0] = 1.0
        else:
            g[1] = 1.0
        eta = math.sqrt(8 * math.log(d) / T) if eta_rule == 'lifetime' else math.sqrt(8 * math.log(d) / t)
        p = np.exp(logw - logw.max()); p /= p.sum()
        if t > H:
            reg += g[1] - p @ g
            regfs += g[1] - p_fs @ g
        logw += eta * g  # payoffs: Hedge on gains
        v = p_fs * np.exp(eta_fs * (g - g.max())); v /= v.sum()
        p_fs = a_fs / d + (1 - a_fs) * v
    print(f'prefix Hedge ({eta_rule} step), d={d}, H={H}, tau={tau}: window regret per round {reg / tau:.4f}; '
          f'fixed share tuned to tau on same path: {regfs / tau:.4f}')


def random_check(d, H, tau, reps=20, seed=0):
    """Fixed share on drifting random payoffs: max window regret against FS-c bound (sanity check, not a proof)."""
    rng = np.random.default_rng(seed)
    eta, a, L = fs_tuned(tau, d)
    bound = math.sqrt(tau * L / 2)
    worst = -1e9
    for _ in range(reps):
        T = H + tau
        p = np.full(d, 1 / d)
        cum = np.zeros(d); learner = 0.0
        mu = rng.uniform(0, 1, d)
        for t in range(1, T + 1):
            if rng.uniform() < 0.001:
                mu = rng.uniform(0, 1, d)
            g = np.clip(mu + rng.normal(0, 0.3, d), 0, 1)
            if t > H:
                cum += g; learner += p @ g
            v = p * np.exp(eta * (g - g.max())); v /= v.sum()
            p = a / d + (1 - a) * v
        worst = max(worst, (cum.max() - learner))
    print(f'random drifting check d={d} H={H} tau={tau}: worst window mixed regret {worst:.1f} vs bound {bound:.1f}')


if __name__ == '__main__':
    d = 9
    rows('SKU-day clock', d, 49196, 10120)
    H_day = 7939 / (1642 / 164)
    rows('Day clock (units of K_day)', d, H_day, 164)
    mistune(d, 10120, [0.25, 0.5, 1, 2, 4])
    mistune(d, 164, [30 / 164, 0.5, 1, 2, 365 / 164])
    prefix_hedge_example(d, 49196, 10120, 'lifetime')
    prefix_hedge_example(d, 49196, 10120, 'anytime')
    prefix_hedge_example(d, 793, 164, 'lifetime')
    random_check(d, 2000, 1000, reps=10)
    print('\nH_day =', H_day, 'T_life_day =', H_day + 164, 'cells per day =', 10120 / 164)
