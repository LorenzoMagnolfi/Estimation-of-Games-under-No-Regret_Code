"""Offline evaluation of regret allowances on stored Swappa moment arrays (9 November 2026).

Reads `output/swappa_core_20261108/reconciliation_diag_arrays_<spec>_<seller>.npz`, written by
`swappa_reconciliation_diag_20261109.py` (Unity job 11390437), and inverts the cost-ratio set for any
allowance formula without re-evaluating demand. Two routes:

  expectation  keep theta if (m_hat_k - eps(theta)) / sd_k <= z, sd from HAC(7)/T + demand delta method
               + demand-moment cross-covariance, Bonferroni over the comparator's moments;
  realized     keep theta if m_hat_k - z_D sd_D,k <= bound(theta), the pathwise window bound in daily
               units, with a Bonferroni delta-method projection of demand only (no HAC term).

All spans are evaluated at each candidate theta. `maxKday` / `maxKc` are window maxima, hence LOWER
bounds on any span a learner could know in advance; rows using them are optimistic for that route.

Usage: python swappa_allowance_eval_20261109.py [--arrays PATH] [--json OUT]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'output' / 'swappa_core_20261108' / 'reconciliation_diag_arrays_raw_d0_GadgetPickup.npz'
D = 9
PLATFORM_DAYS = 4900          # upper bound on days since the learner started: Swappa operating since 2010


def load(path):
    z = np.load(path, allow_pickle=False)
    out = {}
    for key in z.files:
        c, k = key.split('__', 1)
        out.setdefault(c, {})[k] = z[key]
    return out


def lrcov_blocks(a, th):
    S_h = (a['Haa'] - th * (a['Hab'] + a['Hab'].T) + th ** 2 * a['Hbb']) / float(a['T'])
    S_d = a['Daa'] - th * (a['Dab'] + a['Dab'].T) + th ** 2 * a['Dbb']
    Cx = a['Caa'] - th * (a['Cab'] + a['Cba']) + th ** 2 * a['Cbb']
    return S_h, S_d, Cx + Cx.T


def span(mask, TH):
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)
    return [float(TH[idx[0]]), float(TH[idx[-1]]), bool((np.diff(idx) > 1).any())]


def invert(a, allowance, route, alpha):
    TH = a['TH']
    n = len(a['mA'])
    ok = np.zeros(len(TH), bool)
    z = norm.ppf(1 - alpha / n)
    for gi, th in enumerate(TH):
        m = a['mA'] - th * a['mB']
        S_h, S_d, Cx = lrcov_blocks(a, th)
        if route == 'expectation':
            sd = np.sqrt(np.maximum(np.diag(S_h + S_d + Cx), 1e-300))
            ok[gi] = bool(((m - allowance[gi]) / sd <= z).all())
        else:
            sd = np.sqrt(np.maximum(np.diag(S_d), 0))
            ok[gi] = bool((m - z * sd <= allowance[gi]).all())
    return span(ok, TH)


def lerner(iv, a):
    if iv is None:
        return None
    r = float(a['mean_ref_sold']) / float(a['mean_price_sold'])
    return [round(100 * (1 - iv[1] * r), 2), round(100 * (1 - iv[0] * r), 2)]


def q20_factor(H, tau, delta_total, d=D):
    """Window sum of Q20-S per-round bounds (units of K), Hedge, fixed-window allocation."""
    s = np.arange(H + 1, H + tau + 1, dtype=float)
    f = np.sqrt(np.log(d)) * np.sqrt(s) / (s - 1) + np.sqrt(2 * np.log(2 * d * tau / delta_total) / (s - 1))
    return float(np.minimum(f, 1.0).sum())


def q20e_factor(H, tau, d=D):
    """Window sum of the delta-free Q20-E bounds on E rho_s (units of K), Hedge c = 1 (03_Q20_VERIFICATION.md)."""
    s = np.arange(H + 1, H + tau + 1, dtype=float)
    f = np.sqrt(np.log(d)) * np.sqrt(s) / (s - 1) + np.sqrt(2 * np.log(d) / (s - 1))
    return float(np.minimum(f, 1.0).sum())


def fixed_share_factor(tau, delta, d=D):
    """Fixed share tuned to the window (alpha = 1/tau, eta = sqrt(8 L / tau)), Cesa-Bianchi, Gaillard, Lugosi and
    Stoltz (2012), Thm 2 and Cor 2, as assembled in 04_Q21_CONSTRUCTION.md: mixed term sqrt((tau/2) L) and the
    actual-action conversion of eq:s4mixedtail sqrt((tau/2) ln(1/delta)), in units of a common span."""
    a_ = 1.0 / tau
    h = -a_ * np.log(a_) - (1 - a_) * np.log(1 - a_)
    L = np.log(d) + tau * h
    return float(np.sqrt(tau / 2 * L) + np.sqrt(tau / 2 * np.log(1 / delta))), float(L)


def q20e_bernstein_rms(a, H_scale=1.0, d=D):
    """Proposed main allowance (9 November, after Lorenzo's steer): SKU-day clock, expectation route, the
    delta-free Q20-E bound of 03_Q20_VERIFICATION.md with Bernstein in place of Hoeffding in the concentration
    step. The learner scales its updates by one constant, the root-mean-square cell span; the largest span enters
    only the lower-order Bernstein term 2 Kbar ln d / (3 (s - 1)). Daily units; H multiplied by H_scale (sensitivity)."""
    T, nC = float(a['T']), int(a['nC'])
    Hc = float(a['H_cell']) * H_scale
    s = np.arange(Hc + 1, Hc + nC + 1, dtype=float)
    lead = (np.sqrt(np.log(d)) * np.sqrt(s) / (s - 1) + np.sqrt(2 * np.log(d) / (s - 1))).sum()
    tail = (2 * np.log(d) / (3 * (s - 1))).sum()          # Bernstein scale term 2 Kbar / 3 (08_CLOSING_PROOFS.md, Lemma 1)
    return (lead * np.sqrt(a['sqKc'] / nC) + tail * a['maxKc']) / T


def invert_maxt(a, allowance, alpha=.05, ndraw=20000, seed=20261109):
    """Least-favorable max-t version of the expectation route (HAC + demand + cross covariance)."""
    TH = a['TH']
    n = len(a['mA'])
    Z0 = np.random.default_rng(seed).standard_normal((ndraw, n))
    ok = np.zeros(len(TH), bool)
    for gi, th in enumerate(TH):
        m = a['mA'] - th * a['mB']
        S_h, S_d, Cx = lrcov_blocks(a, th)
        S = S_h + S_d + Cx
        sd = np.sqrt(np.maximum(np.diag(S), 1e-300))
        w, U = np.linalg.eigh(S / np.outer(sd, sd))
        Z = Z0 @ (U * np.sqrt(np.maximum(w, 0))).T
        ok[gi] = bool(((m - allowance[gi]) / sd).max() <= np.quantile(Z.max(1), 1 - alpha))
    return span(ok, TH)


def standard_allowances(a):
    """Named allowances in daily profit units, each an array over the theta grid."""
    T, nC = float(a['T']), float(a['nC'])
    Hc, Hd = float(a['H_cell']), float(a['H_day'])
    l9 = np.log(D)
    out = {}
    # expectation route
    out[('expectation', 'E_placeholder_cell_local')] = 2 * np.sqrt(l9 / Hc) * a['sumKc'] / T
    out[('expectation', 'E_placeholder_day_local')] = 2 * np.sqrt(l9 / Hd) * a['meanKday']
    out[('expectation', 'E_placeholder_day_common')] = 2 * np.sqrt(l9 / Hd) * a['maxKday']
    out[('expectation', 'Q20S_day_local')] = a['meanKday'] * q20_factor(Hd, int(T), .025) / T
    out[('expectation', 'Q20S_day_common')] = a['maxKday'] * q20_factor(Hd, int(T), .025) / T
    out[('expectation', 'Q20E_day_common')] = a['maxKday'] * q20e_factor(Hd, int(T)) / T
    out[('expectation', 'Q20E_day_local')] = a['meanKday'] * q20e_factor(Hd, int(T)) / T          # local span: not licensed
    out[('expectation', 'Q20E_cell_common')] = a['maxKc'] * q20e_factor(Hc, int(nC)) / T
    out[('expectation', 'Q20S_day_rms')] = np.sqrt(a['sqKday'] / T) * q20_factor(Hd, int(T), .025) / T   # needs a heterogeneous-span Q20
    # realized route (pathwise), daily clock
    cm = np.sqrt(2 * T * np.log(D / .0125))
    out[('realized', 'Q20W_day_local')] = a['meanKday'] * (q20_factor(Hd, int(T), .0125) + cm) / T
    out[('realized', 'Q20W_day_common')] = a['maxKday'] * (q20_factor(Hd, int(T), .0125) + cm) / T
    tail = np.sqrt(2 * np.log(D / .025))
    for life, lab in [(Hd + T, 'Hratings'), (PLATFORM_DAYS + T, 'platformage')]:
        out[('realized', f'AWform_day_localsq_{lab}')] = np.sqrt(a['sqKday']) * (np.sqrt(2 * np.log(D * life)) + tail) / T
        out[('realized', f'AWform_day_common_{lab}')] = np.sqrt(T) * a['maxKday'] * (np.sqrt(2 * np.log(D * life)) + tail) / T
    fsd, Ld = fixed_share_factor(int(T), .025)
    out[('realized', 'FS_day_common')] = a['maxKday'] * fsd / T
    out[('realized', 'FS_day_local_mean')] = a['meanKday'] * fsd / T                  # not licensed
    out[('realized', 'FS_day_rss_pretuned')] = np.sqrt(a['sqKday']) * (np.sqrt(Ld / 2) + np.sqrt(np.log(1 / .025) / 2)) / T  # needs step tuned to sum K^2 in advance
    out[('realized', 'AWform_cell_localsq_Hratings')] = np.sqrt(a['sqKc']) * (np.sqrt(2 * np.log(D * (Hc + nC))) + tail) / T
    return out


def phi_local_rows(a, idx, delta_R=.0125, delta_D=.025):
    """HYPOTHETICAL interval Phi learner (Gordon-Greenwald-Marks style, 04_Q21_CONSTRUCTION.md Section 6): an
    external fixed-share learner over the maps in Phi with per-round span 2 max_phi K_t(phi), heterogeneous and
    pretuned, plus the comparator-specific Azuma conversion with a union over Phi. K_t(phi) is evaluated at the
    realized action only (|gain of phi on day t|), a LOWER bound on max_a |g_t(phi(a)) - g_t(a)|. No theorem yet."""
    TH, A, B, T = a['TH'], a['A'], a['B'], float(a['T'])
    nP = len(idx)
    _, L = fixed_share_factor(int(T), delta_R, d=nP + 1)
    z = norm.ppf(1 - delta_D / nP)
    ok = np.zeros(len(TH), bool)
    for gi, th in enumerate(TH):
        G = (A - th * B)[idx]
        Kphi = np.abs(G)
        eps = (np.sqrt(2 * L * (Kphi.max(0) ** 2).sum()) + np.sqrt(2 * (Kphi ** 2).sum(1) * np.log(nP / delta_R))) / T
        m = (a['mA'] - th * a['mB'])[idx]
        _, S_d, _ = lrcov_blocks(a, th)
        sd = np.sqrt(np.maximum(np.diag(S_d), 0))[idx]
        ok[gi] = bool((m - z * sd <= eps).all())
    return span(ok, TH)


def main():
    args = sys.argv[1:]
    path = Path(args[args.index('--arrays') + 1]) if '--arrays' in args else DEFAULT
    out_json = Path(args[args.index('--json') + 1]) if '--json' in args else None
    arr = load(path)
    res = {}
    for cname, a in arr.items():
        TH = a['TH']
        mid = int(np.argmin(np.abs(TH - .8)))
        prof08 = None
        res[cname] = {}
        for (route, name), eps in standard_allowances(a).items():
            alpha = .05 if name.startswith('E_placeholder') else .025
            iv = invert(a, eps, route, alpha)
            res[cname][name] = dict(route=route, alpha_sampling_or_demand=alpha, theta=iv, lerner=lerner(iv, a),
                                    allowance_at_theta08=float(eps[mid]))
            print(f'{cname:8s} {route:11s} {name:34s} {lerner(iv, a)}  eps(0.8)={eps[mid]:.1f}', flush=True)
    if 'phi' in arr:
        res['phi_hypothetical_local_Phi'] = {}
        for lab, idx in [('small_shifts_pm1_pm2.5', [2, 3, 4, 5]), ('shifts_to_pm5', [1, 2, 3, 4, 5, 6]), ('all_eight', list(range(8)))]:
            iv = phi_local_rows(arr['phi'], idx)
            res['phi_hypothetical_local_Phi'][lab] = dict(theta=iv, lerner=lerner(iv, arr['phi']), status='hypothetical: no theorem')
            print(f'phi      realized    HYPOTHETICAL local Phi {lab:22s} {lerner(iv, arr["phi"])}', flush=True)
    res['main_Q20E_bernstein_rms_skuday'] = {}
    for cname, a in arr.items():
        mid = int(np.argmin(np.abs(a['TH'] - .8)))
        for hs in [1.0, .25, 1 / 16, 1 / 64]:
            eps = q20e_bernstein_rms(a, hs)
            ivb, ivm = invert(a, eps, 'expectation', .05), invert_maxt(a, eps)
            res['main_Q20E_bernstein_rms_skuday'][f'{cname}|H_x{hs:g}'] = dict(bonferroni=lerner(ivb, a), maxt=lerner(ivm, a),
                theta_bonferroni=ivb, theta_maxt=ivm, allowance_at_theta08=float(eps[mid]))
            print(f'MAIN {cname:8s} H x {hs:<7g} Bonferroni {lerner(ivb, a)}  max-t {lerner(ivm, a)}  eps(0.8)={eps[mid]:.2f}', flush=True)
    if out_json:
        out_json.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
