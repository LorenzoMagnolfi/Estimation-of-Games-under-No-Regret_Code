"""Evaluate the closed SKU-day chain (paper repo JPE Revision/swappa-reconciliation-2026-11-09/08_CLOSING_PROOFS.md)
on the arrays written by `swappa_closing_skuday_20261109.py` (Unity). Runs in seconds; no demand re-evaluation.

Routes (all with SKU-day rounds, batched daily feedback, the 61-point markup grid as the action set, d = 61):
  test of record   Proposition 5: allowance = window average of the Lemma 2 high-probability per-cell bounds
                   (delta_H = 0.0125 spread over the window's days) plus the grid-rounding term; test with the lag-0
                   sample variance of daily gains + demand delta method + lag-0 demand cross term, alpha_s = 0.0375,
                   Bonferroni or least-favorable max-t over the tested markups;
  finite-sample    Lemma 4: Lemma 2 allowance + Freedman term (delta_M = 0.0125), demand projection at 0.025;
  expectation      Lemma 1 allowance with the same test (reported as a sensitivity: Proposition 4 does not cover it).
Spans: kappa = root-mean-square cell span over the grid at each theta, Kbar = window maximum (and 10 x), DEFF =
max(1, estimated design effect). Contexts: one context (`grid`) and three phone-generation contexts (`grid_gen`) with
H split by the context's share of cells and the error budget split equally across contexts.

Usage: python swappa_closing_eval_20261109.py [--arrays PATH] [--json OUT]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'output' / 'swappa_core_20261108' / 'closing_skuday_arrays_raw_d0_GadgetPickup.npz'
D = 61
DELTA_H, DELTA_M, ALPHA = .0125, .0125, .05


def span(mask, TH):
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)
    return [float(TH[idx[0]]), float(TH[idx[-1]]), bool((np.diff(idx) > 1).any())]


def lerner(iv, z):
    if iv is None:
        return None
    r = float(z['mean_ref_sold']) / float(z['mean_price_sold'])
    return [round(100 * (1 - iv[1] * r), 2), round(100 * (1 - iv[0] * r), 2)]


def window_days(z, share, H):
    """Per window day: cells B_t of the context and cells observed before the day, n_t (presample H plus window)."""
    B = z['cells_per_day'] * share
    n = H + np.concatenate([[0.0], np.cumsum(B)[:-1]])
    return B, n


def lemma2_daily(kappa, kbar, deff, B, n, delta, Bbar, d=D):
    """Window-average daily allowance from the Lemma 2 per-cell bounds b_t (Hedge, c = 1, kappa_L = kappa)."""
    T = len(B)
    lt = np.log((d - 1) * T / delta)
    b = kappa * np.sqrt(np.log(d)) * np.sqrt(n + 1) / n + kappa * np.sqrt(deff) * np.sqrt(2 * lt / n) + 2 * Bbar * kbar * lt / (3 * n)
    return float((B * np.minimum(b, kbar)).sum() / T)


def lemma1_daily(kappa, kbar, deff, B, n, Bbar, d=D):
    T = len(B)
    b = kappa * np.sqrt(np.log(d)) * np.sqrt(n + 1) / n + kappa * np.sqrt(deff) * np.sqrt(2 * np.log(d) / n) + 2 * Bbar * kbar * np.log(d) / (3 * n)
    return float((B * np.minimum(b, kbar)).sum() / T)


def freedman_daily(kappa, kbar, deffA, T, Btil, Bbar, delta, d=D):
    return float((np.sqrt(2 * T * Btil * kappa ** 2 * deffA * np.log(d / delta)) + 2 * Bbar * kbar * np.log(d / delta) / 3) / T)


def cov_at(z, c, th, lag0=True):
    t = '0' if lag0 else ''
    S_h = (z[f'{c}__Haa{t}'] - th * (z[f'{c}__Hab{t}'] + z[f'{c}__Hab{t}'].T) + th ** 2 * z[f'{c}__Hbb{t}']) / float(z['T'])
    S_d = z[f'{c}__Daa'] - th * (z[f'{c}__Dab'] + z[f'{c}__Dab'].T) + th ** 2 * z[f'{c}__Dbb']
    Cx = z[f'{c}__Caa{t}'] - th * (z[f'{c}__Cab{t}'] + z[f'{c}__Cba{t}']) + th ** 2 * z[f'{c}__Cbb{t}']
    return S_h, S_d, Cx + Cx.T


def means(z, c, days=None):
    A, B = z[f'{c}__A'], z[f'{c}__B']
    if days is not None:
        A, B = A[:, days], B[:, days]
    return A.mean(1), B.mean(1)


def invert(z, c, eps, route, alpha, maxt=False, lag0=True, draws=None):
    TH = z['TH']
    mA, mB = means(z, c)
    n = len(mA)
    ok = np.zeros(len(TH), bool)
    for gi, th in enumerate(TH):
        m = mA - th * mB
        S_h, S_d, Cx = cov_at(z, c, th, lag0)
        e = eps[:, gi] if eps is not None else 0.0
        if route == 'test':
            S = S_h + S_d + Cx
            sd = np.sqrt(np.maximum(np.diag(S), 1e-300))
            t = (m - e) / sd
            if maxt:
                w, U = np.linalg.eigh(S / np.outer(sd, sd))
                Zm = (draws[:, :n] @ (U * np.sqrt(np.maximum(w, 0))).T).max(1)
                ok[gi] = bool(t.max() <= np.quantile(Zm, 1 - alpha))
            else:
                ok[gi] = bool((t <= norm.ppf(1 - alpha / n)).all())
        else:                                                   # pathwise: demand projection only
            sd = np.sqrt(np.maximum(np.diag(S_d), 0))
            ok[gi] = bool((m - norm.ppf(1 - alpha / n) * sd <= e).all())
    return span(ok, TH)


def allowances(z, c, H_scale=1.0, kbar_scale=1.0, deff_floor=1.0):
    """Per-moment allowance arrays (moments x theta) for the three routes."""
    TH = z['TH']
    T, Btil = int(z['T']), float(z['Btil'])
    ctxs = ['all'] if c == 'grid' else [str(x) for x in np.unique(z['grid_gen__moment_context'])]
    mctx = np.array(['all'] * len(z['grid__A'])) if c == 'grid' else z['grid_gen__moment_context'].astype(str)
    out = {k: np.zeros((len(mctx), len(TH))) for k in ['test', 'finite', 'expect']}
    dH, dM = DELTA_H / len(ctxs), DELTA_M / len(ctxs)
    for k in ctxs:
        nk = z[f'span__{k}__nC']
        rows = mctx == k
        for gi in range(len(TH)):
            share = float(nk[gi]) / float(z['nC'])
            Bt, n = window_days(z, share, float(z['H_cell']) * share * H_scale)
            Bbar = float(Bt.max())
            kap = np.sqrt(z[f'span__{k}__sqK'][gi] / nk[gi])
            kb = z[f'span__{k}__maxK'][gi] * kbar_scale
            df = max(deff_floor, z[f'span__{k}__deff'][gi])
            dfA = max(deff_floor, z[f'span__{k}__deffA'][gi])
            rnd = z['rounding_daily'][gi] * share
            hp = lemma2_daily(kap, kb, df, Bt, n, dH, Bbar) + rnd
            out['test'][rows, gi] = hp
            out['finite'][rows, gi] = hp + freedman_daily(kap, kb, dfA, T, Btil * share, Bbar, dM)
            out['expect'][rows, gi] = lemma1_daily(kap, kb, df, Bt, n, Bbar) + rnd
    return out


PHI_DEFAULT = ROOT / 'output' / 'swappa_core_20261108' / 'closing_phi_arrays_raw_d0_GadgetPickup.npz'


def phi_allowances(zs, aset, d, H_scale=1.0, blk_scale=1.0, deff_override=None, maximal=True):
    """Daily allowances (per theta) for the level-shift comparator on the SKU-day clock, action set `aset`
    ('a3', d = 9, or 'lat', d = 41): Lemma 1 (expectation), Lemma 2 with one maximal event (test of record),
    and Lemma 3 (finite sample). Spans and design effects from closing_phi_arrays."""
    TH = zs['TH']
    T = int(zs['T']); nC = float(zs['nC']); Btil = nC / T
    B, n = window_days(zs, 1.0, float(zs['H_cell']) * H_scale)
    n_end = n[-1] + B[-1]
    xH, xM = np.log((d - 1) / DELTA_H), np.log(d / DELTA_M)
    out = {k: np.zeros(len(TH)) for k in ['expect', 'test', 'finite']}
    for gi in range(len(TH)):
        kap = np.sqrt(zs[f'{aset}__sqK'][gi] / nC)
        blk = zs[f'{aset}__blockmax'][gi] * blk_scale
        df = zs[f'{aset}__deff'][gi] if deff_override is None else deff_override
        dfA = zs[f'{aset}__deffA'][gi] if deff_override is None else deff_override
        reg = kap * np.sqrt(np.log(d)) * np.sqrt(n + 1) / n
        e1 = reg + kap * np.sqrt(df) * np.sqrt(2 * np.log(d) / n) + 2 * blk * np.log(d) / (3 * n)
        if maximal:
            e2 = reg + (np.sqrt(2 * n_end * kap ** 2 * df * xH) + 2 * blk * xH / 3) / n
        else:
            lt = np.log((d - 1) * T / DELTA_H)
            e2 = reg + kap * np.sqrt(df) * np.sqrt(2 * lt / n) + 2 * blk * lt / (3 * n)
        out['expect'][gi] = float((B * e1).sum() / T)
        out['test'][gi] = float((B * e2).sum() / T)
        out['finite'][gi] = out['test'][gi] + float((np.sqrt(2 * T * Btil * kap ** 2 * dfA * xM) + 2 * blk * xM / 3) / T)
    return out


def phi_main(res, rec, draws):
    zs = dict(np.load(PHI_DEFAULT, allow_pickle=False))
    zg = dict(np.load(DEFAULT, allow_pickle=False))
    TH = zs['TH']
    mid = int(np.argmin(np.abs(TH - .8)))
    prof = zs['daily_profit']
    for aset, d, zm, c in [('a3', 9, zg, 'phi'), ('lat', 41, zs, 'lat')]:
        n = len(zm[f'{c}__A'])
        al = phi_allowances(zs, aset, d)
        tile = lambda v: np.tile(v, (n, 1))  # noqa: E731
        print(f'--- level shifts, action set {aset} (d = {d}): allowance at theta 0.8, dollars a day (share of daily profit): '
              + ', '.join(f'{k} {v[mid]:.1f} ({v[mid] / prof[mid]:.3f})' for k, v in al.items()), flush=True)
        rec(f'{aset} | steady state, lag 0, Bonferroni', invert(zm, c, None, 'test', ALPHA))
        rec(f'{aset} | expectation (Lemma 1), lag 0, Bonferroni', invert(zm, c, tile(al['expect']), 'test', ALPHA))
        rec(f'{aset} | TEST OF RECORD (Lemma 2 + Prop 5), Bonferroni', invert(zm, c, tile(al['test']), 'test', ALPHA - DELTA_H))
        rec(f'{aset} | TEST OF RECORD (Lemma 2 + Prop 5), max-t', invert(zm, c, tile(al['test']), 'test', ALPHA - DELTA_H, maxt=True, draws=draws))
        rec(f'{aset} | finite sample (Lemma 4), demand projection', invert(zm, c, tile(al['finite']), 'path', ALPHA - DELTA_H - DELTA_M))
        for lab, kw in [('H x 1/4', dict(H_scale=.25)), ('H x 1/16', dict(H_scale=1 / 16)), ('Kblk x 2', dict(blk_scale=2.0)),
                        ('DEFF = 1 (cells independent within a day)', dict(deff_override=1.0)),
                        ('union over days instead of one maximal event', dict(maximal=False))]:
            a2 = phi_allowances(zs, aset, d, **kw)
            rec(f'{aset} | test of record, {lab}, Bonferroni', invert(zm, c, tile(a2['test']), 'test', ALPHA - DELTA_H))
        res[f'{aset} | allowance at theta 0.8'] = {k: float(v[mid]) for k, v in al.items()}
        res[f'{aset} | span at theta 0.8'] = {k.split('__')[1]: float(zs[k][mid]) for k in zs if k.startswith(aset + '__') and np.ndim(zs[k]) == 1 and len(zs[k]) == len(TH)}
    res['daily_profit_at_theta08'] = float(prof[mid])


def main():
    args = sys.argv[1:]
    path = Path(args[args.index('--arrays') + 1]) if '--arrays' in args else DEFAULT
    out_json = Path(args[args.index('--json') + 1]) if '--json' in args else None
    z = dict(np.load(path, allow_pickle=False))
    TH = z['TH']
    mid = int(np.argmin(np.abs(TH - .8)))
    draws = np.random.default_rng(20261109).standard_normal((20000, 200))
    res = {}

    def rec(name, iv):
        res[name] = dict(theta=iv, lerner=lerner(iv, z))
        print(f'{name:70s} {lerner(iv, z)}', flush=True)

    for c in ['phi', 'grid', 'grid_gen']:
        rec(f'{c} | steady state, HAC(7), Bonferroni', invert(z, c, None, 'test', ALPHA, lag0=False))
        rec(f'{c} | steady state, lag 0, Bonferroni', invert(z, c, None, 'test', ALPHA))
    for c in ['grid', 'grid_gen']:
        base = allowances(z, c)
        print(f'--- {c}: allowance at theta 0.8 (dollars a day): test {base["test"][0, mid]:.2f}, finite {base["finite"][0, mid]:.2f}, '
              f'expectation {base["expect"][0, mid]:.2f}', flush=True)
        rec(f'{c} | TEST OF RECORD (Prop 5), Bonferroni', invert(z, c, base['test'], 'test', ALPHA - DELTA_H))
        rec(f'{c} | TEST OF RECORD (Prop 5), max-t', invert(z, c, base['test'], 'test', ALPHA - DELTA_H, maxt=True, draws=draws))
        rec(f'{c} | finite-sample (Lemma 4), demand projection', invert(z, c, base['finite'], 'path', ALPHA - DELTA_H - DELTA_M))
        rec(f'{c} | expectation variant (Lemma 1), Bonferroni', invert(z, c, base['expect'], 'test', ALPHA))
        for hs in [.25, 1 / 16]:
            al = allowances(z, c, H_scale=hs)
            rec(f'{c} | test of record, H x {hs:g}, Bonferroni', invert(z, c, al['test'], 'test', ALPHA - DELTA_H))
        al = allowances(z, c, kbar_scale=10.0)
        rec(f'{c} | test of record, Kbar x 10, Bonferroni', invert(z, c, al['test'], 'test', ALPHA - DELTA_H))
        al = allowances(z, c, deff_floor=float(z['Btil']))
        rec(f'{c} | test of record, DEFF = cells per day (day clock bound), Bonferroni', invert(z, c, al['test'], 'test', ALPHA - DELTA_H))
        res[f'{c} | allowance at theta 0.8'] = {k: float(v[0, mid]) for k, v in base.items()}
    phi_main(res, rec, draws)
    # half samples, steady state fitted sets on the grid comparator
    dd = z['day_dates']
    for lab, days in [('first half', dd < '2023-10-01'), ('second half', dd >= '2023-10-01')]:
        mA, mB = means(z, 'grid', days)
        rec(f'grid | steady state fitted set, {lab}', span(np.array([bool((mA - th * mB <= 0).all()) for th in TH]), TH))
    res['span_at_theta08'] = {k.replace('span__', ''): float(z[k][mid]) for k in z if k.startswith('span__')}
    res['rounding_daily_at_theta08'] = float(z['rounding_daily'][mid])
    print(json.dumps(res['span_at_theta08']), flush=True)
    if out_json:
        out_json.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
