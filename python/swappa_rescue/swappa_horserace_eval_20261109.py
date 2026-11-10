"""Real-data horse race (9 November 2026): the equilibrium benchmark against the steady state and the test of record,
for the 13 sellers with at least 300 sales, under the Decluttr reference (declared) and the internal reference.

Per seller and reference: the equilibrium first-order-condition estimate and interval (full window, each half, each
month); the steady-state and test-of-record sets (full window and halves); the regret frontier (the smallest share of
daily expected profit the seller must be forgoing for each candidate theta to be true); the monthly interval regret at
the equilibrium point. Runs in about a minute on the arrays of `swappa_horserace_tables_20261109.py`.

Usage: python swappa_horserace_eval_20261109.py [--tables DIR] [--json OUT]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swappa_horserace_core_20261109 import (TH, MID, DELTA_H, ALPHA, day_tables, sold_ratio, lerner, span_stats,  # noqa: E402
                                            allowance_record, moments, demand_terms, equilibrium, inequality_set,
                                            regret_frontier, connected)

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / 'output' / 'swappa_horserace_20261109'
LONGEVITY = ROOT / 'output' / 'swappa_core_20261108' / 'branch_longevity.json'
SPLIT = '2023-10-01'


def seller_name(path, lon):
    stem = path.stem.split('_', 3)[3]
    for k in lon:
        if k.replace(' ', '_') == stem:
            return k
    raise KeyError(stem)


def run_seller(z, ref, dem_all, H):
    Rd, CRd = day_tables(z, ref)
    T = Rd.shape[0]
    ratio = sold_ratio(z, ref)
    lev = np.full(T, MID)
    A, B, dR, dC, R0, C0 = moments(Rd, CRd, lev)
    dem = demand_terms(z, ref, dem_all['V'])
    sp_ = span_stats(z, ref)
    cpd = z['cells_per_day']
    eps = allowance_record(sp_, cpd, H)
    out = dict(ratio=ratio, H_cell=H, T=T, nC=int(z['nC']), sales=int(z['sales']))
    thN, ivN, se = equilibrium(dR, dC, dem=dem)
    out['equilibrium'] = dict(theta=thN, theta_iv=ivN, se_theta=se, lerner=lerner(thN, ratio), lerner_iv=lerner(ivN, ratio))
    thN0, ivN0, se0 = equilibrium(dR, dC, dem=None)
    out['equilibrium_no_demand'] = dict(theta_iv=ivN0, lerner_iv=lerner(ivN0, ratio))
    st = inequality_set(A, B, None, dem=dem)
    rc = inequality_set(A, B, eps, dem=dem, alpha=ALPHA - DELTA_H)
    out['steady'] = dict(theta=st, lerner=lerner(st, ratio))
    out['record'] = dict(theta=rc, lerner=lerner(rc, ratio))
    gi = int(np.argmin(np.abs(TH - thN)))
    prof = sp_['profit']
    out['profit_at_thN'] = float(prof[gi])
    out['allowance_at_thN'] = float(eps[gi])
    out['allowance_share_at_thN'] = float(eps[gi] / prof[gi])
    R, share = regret_frontier(A, B, prof)
    out['frontier'] = {}
    for cut in [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]:
        cc = connected(share <= max(cut, float(share[gi])), gi)
        out['frontier'][f'{cut:g}'] = lerner([TH[cc[0]], TH[cc[1]]], ratio) if cc else None
    out['frontier_curve'] = dict(lerner=[round(100 * (1 - t * ratio), 3) for t in TH], regret_share=share.round(5).tolist())
    # halves and months
    dates = z['day_dates'].astype(str)
    halves = {'first': np.flatnonzero(dates < SPLIT), 'second': np.flatnonzero(dates >= SPLIT)}
    out['halves'] = {}
    for hk, dd in halves.items():
        Ah, Bh = A[:, dd], B[:, dd]
        n_before = float(cpd[:dd[0]].sum())
        eh = allowance_record(sp_, cpd[dd], H, n_before=n_before)
        th_h, iv_h, _ = equilibrium(dR[dd], dC[dd], dem=dem, days=dd)
        st_h = inequality_set(Ah, Bh, None, dem=dem, days=dd)
        rc_h = inequality_set(Ah, Bh, eh, dem=dem, days=dd, alpha=ALPHA - DELTA_H)
        out['halves'][hk] = dict(days=len(dd), eq_theta=th_h, eq_iv=iv_h, eq_lerner=lerner(th_h, ratio), eq_lerner_iv=lerner(iv_h, ratio),
                                 steady=st_h, steady_lerner=lerner(st_h, ratio), record=rc_h, record_lerner=lerner(rc_h, ratio))
    months = np.array([d[:7] for d in dates])
    out['months'] = {}
    mr = 0.0
    for mth in sorted(set(months)):
        dd = np.flatnonzero(months == mth)
        th_m, iv_m, _ = equilibrium(dR[dd], dC[dd], dem=dem, days=dd)
        out['months'][mth] = dict(days=len(dd), eq_theta=th_m, eq_iv=iv_m, eq_lerner_iv=lerner(iv_m, ratio))
        mr += max(0.0, float((A[:, dd].sum(1) - thN * B[:, dd].sum(1)).max()))
    out['monthly_interval_regret_at_thN'] = mr / T
    out['monthly_interval_regret_share_at_thN'] = mr / T / float(prof[gi])

    def disjoint(a, b):
        return a is not None and b is not None and (a[1] < b[0] or b[1] < a[0])
    hv = out['halves']
    out['halves_disjoint'] = dict(equilibrium=disjoint(hv['first']['eq_iv'], hv['second']['eq_iv']),
                                  steady=disjoint(hv['first']['steady'], hv['second']['steady']) or hv['first']['steady'] is None or hv['second']['steady'] is None,
                                  record=disjoint(hv['first']['record'], hv['second']['record']) or hv['first']['record'] is None or hv['second']['record'] is None)
    ivs = [v['eq_iv'] for v in out['months'].values() if v['eq_iv'] is not None]
    out['months_eq_pairs_disjoint'] = int(sum(disjoint(a, b) for i, a in enumerate(ivs) for b in ivs[i + 1:]))
    out['months_eq_pairs'] = len(ivs) * (len(ivs) - 1) // 2
    out['months_eq_common_point'] = bool(max(v[0] for v in ivs) <= min(v[1] for v in ivs))
    return out


def main():
    args = sys.argv[1:]
    tdir = Path(args[args.index('--tables') + 1]) if '--tables' in args else TABLES
    out_json = Path(args[args.index('--json') + 1]) if '--json' in args else tdir / 'horserace_eval_raw_d0.json'
    lon = json.loads(LONGEVITY.read_text())
    dem_all = dict(np.load(tdir / 'demand_raw_d0.npz', allow_pickle=False))
    res = {}
    for p in sorted(tdir.glob('tables_raw_d0_*.npz')):
        name = seller_name(p, lon)
        z = dict(np.load(p, allow_pickle=False))
        H = float(lon[name]['implied_presample_listing_days']) * float(z['nC']) / float(z['n_rows'])
        res[name] = {}
        for ref in ['dec', 'int']:
            r = run_seller(z, ref, dem_all, H)
            res[name][ref] = r
            e = r['equilibrium']
            print(f"{name:26s} {ref} H={H:8.0f} eq {e['lerner']:6.2f} {e['lerner_iv']}  steady {r['steady']['lerner']}  record {r['record']['lerner']}"
                  f"  halves eq {r['halves']['first']['eq_lerner_iv']} {r['halves']['second']['eq_lerner_iv']} disj {r['halves_disjoint']}"
                  f"  months disjoint pairs {r['months_eq_pairs_disjoint']}/{r['months_eq_pairs']}  mreg {100 * r['monthly_interval_regret_share_at_thN']:.2f}%",
                  flush=True)
    out_json.write_text(json.dumps(res, indent=1, default=float))
    print('HORSERACE_EVAL_COMPLETE', out_json)


if __name__ == '__main__':
    main()
