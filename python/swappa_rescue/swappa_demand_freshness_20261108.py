"""Demand with repricing-freshness controls (8 November 2026).

Motivation: the batch-markdown event study (swappa_markdown_events_20261108.py) finds that a
seller's stock hazard in the repriced SKUs roughly doubles on and after a repricing day whatever
the size of the cut, so price changes carry a non-price effect (visibility, notifications, or
the seller's other same-day actions). If repricing days are also days of lower prices, the
logit attributes part of that effect to the price and overstates the own-price response. This
script adds to the richer-rivals specification (raw_d1) covariates that are held fixed under a
counterfactual level deviation: changed today, cut today, an observation gap before today, the
log of days since the last observed price change, and an indicator for two days or fewer since
the change. It fits the full sample and the pilot's listing-disjoint holdout, saves the fit in
the 6 November coefficient format (so the core inference script can use it as `raw_d8_fresh`),
and reports the price coefficients, holdout log loss, and the focal sellers' responses to a
uniform 5 percent cut next to the same quantities for raw_d1.

Usage: python swappa_demand_freshness_20261108.py
Outputs: output/swappa_core_20261108/raw_d8_fresh_coef.npz, raw_d8_fresh.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swappa_common_20261108 import (PILOT_OUT, CORE_OUT, SPECS, FOCAL, continuous, design, fit,  # noqa: E402
                                    add_freshness, load_saved_fit)


def logloss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def effects(s, X, b, names, sd, kw):
    q = expit(X @ b)
    base, _ = continuous(s, **kw)
    out = {}
    for name in FOCAL:
        ix = s.Seller.eq(name).to_numpy()
        alt, _ = continuous(s, shift=-.05, **kw)
        q1 = expit(X @ b + ((alt - base) / sd) @ b[1:len(names) + 1])
        out[name] = dict(predicted_base_sales=float(q[ix].sum()), change_pct_5pct_cut=float(100 * (q1[ix].sum() / q[ix].sum() - 1)))
    return out


def main():
    t0 = time.time()
    CORE_OUT.mkdir(parents=True, exist_ok=True)
    s = add_freshness(pd.read_parquet(PILOT_OUT / 'raw_reference_panel.parquet').reset_index(drop=True))
    y = s.sale.to_numpy()
    desc = dict(share_changed_today=float(s.chg_today.mean()), share_cut_today=float(s.cut_today.mean()),
                share_gap_before=float(s.gap_before.mean()), dsc_quantiles=np.percentile(s.dsc, [10, 25, 50, 75, 90]).tolist(),
                sale_rate_by_changed_today=s.groupby('chg_today').sale.mean().to_dict(),
                sale_rate_by_dsc_bin=s.assign(b=pd.cut(s.dsc, [-1, 0, 1, 2, 4, 7, 14, 1000])).groupby('b', observed=True).sale.mean().round(5).astype(float).to_dict())
    desc['sale_rate_by_dsc_bin'] = {str(k): v for k, v in desc['sale_rate_by_dsc_bin'].items()}
    print('descriptives', json.dumps(desc, default=str), flush=True)
    kw = SPECS['raw_d8_fresh']
    X, names, mean, sd = design(s, **kw)
    nd = len(names) + 1
    beta, conv = fit(X, y, nd)
    hold = s.holdout.to_numpy()
    bh, convh = fit(X[~hold], y[~hold], nd)
    pred = expit(X[hold] @ bh)
    rec = dict(tag='raw_d8_fresh', design=kw, rows=int(len(s)), parameters=int(X.shape[1]), convergence=conv, holdout_convergence=convh,
               coefficients={n: float(beta[1 + i] / sd[i]) for i, n in enumerate(names)},
               holdout_logloss=logloss(y[hold], pred), holdout_actual_sales=float(y[hold].sum()), holdout_predicted_sales=float(pred.sum()),
               effects=effects(s, X, beta, names, sd, kw), descriptives=desc, seconds=round(time.time() - t0, 1))
    hold_df = s.loc[hold, ['Seller', 'sale']].assign(pred=pred)
    rec['focal_holdout'] = hold_df[hold_df.Seller.isin(FOCAL)].groupby('Seller').agg(actual=('sale', 'sum'), predicted=('pred', 'sum')).to_dict('index')
    # comparison with raw_d1 (saved fit): coefficients and the same holdout
    d1 = load_saved_fit('raw_d1')
    X1, names1, mean1, sd1 = design(s, mean=d1['mean'], sd=d1['sd'], **SPECS['raw_d1'])
    assert names1 == d1['names']
    pred1_full = expit(X1 @ d1['beta'])
    b1h, _ = fit(X1[~hold], y[~hold], len(names1) + 1)
    pred1 = expit(X1[hold] @ b1h)
    rec['raw_d1_comparison'] = dict(coefficients={n: float(d1['beta'][1 + i] / d1['sd'][i]) for i, n in enumerate(names1[:4])},
                                    holdout_logloss=logloss(y[hold], pred1), effects=effects(s, X1, d1['beta'], names1, d1['sd'], SPECS['raw_d1']),
                                    insample_logloss_d1=logloss(y, pred1_full), insample_logloss_fresh=logloss(y, expit(X @ beta)))
    np.savez_compressed(CORE_OUT / 'raw_d8_fresh_coef.npz', beta=beta, names=np.asarray(names), mean=mean, sd=sd)
    (CORE_OUT / 'raw_d8_fresh.json').write_text(json.dumps(rec, indent=1, default=str))
    print('PRICE fresh', {n: round(rec['coefficients'][n], 3) for n in names[:4]}, flush=True)
    print('PRICE d1   ', {n: round(v, 3) for n, v in rec['raw_d1_comparison']['coefficients'].items()}, flush=True)
    print('FRESH coefs', {n: round(rec['coefficients'][n], 3) for n in names if n in ('chg_today', 'cut_today', 'gap_before', 'log1p_dsc', 'dsc_le2')}, flush=True)
    print('holdout logloss fresh', round(rec['holdout_logloss'], 6), 'd1', round(rec['raw_d1_comparison']['holdout_logloss'], 6), flush=True)
    print('5pct cut response fresh', rec['effects'], flush=True)
    print('5pct cut response d1   ', rec['raw_d1_comparison']['effects'], flush=True)
    print('FRESHNESS_COMPLETE', rec['seconds'], flush=True)


if __name__ == '__main__':
    main()
