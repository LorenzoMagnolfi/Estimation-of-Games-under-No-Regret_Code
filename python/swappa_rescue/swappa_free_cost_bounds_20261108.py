"""Model-free cost bounds under the level-deviation moments (8 November 2026).

Reproduces the revival's attribution exercise (`affine_attribution.py`, part B) on the
native-reference panel and the 6 November fits: listing-level costs c_j are left free, bounded
only by c_j >= 0 (and, in the capped version, by the listing's lowest posted price), and the
level-deviation restrictions sum_j c_j dS_{k,j} >= dR_k for the eight shifts are imposed at
fitted demand. Linear programs give the range of (i) the sold-unit mean cost (the paper's
target) and (ii) the marginal-response-weighted mean cost, the functional the deviations load
on (Proposition 1 of the revival note). A sampling allowance is added as in the revival: the
Bonferroni critical value times the analytic standard error of each window-sum moment at the
proportional-model fitted midpoint (an approximation, disclosed).

Usage: python swappa_free_cost_bounds_20261108.py [--specs raw_d0,raw_d6]
Output: docs/evidence/swappa_free_cost_bounds_20261108.json
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
import scipy.sparse as sp  # noqa: E402
from scipy.optimize import linprog  # noqa: E402
from scipy.special import expit  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swappa_common_20261108 import (ROOT, PILOT_OUT, SPECS, MIN_SALES, continuous, check_design_matches_saved,  # noqa: E402
                                    penalized_hessian_inverse, twoway_cluster_cov, bartlett_lrcov, add_freshness)

EVID = ROOT / 'docs' / 'evidence' / 'swappa_free_cost_bounds_20261108.json'
K8 = [-.10, -.05, -.025, -.01, .01, .025, .05, .10]
TH = np.round(np.arange(.2, 1.20001, .0025), 4)


def main():
    args = sys.argv[1:]
    specs = args[args.index('--specs') + 1].split(',') if '--specs' in args else ['raw_d0', 'raw_d6']
    out = dict(date='2026-11-08', shifts=K8, results={})
    t0 = time.time()
    for tag in specs:
        kw = {k: v for k, v in SPECS[tag].items() if k != 'product_week'}
        s = add_freshness(pd.read_parquet(PILOT_OUT / 'raw_reference_panel.parquet').reset_index(drop=True))
        X, names, mean, sd, beta = check_design_matches_saved(s, tag)
        nd = len(names) + 1
        Hinv, q = penalized_hessian_inverse(X, beta, nd)
        V, _ = twoway_cluster_cov(X, s, beta, nd, Hinv, q)
        counts = s.groupby('Seller').sale.sum()
        sellers = counts[counts >= MIN_SALES].sort_values(ascending=False).index.tolist()
        out['results'][tag] = {}
        for name in sellers:
            ix = s.Seller.eq(name).to_numpy()
            g = s.loc[ix].reset_index(drop=True)
            XX = X[ix].tocsr()
            z0 = XX @ beta
            dense0, _ = continuous(g, **kw)
            price, ref = g.Price.to_numpy(dtype=float), g.ref.to_numpy()
            lcodes, linv = np.unique(g.Code.to_numpy(), return_inverse=True)
            nJ = len(lcodes)
            Lm = sp.csr_matrix((np.ones(len(g)), (linv, np.arange(len(g)))), shape=(nJ, len(g)))
            days, dinv = np.unique(g.date_str.to_numpy(), return_inverse=True)
            T = len(days)
            D = sp.csr_matrix((np.ones(len(g)), (dinv, np.arange(len(g)))), shape=(T, len(g)))

            def ev(k):
                zz, _ = continuous(g, shift=k, **kw)
                delta = (zz - dense0) / sd
                qq = expit(z0 + delta @ beta[1:nd])
                w = qq * (1 - qq)
                p1 = price * (1 + k)
                gr = np.asarray(XX.T @ (p1 * w)).ravel(); gr[1:nd] += delta.T @ (p1 * w)
                gc = np.asarray(XX.T @ (ref * w)).ravel(); gc[1:nd] += delta.T @ (ref * w)
                return dict(RJ=np.asarray(Lm @ (p1 * qq)).ravel(), SJ=np.asarray(Lm @ qq).ravel(),
                            R=np.asarray(D @ (p1 * qq)).ravel(), CR=np.asarray(D @ (ref * qq)).ravel(), gR=gr, gCR=gc)
            base = ev(0.0)
            alts = [ev(k) for k in K8]
            dS = np.array([a['SJ'] - base['SJ'] for a in alts])            # n x J
            dR = np.array([(a['RJ'] - base['RJ']).sum() for a in alts])    # n
            A_ub, b_ub = -dS, -dR
            sold_j = np.bincount(linv, weights=g.sale.to_numpy(), minlength=nJ)
            minp_j = np.full(nJ, np.inf); np.minimum.at(minp_j, linv, price)
            p_j = np.bincount(linv, weights=price, minlength=nJ) / np.bincount(linv, minlength=nJ)
            r_j = np.bincount(linv, weights=ref, minlength=nJ) / np.bincount(linv, minlength=nJ)
            w_sold = sold_j / sold_j.sum()
            w_marg = np.maximum(dS[K8.index(-.01)], 0); w_marg = w_marg / w_marg.sum()
            # proportional model at fitted demand and the T1 allowance at its midpoint
            a = np.array([a_['R'].sum() - base['R'].sum() for a_ in alts])
            bcoef = np.array([a_['CR'].sum() - base['CR'].sum() for a_ in alts])
            lo_th = max(a[i] / bcoef[i] for i in range(len(K8)) if bcoef[i] > 0)
            hi_th = min(a[i] / bcoef[i] for i in range(len(K8)) if bcoef[i] < 0)
            th_mid = .5 * (lo_th + hi_th)
            crit = norm.ppf(1 - .05 / len(K8))
            slack = []
            for i, a_ in enumerate(alts):
                gd = (a_['R'] - base['R']) - th_mid * (a_['CR'] - base['CR'])
                v = bartlett_lrcov(gd[:, None], gd[:, None])[0, 0] / T
                grad = ((a_['gR'] - base['gR']) - th_mid * (a_['gCR'] - base['gCR'])) / T
                v += grad @ V @ grad
                slack.append(crit * np.sqrt(v) * T)
            b_t1 = b_ub + np.array(slack)
            rec = dict(sales=int(sold_j.sum()), listings=int(nJ), sold_mean_price=float(w_sold @ p_j), marginal_weighted_price=float(w_marg @ p_j),
                       proportional_fit_theta=[float(lo_th), float(hi_th)], bounds={})
            for allow_name, bvec in [('fit', b_ub), ('T1approx', b_t1)]:
                for capname, ub in [('c>=0', None), ('0<=c<=min price', minp_j)]:
                    bounds = [(0, None if ub is None else float(ub[j])) for j in range(nJ)]
                    for wname, w in [('sold_mean', w_sold), ('marginal_weighted', w_marg)]:
                        lo = linprog(w, A_ub=A_ub, b_ub=bvec, bounds=bounds, method='highs')
                        hi = linprog(-w, A_ub=A_ub, b_ub=bvec, bounds=bounds, method='highs')
                        lo_v = float(lo.fun) if lo.status == 0 else None
                        hi_v = float(-hi.fun) if hi.status == 0 else ('unbounded' if hi.status == 3 else None)
                        pr = rec['sold_mean_price'] if wname == 'sold_mean' else rec['marginal_weighted_price']
                        lern = None if (lo_v is None or not isinstance(hi_v, float)) else [round(100 * (1 - hi_v / pr), 2), round(100 * (1 - lo_v / pr), 2)]
                        rec['bounds'][f'{allow_name}|{capname}|{wname}'] = dict(cost=[lo_v, hi_v], lerner=lern)
            out['results'][tag][name] = rec
            mw = rec['bounds']['T1approx|0<=c<=min price|marginal_weighted']['lerner']
            mu = rec['bounds']['T1approx|c>=0|marginal_weighted']['lerner']
            sm = rec['bounds']['fit|0<=c<=min price|sold_mean']['cost']
            print(f'[{tag}] {name[:22]:22s} marg-weighted Lerner T1approx capped {mw} uncapped {mu} | sold-mean cost fit capped {None if sm[0] is None else [round(x, 1) if isinstance(x, float) else x for x in sm]} | prop theta fit [{lo_th:.4f}, {hi_th:.4f}]', flush=True)
    out['seconds'] = round(time.time() - t0, 1)
    EVID.write_text(json.dumps(out, indent=1, default=float))
    print('FREE_COST_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
