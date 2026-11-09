"""Span law for the level-shift (A3) comparator on the SKU-day clock (9 November 2026), for 08_CLOSING_PROOFS.md.

The seller's level is chosen by the learner from an action set of multiplicative levels relative to the profile price;
the observed price is taken as the profile price at the learner's modal level. Two action sets:
  a3   the nine levels {0, +-1, +-2.5, +-5, +-10 percent} of the 8 November note (d = 9);
  lat  the log lattice exp(0.005 j) - 1, j = -20..20 (d = 41), which contains log shifts of 0.01, 0.025, 0.05, 0.10.
For every theta on the grid it saves the moments of the span law per action set: sum of squared cell spans, largest
cell span, largest and mean daily sum of cell spans (the block bound of the Bernstein step), the design effect DEFF of
day sums of the differences against the best action, and DEFF_A of the gains against the actual prices. For `lat`
it also saves the daily gain series and covariance blocks of the eight lattice shifts j = +-2, +-5, +-10, +-20.

Usage: python swappa_closing_phi_20261109.py [--spec raw_d0] [--seller GadgetPickup]
Output: output/swappa_core_20261108/closing_phi_arrays_<spec>_<seller>.npz
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
from scipy.special import expit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swappa_common_20261108 import (PILOT_OUT, SPECS, continuous, check_design_matches_saved,  # noqa: E402
                                    penalized_hessian_inverse, twoway_cluster_cov, add_freshness, date_influence)
from swappa_core_inference_20261108 import Seller, K8, TH  # noqa: E402
from swappa_closing_skuday_20261109 import OUT, blocks  # noqa: E402

LAT_J = np.arange(-20, 21)
MOMENT_J = [-20, -10, -5, -2, 2, 5, 10, 20]


def main():
    args = sys.argv[1:]
    spec = args[args.index('--spec') + 1] if '--spec' in args else 'raw_d0'
    seller = args[args.index('--seller') + 1] if '--seller' in args else 'GadgetPickup'
    t0 = time.time()
    s = add_freshness(pd.read_parquet(PILOT_OUT / 'raw_reference_panel.parquet').reset_index(drop=True))
    X, names, mean, sd, beta = check_design_matches_saved(s, spec)
    nd = len(names) + 1
    kw = {k: v for k, v in SPECS[spec].items() if k != 'product_week'}
    Hinv, q = penalized_hessian_inverse(X, beta, nd)
    V, _ = twoway_cluster_cov(X, s, beta, nd, Hinv, q)
    dates, PSI = date_influence(X, s, beta, nd, Hinv, q)
    m = Seller(s, X, beta, nd, SPECS[spec], sd, seller, dates)
    g, T = m.g, m.T
    PSI_s = PSI[m.days]
    cell_key = (g.sku.astype(str) + '|' + g.date_str).to_numpy()
    ccodes, cinv = np.unique(cell_key, return_inverse=True)
    nC = len(ccodes)
    C = sp.csr_matrix((np.ones(len(g)), (cinv, np.arange(len(g)))), shape=(nC, len(g)))
    row_day = np.searchsorted(m.days, m.dpos)
    cell_day = np.zeros(nC, int)
    cell_day[cinv] = row_day
    Dc = sp.csr_matrix((np.ones(nC), (np.arange(nC), cell_day)), shape=(nC, T))

    def cell_payoffs(shifts):
        R, CR = [], []
        for k in shifts:
            zz, _ = continuous(g, shift=float(k), **kw)
            qq = expit(m.z0 + ((zz - m.dense0) / sd) @ beta[1:nd])
            R.append(m.price * (1 + k) * qq); CR.append(m.ref * qq)
        return np.asarray(np.array(R) @ C.T), np.asarray(np.array(CR) @ C.T)

    sets = {'a3': np.array([0.0] + list(K8)), 'lat': np.exp(0.005 * LAT_J) - 1}
    payoffs = {k: cell_payoffs(v) for k, v in sets.items()}
    print(f'payoffs {time.time() - t0:.1f}s', flush=True)
    arrays = dict(TH=TH, lat_shifts=sets['lat'], a3_shifts=sets['a3'], moment_j=np.array(MOMENT_J))
    for name, (Rc, CRc) in payoffs.items():
        zero = 0 if name == 'a3' else int(np.flatnonzero(LAT_J == 0)[0])
        st = {k: np.zeros(len(TH)) for k in ['sqK', 'maxK', 'blockmax', 'blockmean', 'deff', 'deffA', 'astar']}
        for gi, th in enumerate(TH):
            P = Rc - th * CRc
            K = P.max(0) - P.min(0)
            blk = np.asarray(Dc.T @ K).ravel()
            astar = int(np.argmax(P.mean(1)))
            k2, Bt = float((K ** 2).mean()), nC / T
            Y = np.delete(P - P[astar], astar, 0)
            Ga = np.delete(P - P[zero], zero, 0)
            vY = np.asarray(Dc.T @ Y.T).T.var(1, ddof=1)
            vA = np.asarray(Dc.T @ Ga.T).T.var(1, ddof=1)
            for k, v in [('sqK', (K ** 2).sum()), ('maxK', K.max()), ('blockmax', blk.max()), ('blockmean', blk.mean()),
                         ('deff', vY.max() / (Bt * k2)), ('deffA', vA.max() / (Bt * k2)), ('astar', sets[name][astar])]:
                st[k][gi] = float(v)
        arrays.update({f'{name}__{k}': v for k, v in st.items()})
    # daily gain series and covariance blocks for the eight lattice shifts
    A, B, GA, GB = [], [], [], []
    base = m.ev(shift=0.0, key='base')
    for j in MOMENT_J:
        e = m.ev(shift=float(np.exp(0.005 * j) - 1))
        A.append(e['R'] - base['R']); B.append(e['CR'] - base['CR'])
        GA.append((e['gR'] - base['gR']) / T); GB.append((e['gCR'] - base['gCR']) / T)
    A, B, GA, GB = map(np.array, (A, B, GA, GB))
    arrays['lat__A'], arrays['lat__B'] = A, B
    arrays['lat__Daa'], arrays['lat__Dab'], arrays['lat__Dbb'] = GA @ V @ GA.T, GA @ V @ GB.T, GB @ V @ GB.T
    for lags, tag in [(7, ''), (0, '0')]:
        for k, v in blocks(A, B, GA, GB, V, PSI_s, lags).items():
            arrays[f'lat__{k}{tag}'] = v
    sold = g.sale.eq(1).to_numpy()
    arrays.update(mean_price_sold=float(m.price[sold].mean()), mean_ref_sold=float(m.ref[sold].mean()), T=T, nC=nC,
                  cells_per_day=np.bincount(cell_day, minlength=T), day_dates=dates[m.days].astype(str),
                  daily_profit=(base['R'] - TH[:, None] * base['CR']).mean(1))
    lon = json.loads((OUT / 'branch_longevity.json').read_text())[seller]
    arrays['H_cell'] = float(lon['implied_presample_listing_days']) * nC / len(g)
    np.savez_compressed(OUT / f'closing_phi_arrays_{spec}_{seller.replace(" ", "_")}.npz', **arrays)
    mid = int(np.argmin(np.abs(TH - .8)))
    print(json.dumps({k: round(float(v[mid]), 3) for k, v in arrays.items() if k.startswith(('a3__', 'lat__')) and np.ndim(v) == 1 and len(v) == len(TH)}), flush=True)
    print('CLOSING_PHI_COMPLETE', round(time.time() - t0, 1), flush=True)


if __name__ == '__main__':
    main()
