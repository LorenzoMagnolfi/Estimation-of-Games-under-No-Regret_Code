"""Horse race against the equilibrium benchmark (9 November 2026): per-cell payoff tables on a log-price lattice.

For every seller with at least 300 terminal sales, under demand `raw_d0` (the 6 November fit, native-workbook
references), and for every SKU-day cell and every level j = -40..40 of the log lattice (step 0.005, so +-20 percent;
the seller's prices that day scaled by exp(0.005 j), owned-peer means rescaled, rivals fixed), the script saves

  R    expected revenue            sum over the cell's rows of p exp(0.005 j) q_j
  CR   cost-reference mass         sum of r q_j, r = the Decluttr quote (the declared A1 reference)
  CRI  internal-reference mass     sum of r_int q_j, r_int = internal_centered x the panel's Decluttr/internal ratio
  S    expected sales              sum of q_j

so that expected cell profit at cost ratio theta is R - theta CR (or R - theta CRI). It also saves, for the demand
delta method and the lag-0 cross term: the gradients of the window means of R, CR, CRI at the levels in GJ with respect
to the demand coefficients, the two-way clustered covariance V (shared), and the date influence rows of the seller's
days. Everything downstream (equilibrium first-order condition, steady-state inequalities, the test of record, the
laboratory with known truth) runs offline on these arrays: `swappa_horserace_eval_20261109.py`,
`swappa_horserace_lab_20261109.py`.

Checks: the saved standardization is reproduced (check_design_matches_saved); the level-0 table reproduces the
observed-price revenue sum to 1e-8; the j = +-2 day sums equal the shifted-level evaluation of Seller.ev.

Usage: python swappa_horserace_tables_20261109.py [--spec raw_d0] [--sellers all|GadgetPickup,...]
Output: output/swappa_horserace_20261109/tables_<spec>_<seller>.npz, demand_<spec>.npz, MANIFEST lines on stdout.
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
from swappa_common_20261108 import (ROOT, PILOT_OUT, SPECS, MIN_SALES, continuous, check_design_matches_saved,  # noqa: E402
                                    penalized_hessian_inverse, twoway_cluster_cov, add_freshness, date_influence)
from swappa_core_inference_20261108 import Seller  # noqa: E402

OUT = ROOT / 'output' / 'swappa_horserace_20261109'
LAT = np.arange(-40, 41)                      # log lattice index, step 0.005
GJ = [-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20]


def main():
    args = sys.argv[1:]
    spec = args[args.index('--spec') + 1] if '--spec' in args else 'raw_d0'
    sel = args[args.index('--sellers') + 1] if '--sellers' in args else 'all'
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s = add_freshness(pd.read_parquet(PILOT_OUT / 'raw_reference_panel.parquet').reset_index(drop=True))
    X, names, mean, sd, beta = check_design_matches_saved(s, spec)
    nd = len(names) + 1
    kw = {k: v for k, v in SPECS[spec].items() if k != 'product_week'}
    Hinv, q = penalized_hessian_inverse(X, beta, nd)
    V, vinfo = twoway_cluster_cov(X, s, beta, nd, Hinv, q)
    dates, PSI = date_influence(X, s, beta, nd, Hinv, q)
    both = s.Ref_Price.notna() & s.internal_centered.notna() & (s.internal_centered > 0)
    ratio_int = float(np.median(s.loc[both, 'Ref_Price'] / s.loc[both, 'internal_centered']))
    np.savez_compressed(OUT / f'demand_{spec}.npz', V=V, beta=beta, names=np.array(names), nd=nd,
                        dates=np.asarray(dates).astype(str), ratio_int=ratio_int)
    print(f'setup {time.time() - t0:.1f}s; V clipped {vinfo}; Decluttr/internal ratio {ratio_int:.4f}', flush=True)
    counts = s.groupby('Seller').sale.sum()
    sellers = counts[counts >= MIN_SALES].sort_values(ascending=False).index.tolist()
    if sel != 'all':
        sellers = [x for x in sellers if x in sel.split(',')]
    for name in sellers:
        t1 = time.time()
        m = Seller(s, X, beta, nd, SPECS[spec], sd, name, dates)
        g, T = m.g, m.T
        rint = g.internal_centered.to_numpy(dtype=float) * ratio_int
        rint = np.where(np.isfinite(rint) & (rint > 0), rint, m.ref)          # no internal index: Decluttr quote
        cell_key = (g.sku.astype(str) + '|' + g.date_str).to_numpy()
        ccodes, cinv = np.unique(cell_key, return_inverse=True)
        nC = len(ccodes)
        C = sp.csr_matrix((np.ones(len(g)), (cinv, np.arange(len(g)))), shape=(nC, len(g)))
        row_day = np.searchsorted(m.days, m.dpos)
        cell_day = np.zeros(nC, int)
        cell_day[cinv] = row_day
        R, CR, CRI, S = (np.zeros((nC, len(LAT))) for _ in range(4))
        grads = {k: np.zeros((len(GJ), len(beta))) for k in ['R', 'CR', 'CRI']}
        for li, j in enumerate(LAT):
            shift = float(np.exp(0.005 * j) - 1)
            zz, _ = continuous(g, shift=shift, **kw)
            delta = (zz - m.dense0) / sd
            qq = expit(m.z0 + delta @ beta[1:nd])
            p1 = m.price * (1 + shift)
            R[:, li] = C @ (p1 * qq)
            CR[:, li] = C @ (m.ref * qq)
            CRI[:, li] = C @ (rint * qq)
            S[:, li] = C @ qq
            if j in GJ:
                w = qq * (1 - qq)
                gi = GJ.index(j)
                for key, v in [('R', p1 * w), ('CR', m.ref * w), ('CRI', rint * w)]:
                    gr = np.asarray(m.XX.T @ v).ravel()
                    gr[1:nd] += delta.T @ v
                    grads[key][gi] = gr / T                                   # gradient of the window mean
        # checks
        base = m.ev(shift=0.0, key='base')
        Dc = sp.csr_matrix((np.ones(nC), (np.arange(nC), cell_day)), shape=(nC, T))
        day0 = np.asarray(Dc.T @ R[:, LAT == 0]).ravel()
        assert np.allclose(day0, base['R'], rtol=1e-8, atol=1e-8), 'level-0 revenue mismatch'
        e2 = m.ev(shift=float(np.exp(0.01) - 1))
        assert np.allclose(np.asarray(Dc.T @ R[:, LAT == 2]).ravel(), e2['R'], rtol=1e-8, atol=1e-8)
        assert np.allclose(np.asarray(Dc.T @ CR[:, LAT == 2]).ravel(), e2['CR'], rtol=1e-8, atol=1e-8)
        assert np.allclose(grads['R'][GJ.index(2)], (e2['gR']) / T, rtol=1e-7, atol=1e-9)
        sold = g.sale.eq(1).to_numpy()
        np.savez_compressed(
            OUT / f'tables_{spec}_{name.replace(" ", "_")}.npz',
            lat=LAT, gj=np.array(GJ), R=R, CR=CR, CRI=CRI, S=S, cell_day=cell_day, cells_per_day=np.bincount(cell_day, minlength=T),
            day_dates=np.asarray(dates)[m.days].astype(str), gR=grads['R'], gCR=grads['CR'], gCRI=grads['CRI'],
            PSI_s=PSI[m.days], n_rows=len(g), nC=nC, T=T, sales=int(sold.sum()),
            sum_p_sold=float(m.price[sold].sum()), sum_r_sold=float(m.ref[sold].sum()), sum_ri_sold=float(rint[sold].sum()),
            share_rint_imputed=float(np.mean(~(np.isfinite(g.internal_centered.to_numpy(dtype=float)) & (g.internal_centered.to_numpy(dtype=float) > 0)))))
        print(f'{name}: rows {len(g)}, cells {nC}, days {T}, sales {int(sold.sum())}, {time.time() - t1:.1f}s', flush=True)
    print('HORSERACE_TABLES_COMPLETE', round(time.time() - t0, 1), flush=True)


if __name__ == '__main__':
    main()
