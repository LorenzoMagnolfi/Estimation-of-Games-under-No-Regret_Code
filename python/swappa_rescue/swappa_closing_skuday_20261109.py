"""Arrays for the closed SKU-day chain of 9 November 2026 (paper repo JPE Revision/swappa-reconciliation-2026-11-09/
08_CLOSING_PROOFS.md): contextual external regret on the cell's markup over the internal market index, on a grid
fixed before seeing the data, SKU-day rounds, stochastic environment.

For one seller and one demand fit, evaluates expected cell profits at every grid markup and at the actual prices,
and saves, for every candidate theta on the grid [0.2, 1.2] by 0.0025:
  - daily gain series A (revenue part) and B (value-weighted sales part) for every tested moment, with HAC(7) and
    lag-0 variance blocks, demand delta-method blocks, and demand-moment cross blocks (lag 7 and lag 0);
  - the moments of the span law used by Lemmas 1 to 3: sum of squared cell spans over the grid, the largest span,
    the cell count, the design effects DEFF (grid differences against the best grid level) and DEFF_A (grid gains
    against the actual prices), per context;
  - the grid-rounding term: daily mean of the largest payoff change from moving the actual prices by +-0.5 percent.

Comparators: `grid` (one context, 61 markups from -0.30 to +0.30 log points by 0.01), `grid_gen` (three phone-generation
contexts, the same grid in each), and `phi` (the eight level shifts, kept only as the steady-state benchmark).
Cell payoffs are separable across SKU-day cells in the demand model (owned peers live in the same SKU-day), so a
context's moments are sums of the one-context cell payoffs over the context's cells.

Usage: python swappa_closing_skuday_20261109.py [--spec raw_d0] [--seller GadgetPickup]
Output: output/swappa_core_20261108/closing_skuday_arrays_<spec>_<seller>.npz and a JSON summary in docs/evidence/.
"""
from __future__ import annotations

import hashlib
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
from swappa_common_20261108 import (ROOT, PILOT_OUT, SPECS, continuous, check_design_matches_saved,  # noqa: E402
                                    penalized_hessian_inverse, twoway_cluster_cov, add_freshness, bartlett_lrcov,
                                    date_influence, coef_path)
from swappa_core_inference_20261108 import Seller, K8, TH  # noqa: E402

OUT = ROOT / 'output' / 'swappa_core_20261108'
GRID = np.round(np.arange(-0.30, 0.30001, 0.01), 2)


def gen_group(dev):
    g = pd.Series(dev).str.replace('apple-iphone-', '', regex=False).str.extract(r'^(\d+)')[0].astype(float).to_numpy()
    return np.where(g >= 13, 'g13_15', np.where(g >= 11, 'g11_12', 'older_se'))


def blocks(A, B, GA, GB, V, PSI_s, lags):
    """Variance blocks for n daily series at a given Bartlett lag (0 = plain sample covariance)."""
    out = dict(Haa=bartlett_lrcov(A.T, A.T, lags), Hab=bartlett_lrcov(A.T, B.T, lags), Hbb=bartlett_lrcov(B.T, B.T, lags))
    UA, UB = PSI_s @ GA.T, PSI_s @ GB.T
    out.update(Caa=bartlett_lrcov(A.T, UA, lags), Cab=bartlett_lrcov(A.T, UB, lags),
               Cba=bartlett_lrcov(B.T, UA, lags), Cbb=bartlett_lrcov(B.T, UB, lags))
    return out


def deff(Y, Dmat, kappa2, Btil):
    """Design effect of day-block sums of the rows of Y (levels x cells): Var(day sum) / (Btil kappa^2), max over rows.
    Dmat: sparse cells x days indicator for the cells of Y."""
    S = np.asarray((Dmat.T @ Y.T)).T
    v = S.var(1, ddof=1)
    return float(v.max() / (Btil * kappa2)), float(np.median(v) / (Btil * kappa2))


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
    V, vinfo = twoway_cluster_cov(X, s, beta, nd, Hinv, q)
    dates, PSI = date_influence(X, s, beta, nd, Hinv, q)
    m = Seller(s, X, beta, nd, SPECS[spec], sd, seller, dates)
    g, T = m.g, m.T
    PSI_s = PSI[m.days]
    sold = g.sale.eq(1).to_numpy()
    mp, mr = float(m.price[sold].mean()), float(m.ref[sold].mean())
    print(f'setup {time.time() - t0:.1f}s', flush=True)

    # cells, day of each cell, markup level of each cell over the internal index
    cell_key = (g.sku.astype(str) + '|' + g.date_str).to_numpy()
    ccodes, cinv = np.unique(cell_key, return_inverse=True)
    nC = len(ccodes)
    C = sp.csr_matrix((np.ones(len(g)), (cinv, np.arange(len(g)))), shape=(nC, len(g)))
    row_day = np.searchsorted(m.days, m.dpos)
    cell_day = np.zeros(nC, int)
    cell_day[cinv] = row_day
    cells_per_day = np.bincount(cell_day, minlength=T)
    Btil, Bbar = nC / T, int(cells_per_day.max())
    den = g.internal_centered.to_numpy(dtype=float)
    valid = np.isfinite(den) & (den > 0)
    lr = np.where(valid, np.log(m.price / np.where(valid, den, 1.0)), 0.0)
    cnt = np.asarray(C @ valid.astype(float)).ravel()
    overall = float(lr[valid].mean())
    cell_level = np.where(cnt > 0, np.asarray(C @ lr).ravel() / np.maximum(cnt, 1), overall)
    row_cell_level = cell_level[cinv]
    gen_row = gen_group(g.device_id.to_numpy())
    gen_cell = np.empty(nC, dtype=object)
    gen_cell[cinv] = gen_row
    contexts = [c for c in ['older_se', 'g11_12', 'g13_15'] if (gen_row == c).sum() >= 300]

    lon = json.loads((OUT / 'branch_longevity.json').read_text())[seller]
    H_listing = float(lon['implied_presample_listing_days'])
    H_cell = H_listing * nC / len(g)
    days_window = (pd.Timestamp(lon['ratings_last'][0]) - pd.Timestamp(lon['ratings_first'][0])).days
    H_day = float(lon['implied_presample_sales']) / (lon['terminal_sales_between'] / days_window)

    def rows_at(shift):
        zz, _ = continuous(g, shift=shift, **kw)
        dl = (zz - m.dense0) / sd
        qq = expit(m.z0 + dl @ beta[1:nd])
        return dl, qq

    def grad(mask, p1, qq, dl):
        w = qq * (1 - qq)
        out = []
        for v in (p1 * w, m.ref * w):
            vm = np.where(mask, v, 0.0)
            gr = np.asarray(m.XX.T @ vm).ravel()
            gr[1:nd] += dl.T @ vm
            out.append(gr)
        return out

    # actual prices (level index 0 of the stacked arrays) and the grid
    dl0, q0 = rows_at(0.0)
    shifts = [np.exp(L - row_cell_level) - 1 for L in GRID]
    Rrow, CRrow, DL, QQ = [m.price * q0], [m.ref * q0], [dl0], [q0]
    for sh in shifts:
        dl, qq = rows_at(sh)
        Rrow.append(m.price * (1 + sh) * qq); CRrow.append(m.ref * qq); DL.append(dl); QQ.append(qq)
    Rrow, CRrow = np.array(Rrow), np.array(CRrow)
    Rc, CRc = np.asarray(Rrow @ C.T), np.asarray(CRrow @ C.T)          # (1 + 61) x cells
    Rd, CRd = np.asarray(Rrow @ m.D.T), np.asarray(CRrow @ m.D.T)      # (1 + 61) x days
    # rounding term: actual prices moved by +-0.5 percent
    rnd = []
    for k in (np.exp(.005) - 1, np.exp(-.005) - 1):
        dl, qq = rows_at(np.full(len(g), k))
        rnd.append((np.asarray((m.price * (1 + k) * qq) @ C.T), np.asarray((m.ref * qq) @ C.T)))
    print(f'grid evaluated {time.time() - t0:.1f}s', flush=True)

    out = dict(date='2026-11-09', spec=spec, seller=seller, grid=GRID.tolist(), T=int(T), nC=int(nC), Btil=Btil, Bbar=Bbar,
               H_cell=H_cell, H_day=H_day, contexts=contexts, overall_level=overall,
               cell_level_quantiles={str(qv): float(np.quantile(cell_level, qv)) for qv in [.01, .1, .25, .5, .75, .9, .99]},
               mean_price_sold=mp, mean_ref_sold=mr, covariance=vinfo)
    arrays = dict(TH=TH, GRID=GRID, day_dates=dates[m.days].astype(str), cells_per_day=cells_per_day,
                  cell_level=cell_level, mean_price_sold=mp, mean_ref_sold=mr, T=T, nC=nC, Btil=Btil, Bbar=Bbar,
                  H_cell=H_cell, H_day=H_day)

    # moments: one context (grid) and generation contexts (grid_gen); phi benchmark
    specs = {'grid': [(None, i) for i in range(len(GRID))],
             'grid_gen': [(c, i) for c in contexts for i in range(len(GRID))]}
    for cname, items in specs.items():
        A, B, GA, GB = [], [], [], []
        cache = {}
        for ctx, i in items:
            mask = np.ones(len(g), bool) if ctx is None else (gen_row == ctx)
            A.append(np.asarray(m.D @ np.where(mask, Rrow[i + 1] - Rrow[0], 0.0)).ravel())
            B.append(np.asarray(m.D @ np.where(mask, CRrow[i + 1] - CRrow[0], 0.0)).ravel())
            key0 = ('base', ctx)
            if key0 not in cache:
                cache[key0] = grad(mask, m.price, QQ[0], DL[0])
            gA1, gB1 = grad(mask, m.price * (1 + shifts[i]), QQ[i + 1], DL[i + 1])
            GA.append((gA1 - cache[key0][0]) / T); GB.append((gB1 - cache[key0][1]) / T)
        A, B, GA, GB = map(np.array, (A, B, GA, GB))
        arrays[f'{cname}__A'], arrays[f'{cname}__B'] = A, B
        arrays[f'{cname}__Daa'], arrays[f'{cname}__Dab'], arrays[f'{cname}__Dbb'] = GA @ V @ GA.T, GA @ V @ GB.T, GB @ V @ GB.T
        for lags, tag in [(7, ''), (0, '0')]:
            for k, v in blocks(A, B, GA, GB, V, PSI_s, lags).items():
                arrays[f'{cname}__{k}{tag}'] = v
        arrays[f'{cname}__moment_context'] = np.array(['all' if c is None else c for c, _ in items])
        arrays[f'{cname}__moment_level'] = np.array([GRID[i] for _, i in items])
        print(f'{cname}: {len(items)} moments {time.time() - t0:.1f}s', flush=True)

    # phi benchmark (steady state only)
    A, B, GA, GB = [], [], [], []
    base = m.ev(shift=0.0, key='base')
    for k in K8:
        e = m.ev(shift=k, key=('phi', k))
        A.append(e['R'] - base['R']); B.append(e['CR'] - base['CR'])
        GA.append((e['gR'] - base['gR']) / T); GB.append((e['gCR'] - base['gCR']) / T)
    A, B, GA, GB = map(np.array, (A, B, GA, GB))
    arrays['phi__A'], arrays['phi__B'] = A, B
    arrays['phi__Daa'], arrays['phi__Dab'], arrays['phi__Dbb'] = GA @ V @ GA.T, GA @ V @ GB.T, GB @ V @ GB.T
    for lags, tag in [(7, ''), (0, '0')]:
        for k, v in blocks(A, B, GA, GB, V, PSI_s, lags).items():
            arrays[f'phi__{k}{tag}'] = v

    # span law, design effects and rounding term at every theta, per context
    ctx_list = ['all'] + contexts
    dmats = {}
    for c in ctx_list:
        cm = np.ones(nC, bool) if c == 'all' else (gen_cell == c)
        dmats[c] = sp.csr_matrix((np.ones(cm.sum()), (np.arange(cm.sum()), cell_day[cm])), shape=(cm.sum(), T))
    stats = {f'{c}__{k}': np.zeros(len(TH)) for c in ctx_list for k in
             ['sqK', 'maxK', 'nC', 'deff', 'deff_med', 'deffA', 'deffA_med', 'astar_level']}
    rnd_day = np.zeros(len(TH))
    for gi, th in enumerate(TH):
        P = Rc - th * CRc                      # row 0 actual, rows 1.. grid
        Pg = P[1:]
        K = Pg.max(0) - Pg.min(0)              # span over the learner's action set (grid)
        Pu, Pdn = rnd[0][0].ravel() - th * rnd[0][1].ravel(), rnd[1][0].ravel() - th * rnd[1][1].ravel()
        rnd_day[gi] = float(np.maximum(np.abs(Pu - P[0]), np.abs(Pdn - P[0])).sum() / T)
        for c in ctx_list:
            cm = np.ones(nC, bool) if c == 'all' else (gen_cell == c)
            Kc = K[cm]
            k2 = float((Kc ** 2).mean())
            Pgc = Pg[:, cm]
            astar = int(np.argmax(Pgc.mean(1)))
            Y = Pgc - Pgc[astar]
            Ga = Pgc - P[0, cm]
            bt = cm.sum() / T
            dmax, dmed = deff(np.delete(Y, astar, 0), dmats[c], k2, bt)
            amax, amed = deff(Ga, dmats[c], k2, bt)
            for k, v in [('sqK', float((Kc ** 2).sum())), ('maxK', float(Kc.max())), ('nC', float(cm.sum())),
                         ('deff', dmax), ('deff_med', dmed), ('deffA', amax), ('deffA_med', amed),
                         ('astar_level', float(GRID[astar]))]:
                stats[f'{c}__{k}'][gi] = v
    arrays.update({f'span__{k}': v for k, v in stats.items()})
    arrays['rounding_daily'] = rnd_day
    mid = int(np.argmin(np.abs(TH - .8)))
    out['at_theta08'] = {k: float(v[mid]) for k, v in stats.items()}
    out['at_theta08']['rounding_daily'] = float(rnd_day[mid])
    out['daily_expected_profit_at_theta08'] = float((base['R'] - .8 * base['CR']).mean())
    np.savez_compressed(OUT / f'closing_skuday_arrays_{spec}_{seller.replace(" ", "_")}.npz', **arrays)
    out['hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     [Path(__file__), Path(__file__).with_name('swappa_common_20261108.py'),
                      Path(__file__).with_name('swappa_core_inference_20261108.py'),
                      PILOT_OUT / 'raw_reference_panel.parquet', coef_path(spec), OUT / 'branch_longevity.json']}
    out['seconds'] = round(time.time() - t0, 1)
    ev = ROOT / 'docs' / 'evidence' / 'swappa_closing_skuday_20261109.json'
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out['at_theta08'], default=float), flush=True)
    print('CLOSING_SKUDAY_COMPLETE', out['seconds'], flush=True)


if __name__ == '__main__':
    main()
