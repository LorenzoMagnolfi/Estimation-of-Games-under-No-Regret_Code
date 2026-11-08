"""Full-pipeline moving-block bootstrap for the Swappa level-deviation moments (8 November 2026).

Each replication resamples 7-day moving blocks of calendar dates (with replacement) to a
pseudo-sample of 164 day positions, keeps every listing-day of the sampled dates, refits the
logit demand on the pseudo-sample (same design as the saved 6 November fit of the chosen
specification, standardization recomputed), recomputes the daily level-deviation gain series
A_{k,d}, B_{k,d} for every seller with at least 300 sales in the original panel, and stores the
bootstrap moment means m*_k(theta) = mean_d(A - theta B) on the theta grid together with the
analytic standard errors (HAC(7)/T plus the two-way-clustered delta method) evaluated in the
pseudo-sample. The local analysis script compares bootstrap dispersion with the analytic
standard errors, forms percentile-t critical values for the max statistic, and reports the
rejection rate at the null point closest to binding. This is the "joint resampling scheme that
refits demand and recomputes moments" of the 6 November memo (item M3).

Usage: python swappa_block_bootstrap_20261108.py --spec raw_d0 --reps 10 --task 0 --out DIR
Writes DIR/<spec>_task<task>.npz with arrays per seller: m_star (reps x n x G), se_star (same),
plus the price coefficients of each replication and the sampled block starts.
"""
from __future__ import annotations

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
from swappa_common_20261108 import (PILOT_OUT, SPECS, MIN_SALES, continuous, design, fit,  # noqa: E402
                                    penalized_hessian_inverse, twoway_cluster_cov, bartlett_lrcov)

TH = np.round(np.arange(.2, 1.20001, .0025), 4)
K8 = [-.10, -.05, -.025, -.01, .01, .025, .05, .10]
BLOCK, LAGS = 7, 7


def resample_positions(dates, rng):
    T = len(dates)
    nb = int(np.ceil(T / BLOCK))
    starts = rng.integers(0, T - BLOCK + 1, nb)
    pos = np.concatenate([np.arange(s0, s0 + BLOCK) for s0 in starts])[:T]
    return pos, starts


def seller_series(g, XX, beta, nd, kw, sd, posidx, T):
    """Daily (position-indexed) gain series and window-mean gradients for the K8 shifts."""
    dense0, _ = continuous(g, **kw)
    z0 = XX @ beta
    price, ref = g.Price.to_numpy(dtype=float), g.ref.to_numpy()
    D = sp.csr_matrix((np.ones(len(g)), (posidx, np.arange(len(g)))), shape=(T, len(g)))
    XT = XX.T.tocsr()

    def ev(k):
        zz, _ = continuous(g, shift=k, **kw)
        delta = (zz - dense0) / sd
        q = expit(z0 + delta @ beta[1:nd])
        w = q * (1 - q)
        p1 = price * (1 + k)
        out = dict(R=np.asarray(D @ (p1 * q)).ravel(), CR=np.asarray(D @ (ref * q)).ravel())
        for key, v in [('gR', p1 * w), ('gCR', ref * w)]:
            gr = np.asarray(XT @ v).ravel()
            gr[1:nd] += delta.T @ v
            out[key] = gr
        return out
    base = ev(0.0)
    alts = [ev(k) for k in K8]
    A = np.array([a['R'] - base['R'] for a in alts])
    B = np.array([a['CR'] - base['CR'] for a in alts])
    GA = np.array([a['gR'] - base['gR'] for a in alts]) / T
    GB = np.array([a['gCR'] - base['gCR'] for a in alts]) / T
    return A, B, GA, GB


def moments_and_se(A, B, GA, GB, V, T):
    mA, mB = A.mean(1), B.mean(1)
    Haa, Hab, Hbb = bartlett_lrcov(A.T, A.T, LAGS), bartlett_lrcov(A.T, B.T, LAGS), bartlett_lrcov(B.T, B.T, LAGS)
    Daa, Dab, Dbb = GA @ V @ GA.T, GA @ V @ GB.T, GB @ V @ GB.T
    m = mA[:, None] - TH[None, :] * mB[:, None]
    var = ((np.diag(Haa)[:, None] - 2 * TH[None, :] * np.diag(Hab)[:, None] + TH[None, :] ** 2 * np.diag(Hbb)[:, None]) / T
           + np.diag(Daa)[:, None] - 2 * TH[None, :] * np.diag(Dab)[:, None] + TH[None, :] ** 2 * np.diag(Dbb)[:, None])
    return m, np.sqrt(np.maximum(var, 1e-300))


def main():
    args = sys.argv[1:]
    spec = args[args.index('--spec') + 1]
    reps = int(args[args.index('--reps') + 1])
    task = int(args[args.index('--task') + 1])
    outdir = Path(args[args.index('--out') + 1])
    outdir.mkdir(parents=True, exist_ok=True)
    kw = SPECS[spec]
    ckw = {k: v for k, v in kw.items() if k != 'product_week'}
    s = pd.read_parquet(PILOT_OUT / 'raw_reference_panel.parquet').reset_index(drop=True)
    dates = np.sort(s.date_str.unique())
    T = len(dates)
    date_code = pd.Categorical(s.date_str, categories=dates).codes
    rows_by_date = [np.flatnonzero(date_code == t) for t in range(T)]
    counts = s.groupby('Seller').sale.sum()
    sellers = counts[counts >= MIN_SALES].sort_values(ascending=False).index.tolist()
    rng = np.random.default_rng(20261108 + 1000 * task)
    store = {name: dict(m=[], se=[]) for name in sellers}
    coefs, starts_all, fitinfo = [], [], []
    t0 = time.time()
    for b in range(reps):
        pos, starts = resample_positions(dates, rng)
        idx = np.concatenate([rows_by_date[t] for t in pos])
        posidx = np.concatenate([np.full(len(rows_by_date[t]), j) for j, t in enumerate(pos)])
        sb = s.iloc[idx].reset_index(drop=True)
        X, names, mean, sd = design(sb, **kw)
        nd = len(names) + 1
        y = sb.sale.to_numpy()
        beta, info = fit(X, y, nd)
        Hinv, q = penalized_hessian_inverse(X, beta, nd)
        V, vinfo = twoway_cluster_cov(X, sb, beta, nd, Hinv, q)
        coefs.append(beta[1:nd] / sd)
        starts_all.append(starts)
        fitinfo.append(dict(info, **vinfo))
        for name in sellers:
            ix = sb.Seller.eq(name).to_numpy()
            if ix.sum() == 0:
                store[name]['m'].append(np.full((len(K8), len(TH)), np.nan))
                store[name]['se'].append(np.full((len(K8), len(TH)), np.nan))
                continue
            A, B, GA, GB = seller_series(sb.loc[ix].reset_index(drop=True), X[ix].tocsr(), beta, nd, ckw, sd, posidx[ix], T)
            m, se = moments_and_se(A, B, GA, GB, V, T)
            store[name]['m'].append(m)
            store[name]['se'].append(se)
        print(f'task {task} rep {b + 1}/{reps} done at {time.time() - t0:.0f}s; fit {info}; V clipped {vinfo}', flush=True)
    out = dict(theta=TH, shifts=np.array(K8), sellers=np.array(sellers), coef_names=np.array(names),
               coefs=np.array(coefs), block_starts=np.array(starts_all), fitinfo=np.array([str(f) for f in fitinfo]))
    for name in sellers:
        key = name.replace(' ', '_').replace('/', '_').replace(',', '').replace('.', '')
        out[f'm__{key}'] = np.array(store[name]['m'])
        out[f'se__{key}'] = np.array(store[name]['se'])
    np.savez_compressed(outdir / f'{spec}_task{task:02d}.npz', **out)
    print('BOOTSTRAP_TASK_COMPLETE', spec, task, reps, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
