"""Shared demand design and fitting code for the 8 November 2026 Swappa inference work.

The functions `continuous`, `design`, and `fit` are copied from
`demand_repair_pilot_20261106.py` (same repository, 6 November 2026) so that the saved
coefficient files `output/swappa_demand_repair_20261106/raw_d*_coef.npz` can be reused
without importing that module (it depends on scikit-learn, which is absent on this machine
and on Unity). The only changes: `continuous` accepts a per-row array for `shift`, and an
optional `price_override` / `peer_override` pair for integer-rounded deviations.
`check_design_matches_saved` verifies that the reconstructed standardization equals the saved
one, which is the same check the 6 November downstream script performed.

No scikit-learn, no threadpoolctl: limit BLAS threads with OMP_NUM_THREADS before import.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
PILOT_OUT = ROOT / 'output' / 'swappa_demand_repair_20261106'
CORE_OUT = ROOT / 'output' / 'swappa_core_20261108'
FOCAL = ['GadgetPickup', 'We Buy Mobile Corp']
MIN_SALES = 300

# Specifications with saved full-sample fits from 6 November (native-workbook references).
SPECS = {
    'raw_d0': dict(),                                   # revival-like baseline
    'raw_d1': dict(rich=True),                          # richer rival summaries
    'raw_d2': dict(rich=True, product_week=True),       # richer rivals plus SKU-by-week effects
    'raw_d6': dict(rich=True, flexible=True),           # richer rivals plus flexible price-gap hinges
    'raw_d8_fresh': dict(rich=True, fresh=True),        # richer rivals plus repricing-freshness controls (8 Nov)
    # the same three designs fitted on the existing (Stata-derived) reference panel, pilot_panel.parquet
    'd0_revival_like': dict(),
    'd1_rich_competition': dict(rich=True),
    'd6_flexible_gap': dict(rich=True, flexible=True),
}


def add_freshness(s):
    """Repricing-freshness covariates, held fixed under counterfactual deviations: whether the
    listing's price changed today relative to a verified previous-day observation (and whether
    it was cut), and the days since the last observed price change (reset at first appearance)."""
    s = s.sort_values(['Code', 'date']).reset_index(drop=True)
    chg = s.fresh_previous.to_numpy() & (s.Price.to_numpy() != s.previous_price.to_numpy())
    first = s.Code.ne(s.Code.shift()).to_numpy()
    s['chg_today'] = chg.astype(float)
    s['cut_today'] = (chg & (s.Price.to_numpy() < s.previous_price.to_numpy())).astype(float)
    s['gap_before'] = ((~s.fresh_previous.to_numpy()) & ~first).astype(float)
    chg_date = s.date.where(chg | first)
    s['dsc'] = (s.date - chg_date.groupby(s.Code).ffill()).dt.days.astype(float)
    return s


def continuous(s, rich=False, lag=False, shift=0.0, rounded=False, flexible=False, fresh=False,
               price_override=None, peer_override=None):
    """Dense covariates at (possibly counterfactual) own prices.

    `shift` may be a scalar or an array with one entry per row (day-specific rescaling for
    the fixed-level comparator). `price_override`/`peer_override` replace the shifted price
    and owned-peer mean directly (integer-rounded deviations)."""
    p = s.previous_price.to_numpy() if lag else s.Price.to_numpy(dtype=float)
    peer = s.previous_peer.to_numpy() if lag else s.peer_mean.to_numpy()
    if rounded:
        raise NotImplementedError('use price_override/peer_override for rounded deviations')
    if price_override is not None:
        p, peer = np.asarray(price_override, dtype=float), np.asarray(peer_override, dtype=float)
    else:
        p, peer = p * (1 + shift), peer * (1 + shift)
    ref = s.ref.to_numpy()
    hasr, hasp = s.rival_n.to_numpy() > 0, s.own_n.to_numpy() > 1
    lp = np.log(p / ref)
    gap = np.where(hasr, np.log(p / s.rival_min.fillna(1).to_numpy()), 0)
    X = dict(lp_ref=lp, gap_up=np.maximum(gap, 0), gap_dn=np.minimum(gap, 0),
             peer_rel=np.where(hasp, np.log(np.where(hasp, peer, 1) / ref), 0))
    age = np.log1p(s.since_seen.to_numpy())
    X['age'] = age
    for i, k in enumerate(np.log([2, 4, 8, 16, 32])):
        X[f'age_k{i}'] = np.maximum(age - k, 0)
    b = s.battery.to_numpy()
    X.update(batt=np.nan_to_num(b, nan=90) / 100, batt_missing=np.isnan(b).astype(float),
             batt_full=(b == 100).astype(float), log_rival_n=np.log1p(s.rival_n), no_rival=(~hasr).astype(float),
             log_own_n=np.log(s.own_n), has_peer=hasp.astype(float), log_interval=np.log(s.interval),
             ref_external=s.ref_external, initial_res=s.initial_res)
    if rich:
        X.update(rival_mean_ref=np.where(hasr, np.log(s.rival_mean.fillna(1) / ref), 0), rival_sd=s.rival_log_sd,
                 rival_median_ref=np.where(hasr, np.log(s.rival_median / ref), 0))
    if flexible:
        for i, k in enumerate([-.2, -.1, .1, .2]):
            X[f'gap_hinge{i}'] = np.maximum(gap - k, 0)
    if fresh:
        dsc = s.dsc.to_numpy()
        X.update(chg_today=s.chg_today.to_numpy(), cut_today=s.cut_today.to_numpy(), gap_before=s.gap_before.to_numpy(),
                 log1p_dsc=np.log1p(dsc), dsc_le2=(dsc <= 2).astype(float))
    z = pd.DataFrame(X).to_numpy(dtype=float)
    assert np.isfinite(z).all(), 'nonfinite covariates'
    return z, list(X)


def design(s, product_week=False, mean=None, sd=None, **kwargs):
    """Sparse design: constant, standardized dense block, fixed effects (first level dropped).
    If `mean`/`sd` are given (saved standardization), they are used instead of recomputing."""
    z, names = continuous(s, **kwargs)
    if sd is None:
        sd = z.std(0)
        sd[sd < 1e-8] = 1
        mean = z.mean(0)
    F = []
    for col in (['sku_week', 'date_str', 'seller_group'] if product_week else ['sku', 'date_str', 'seller_group']):
        c = pd.Categorical(s[col])
        valid = c.codes > 0
        F.append(sp.csr_matrix((np.ones(valid.sum()), (np.flatnonzero(valid), c.codes[valid] - 1)),
                               shape=(len(s), len(c.categories) - 1)))
    X = sp.hstack([np.ones((len(s), 1)), sp.csr_matrix((z - mean) / sd), *F], format='csr')
    return X, names, mean, sd


def fit(X, y, nd, b0=None):
    """Damped Newton logit with a unit ridge on the fixed-effect coefficients (as in the pilot)."""
    b = np.zeros(X.shape[1]) if b0 is None else b0.copy()
    if b0 is None:
        b[0] = np.log(y.mean() / (1 - y.mean()))
    lam = np.r_[np.zeros(nd), np.ones(X.shape[1] - nd)]

    def loss(v):
        zz = X @ v
        return np.logaddexp(0, zz).sum() - y @ zz + .5 * np.sum(lam * v * v)
    old = loss(b)
    for it in range(60):
        q = expit(X @ b)
        grad = X.T @ (q - y) + lam * b
        H = X.T @ X.multiply((q * (1 - q))[:, None]) + sp.diags(lam + 1e-9)
        step = spsolve(H.tocsc(), grad)
        rate = 1.
        while loss(b - rate * step) > old + 1e-8 and rate > 1e-7:
            rate *= .5
        b -= rate * step
        new = loss(b)
        if np.max(np.abs(rate * step)) < 1e-7 or abs(new - old) < 1e-8:
            break
        old = new
    score = float(np.max(np.abs(X.T @ (expit(X @ b) - y) + lam * b)))
    return b, dict(iterations=it + 1, score_max=score, objective=float(loss(b)))


def coef_path(tag):
    for folder in (PILOT_OUT, CORE_OUT):
        if (folder / f'{tag}_coef.npz').exists():
            return folder / f'{tag}_coef.npz'
    raise FileNotFoundError(tag)


def load_saved_fit(tag):
    z = np.load(coef_path(tag), allow_pickle=True)
    return dict(beta=z['beta'], names=[str(n) for n in z['names']], mean=z['mean'], sd=z['sd'])


def check_design_matches_saved(s, tag):
    """Rebuild the design for `tag` and confirm the standardization equals the saved one."""
    kw = SPECS[tag]
    X, names, mean, sd = design(s, **kw)
    saved = load_saved_fit(tag)
    assert names == saved['names'], (tag, names, saved['names'])
    assert np.allclose(mean, saved['mean']) and np.allclose(sd, saved['sd']), tag
    assert X.shape[1] == len(saved['beta']), (tag, X.shape, saved['beta'].shape)
    return X, names, mean, sd, saved['beta']


def penalized_hessian_inverse(X, beta, nd):
    """Inverse of the ridge-penalized Hessian (the bread of the sandwich), dense."""
    q = expit(X @ beta)
    lam = np.r_[np.zeros(nd), np.ones(X.shape[1] - nd)]
    H = (X.T @ X.multiply((q * (1 - q))[:, None])).toarray() + np.diag(lam + 1e-9)
    return np.linalg.inv(H), q


def twoway_cluster_cov(X, s, beta, nd, Hinv=None, q=None):
    """Two-way (listing, date) clustered sandwich with negative eigenvalues clipped (as run)."""
    if Hinv is None:
        Hinv, q = penalized_hessian_inverse(X, beta, nd)
    score = X.multiply((s.sale.to_numpy() - q)[:, None]).tocsr()
    meat = -(score.T @ score).toarray()
    for col in ['Code', 'date_str']:
        codes = pd.Categorical(s[col]).codes
        G = sp.csr_matrix((np.ones(len(s)), (codes, np.arange(len(s)))))
        a = (G @ score).toarray()
        meat += a.T @ a
    V = Hinv @ meat @ Hinv
    w, u = np.linalg.eigh((V + V.T) / 2)
    return (u * np.maximum(w, 0)) @ u.T, dict(negative_eigenvalues=int((w < 0).sum()),
                                              clipped_mass=float(-w[w < 0].sum()))


def date_influence(X, s, beta, nd, Hinv=None, q=None):
    """Influence of each date on beta-hat: psi_d = Hinv @ sum_{i in d} x_i (y_i - q_i).
    Returns (dates sorted, psi matrix T x k)."""
    if Hinv is None:
        Hinv, q = penalized_hessian_inverse(X, beta, nd)
    score = X.multiply((s.sale.to_numpy() - q)[:, None]).tocsr()
    dates = np.sort(s.date_str.unique())
    codes = pd.Categorical(s.date_str, categories=dates).codes
    G = sp.csr_matrix((np.ones(len(s)), (codes, np.arange(len(s)))), shape=(len(dates), len(s)))
    S = (G @ score).toarray()          # T x k daily score sums
    return dates, S @ Hinv.T           # T x k


def bartlett_lrcov(A, B, L=7):
    """Long-run covariance matrix of the columns of A (T x p) with the columns of B (T x q),
    Bartlett kernel with L lags, series demeaned. Returns p x q (not divided by T)."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    T = len(A)
    S = A.T @ B / T
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        S += w * (A[l:].T @ B[:-l] + A[:-l].T @ B[l:]) / T
    return S
