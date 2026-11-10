"""Shared estimators for the horse race against the equilibrium benchmark (9 November 2026).

All objects are day-level: for a window of T days with the seller's observed lattice level i_t (index into the log
lattice of `swappa_horserace_tables_20261109.py`, 40 = observed prices), day tables Rd[t, i] and CRd[t, i] give expected
revenue and cost-reference mass when the day's prices are scaled to lattice level i. Expected daily profit at cost ratio
theta is Rd - theta CRd.

Three procedures, all with the same demand, the same cost model c = theta r, the same variance conventions:

  equilibrium  (benchmark) the seller's price level satisfies the first-order condition of expected profit: the window
               mean of D_t(theta) = [Rd(i_t+1) - Rd(i_t-1)] - theta [CRd(i_t+1) - CRd(i_t-1)] is zero. Point estimate
               theta_N = sum dR / sum dCR; 95 percent interval by inverting |mean D(theta)| <= 1.96 sd(theta) on the grid
               (sd from the lag-0 sample variance of D_t, plus the demand delta method and the lag-0 demand cross term
               when supplied). This is the Rosse/BLP supply side with the level as the price direction, and it is the
               limit of the steady-state inequalities below as the shift shrinks to zero.
  steady state (equilibrium in expectation, PPHI-type inequalities) mean_t[A_k - theta B_k] <= z sd for the eight
               level shifts k, Bonferroni, zero allowance: the 8 November A2 and the 9 November steady-state row.
  record       the test of record of 08_CLOSING_PROOFS.md: the same inequalities with the Lemma 2 allowance (Hedge,
               c = 1, kappa_L = root-mean-square cell span, one maximal event delta_H = 0.0125, experience H cells),
               alpha_s = 0.0375, Bonferroni over the eight shifts.

Shifts are the lattice offsets J8 = (-20, -10, -5, -2, 2, 5, 10, 20), i.e. log shifts of +-1, +-2.5, +-5, +-10 percent;
the learner's action set is A9 = {0} u J8 (d = 9), the lattice version of the A3 levels.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

TH = np.round(np.arange(.2, 1.20001, .0025), 4)
J8 = np.array([-20, -10, -5, -2, 2, 5, 10, 20])
A9 = np.array([0, -20, -10, -5, -2, 2, 5, 10, 20])
MID = 40
DELTA_H, ALPHA = .0125, .05


def span(mask, grid=TH):
    if not np.any(mask):
        return None
    idx = np.flatnonzero(mask)
    return [float(grid[idx[0]]), float(grid[idx[-1]]), bool((np.diff(idx) > 1).any())]


def day_tables(z, ref='dec'):
    """Day sums of the per-cell tables. ref: 'dec' (Decluttr quote) or 'int' (internal index)."""
    cd, T = z['cell_day'], int(z['T'])
    CRc = z['CR'] if ref == 'dec' else z['CRI']
    Rd = np.zeros((T, z['R'].shape[1]))
    CRd = np.zeros_like(Rd)
    np.add.at(Rd, cd, z['R'])
    np.add.at(CRd, cd, CRc)
    return Rd, CRd


def sold_ratio(z, ref='dec'):
    return float((z['sum_r_sold'] if ref == 'dec' else z['sum_ri_sold']) / z['sum_p_sold'])


def lerner(iv, ratio):
    """Map a theta interval (or point) to the revenue-weighted sold-phone Lerner index, percent."""
    if iv is None:
        return None
    if np.ndim(iv) == 0:
        return 100 * (1 - float(iv) * ratio)
    return [round(100 * (1 - iv[1] * ratio), 2), round(100 * (1 - iv[0] * ratio), 2)]


# ----------------------------------------------------------------------------------------------- span statistics
def span_stats(z, ref='dec', theta=TH, offsets=A9, base=MID):
    """Per theta: root-mean-square cell span over the action set, largest daily sum of spans, design effect DEFF
    (as in swappa_closing_phi_20261109.py), and mean daily expected profit at the base level."""
    import scipy.sparse as sp
    CRc = z['CR'] if ref == 'dec' else z['CRI']
    cols = base + offsets
    Rc, Cc = z['R'][:, cols], CRc[:, cols]
    cd, T, nC = z['cell_day'], int(z['T']), int(z['nC'])
    Dm = sp.csr_matrix((np.ones(nC), (cd, np.arange(nC))), shape=(T, nC))
    Bt = nC / T
    out = {k: np.zeros(len(theta)) for k in ['kappa', 'blockmax', 'deff', 'astar', 'profit']}
    for gi, th in enumerate(theta):
        P = Rc - th * Cc                                      # cells x d
        K = P.max(1) - P.min(1)
        blk = Dm @ K
        astar = int(np.argmax(P.mean(0)))
        k2 = float((K ** 2).mean())
        Y = np.delete(P - P[:, [astar]], astar, 1)            # cells x (d-1)
        S = Dm @ Y
        vY = S.var(0, ddof=1)
        out['kappa'][gi] = np.sqrt(k2)
        out['blockmax'][gi] = blk.max()
        out['deff'][gi] = vY.max() / (Bt * k2)
        out['astar'][gi] = offsets[astar]
        out['profit'][gi] = float(P[:, 0].sum() / T)
    return out


def allowance_record(spans, cells_per_day, H, d=9, n_before=0.0, delta=DELTA_H):
    """Window-average daily allowance of the test of record (Lemma 2 with one maximal event), per theta.
    n_before: window cells observed before this (sub)window's first day beyond H (for half samples)."""
    B = np.asarray(cells_per_day, float)
    T = len(B)
    n = H + n_before + np.concatenate([[0.0], np.cumsum(B)[:-1]])
    n = np.maximum(n, 1.0)
    n_end = n[-1] + B[-1]
    x = np.log((d - 1) / delta)
    kap, blk, df = spans['kappa'], spans['blockmax'], np.maximum(spans['deff'], 1.0)
    reg = kap[:, None] * np.sqrt(np.log(d)) * np.sqrt(n + 1)[None] / n[None]
    e2 = reg + (np.sqrt(2 * n_end * kap ** 2 * df * x)[:, None] + (2 * blk * x / 3)[:, None]) / n[None]
    return (B[None] * e2).sum(1) / T


# ----------------------------------------------------------------------------------------------- moments
def moments(Rd, CRd, lev, days=None, J=J8):
    """Daily gains of the level shifts from the observed levels: A (n x T), B (n x T), and the first-order series
    dR, dCR (T,) by central difference at +-1 lattice step. lev: lattice index per day (T,)."""
    T = Rd.shape[0] if days is None else len(days)
    dd = np.arange(Rd.shape[0]) if days is None else np.asarray(days)
    lev = np.asarray(lev)
    R0, C0 = Rd[dd, lev], CRd[dd, lev]
    A = np.array([Rd[dd, lev + j] - R0 for j in J])
    B = np.array([CRd[dd, lev + j] - C0 for j in J])
    dR = Rd[dd, lev + 1] - Rd[dd, lev - 1]
    dC = CRd[dd, lev + 1] - CRd[dd, lev - 1]
    assert A.shape[1] == T
    return A, B, dR, dC, R0, C0


def demand_terms(z, ref, V, J=J8):
    """Gradients (of window means, observed level) for the eight shifts and the first-order difference."""
    gj = list(z['gj'])
    gR = z['gR']
    gC = z['gCR'] if ref == 'dec' else z['gCRI']
    i0 = gj.index(0)
    GA = np.array([gR[gj.index(j)] - gR[i0] for j in J])
    GB = np.array([gC[gj.index(j)] - gC[i0] for j in J])
    gdR = gR[gj.index(1)] - gR[gj.index(-1)]
    gdC = gC[gj.index(1)] - gC[gj.index(-1)]
    return dict(GA=GA, GB=GB, gdR=gdR, gdC=gdC, V=V, PSI=z['PSI_s'])


def _lag0_cov(X, Y):
    X = X - X.mean(-1, keepdims=True)
    Y = Y - Y.mean(-1, keepdims=True)
    return (X * Y).sum(-1) / X.shape[-1]


def equilibrium(dR, dC, theta=TH, dem=None, days=None, level=.05):
    """First-order-condition benchmark. Returns theta_N, its interval (inverted test), and the delta-method se."""
    T = len(dR)
    thN = dR.sum() / dC.sum()
    var = np.zeros(len(theta))
    for gi, th in enumerate(theta):
        u = dR - th * dC
        v = u.var(ddof=1) / T
        if dem is not None:
            g = dem['gdR'] - th * dem['gdC']
            v += g @ dem['V'] @ g
            if dem.get('PSI') is not None:
                U = dem['PSI'] if days is None else dem['PSI'][days]
                v += 2 * _lag0_cov(u, U @ g)
        var[gi] = v
    m = dR.mean() - theta * dC.mean()
    ok = np.abs(m) <= norm.ppf(1 - level / 2) * np.sqrt(var)
    se = float(np.sqrt(np.interp(thN, theta, var)) / abs(dC.mean()))
    return float(thN), span(ok, theta), se


def inequality_set(A, B, eps=None, theta=TH, dem=None, days=None, alpha=ALPHA, return_t=False):
    """Bonferroni set for mean(A_k - theta B_k) - eps(theta) <= z sd_k(theta), lag-0 sample variance plus the
    demand delta method and lag-0 cross term when supplied. eps: None (steady state) or array over theta."""
    n, T = A.shape
    mA, mB = A.mean(1), B.mean(1)
    vAA, vBB, vAB = A.var(1, ddof=1), B.var(1, ddof=1), _lag0_cov(A, B) * T / (T - 1)
    crit = norm.ppf(1 - alpha / n)
    ok = np.zeros(len(theta), bool)
    tmax = np.zeros(len(theta))
    for gi, th in enumerate(theta):
        m = mA - th * mB
        v = (vAA - 2 * th * vAB + th ** 2 * vBB) / T
        if dem is not None:
            G = dem['GA'] - th * dem['GB']
            v = v + np.einsum('ik,kl,il->i', G, dem['V'], G)
            if dem.get('PSI') is not None:
                U = (dem['PSI'] if days is None else dem['PSI'][days]) @ G.T     # T x n
                v = v + 2 * _lag0_cov(A - th * B, U.T)
        e = 0.0 if eps is None else eps[gi]
        t = (m - e) / np.sqrt(np.maximum(v, 1e-300))
        ok[gi] = bool((t <= crit).all())
        tmax[gi] = t.max()
    return (span(ok, theta), tmax) if return_t else span(ok, theta)


def regret_frontier(A, B, profit, theta=TH):
    """Minimal rationalizing regret: the largest window-average daily gain of a level shift at theta, and its share
    of mean daily expected profit at theta."""
    G = A.mean(1)[:, None] - theta[None] * B.mean(1)[:, None]
    R = np.maximum(G.max(0), 0.0)
    share = np.where(profit > 0, R / np.where(profit > 0, profit, 1.0), np.inf)
    return R, share


def connected(mask, center):
    """The connected run of True in `mask` containing index `center` (empty if mask[center] is False)."""
    if not mask[center]:
        return None
    lo = center
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    hi = center
    while hi < len(mask) - 1 and mask[hi + 1]:
        hi += 1
    return lo, hi
