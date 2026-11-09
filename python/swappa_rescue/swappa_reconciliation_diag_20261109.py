"""Reconciliation diagnostics for Niccolo's review of 9 November 2026 (GadgetPickup, one demand fit).

Measures, without changing the maintained specification, the numerical effect of the findings in
his package `JPE_SWAPPA_FEEDBACK_FOR_LORENZO_V01_20261109` that are code-level or quantitative:

  F11  demand-moment cross-covariance missing from the learning-layer variance;
  F12  sampling uncertainty of the estimated allowance (through demand) ignored in the E(H) test;
  F13  AW evaluated at point-estimated demand;
  F04  clock: SKU-day cells versus a daily portfolio round (all cells priced before the day's feedback);
  F06  scale: local candidate-specific cell ranges versus a common span known to the learner;
  F08  exact discrete window sum versus the integral formula of `--eps-mode window`;
  F18  revenue-weighted (ratio of means) versus unweighted sold-unit Lerner index;
  Q20  the allowance implied by Niccolo's proposed Q20-S / Q20-W constants (Hedge in family F);
  plus the half-sample stationarity check (July to September against October to December).

Comparators: `phi` (the eight level shifts of A3) and `ext_day` (the fixed daily level against the
internal market index, nine levels, the proposed main regret notion of 8 November). Demand: one saved
fit (default raw_d0). Every set is a DIAGNOSTIC; none is a proposed headline. Common spans are the
window maxima of the relevant ranges, so they are LOWER bounds on any span a learner could know ex
ante: the common-K rows are optimistic for the common-K route.

Usage: python swappa_reconciliation_diag_20261109.py [--spec raw_d0] [--seller GadgetPickup] [--draws 200]
Output: docs/evidence/swappa_reconciliation_diag_20261109.json and a printed summary.
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
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swappa_common_20261108 import (ROOT, PILOT_OUT, SPECS, continuous, check_design_matches_saved,  # noqa: E402
                                    penalized_hessian_inverse, twoway_cluster_cov, add_freshness, bartlett_lrcov,
                                    date_influence, coef_path)
from swappa_core_inference_20261108 import Seller, K8, TH  # noqa: E402

OUT = ROOT / 'output' / 'swappa_core_20261108'
EVID = ROOT / 'docs' / 'evidence' / 'swappa_reconciliation_diag_20261109.json'
LEVELS = [0.0] + K8
D_ACT = len(LEVELS)                  # nine levels, as in the learning layer
ALPHA, LAGS = .05, 7
DRAWS = np.random.default_rng(20261109).standard_normal((20000, 16))
SPLIT = '2023-10-01'


def span(mask, grid=TH):
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)
    return [float(grid[idx[0]]), float(grid[idx[-1]]), bool((np.diff(idx) > 1).any())]


def q20_round_factor(s, d, delta_s):
    """Q20-S per-round bound in units of K for Hedge (B_reg = ln d, sigma = 1, c = 1), capped at 1."""
    s = np.asarray(s, dtype=float)
    first = np.sqrt(np.log(d)) * np.sqrt(s) / (s - 1)
    second = np.sqrt(2 * np.log(2 * d / delta_s) / (s - 1))
    return np.minimum(first + second, 1.0)


class Pieces:
    """Moment means and variance blocks for n daily series, with the cross block of the core script."""

    def __init__(self, A, B, GA, GB, V, PSI_s, days_mask=None):
        if days_mask is not None:
            A, B = A[:, days_mask], B[:, days_mask]
        self.T = A.shape[1]
        self.mA, self.mB = A.mean(1), B.mean(1)
        self.Haa, self.Hab, self.Hbb = (bartlett_lrcov(A.T, A.T, LAGS), bartlett_lrcov(A.T, B.T, LAGS),
                                        bartlett_lrcov(B.T, B.T, LAGS))
        self.Daa, self.Dab, self.Dbb = GA @ V @ GA.T, GA @ V @ GB.T, GB @ V @ GB.T
        if days_mask is None:
            UA, UB = PSI_s @ GA.T, PSI_s @ GB.T
            self.Caa, self.Cab, self.Cba, self.Cbb = (bartlett_lrcov(A.T, UA, LAGS), bartlett_lrcov(A.T, UB, LAGS),
                                                      bartlett_lrcov(B.T, UA, LAGS), bartlett_lrcov(B.T, UB, LAGS))
        else:
            self.Caa = None

    def at(self, th, hac=True, demand=True, cross=False):
        m = self.mA - th * self.mB
        S = np.zeros((len(m), len(m)))
        if hac:
            S = S + (self.Haa - th * (self.Hab + self.Hab.T) + th ** 2 * self.Hbb) / self.T
        if demand:
            S = S + self.Daa - th * (self.Dab + self.Dab.T) + th ** 2 * self.Dbb
        if cross and self.Caa is not None:
            Cx = self.Caa - th * (self.Cab + self.Cba) + th ** 2 * self.Cbb
            S = S + Cx + Cx.T
        return m, S


def test_set(P, eps=None, alpha=ALPHA, hac=True, demand=True, cross=False, maxt=False):
    """Keep theta if every (m - eps)/sd <= Bonferroni (or least-favorable max-t) critical value."""
    ok = np.zeros(len(TH), bool)
    n = len(P.mA)
    crit_b = norm.ppf(1 - alpha / n)
    for gi, th in enumerate(TH):
        m, S = P.at(th, hac, demand, cross)
        e = 0.0 if eps is None else eps[:, gi]
        sd = np.sqrt(np.maximum(np.diag(S), 1e-300))
        t = (m - e) / sd
        if maxt:
            corr = S / np.outer(sd, sd)
            w, U = np.linalg.eigh(corr)
            Z = DRAWS[:, :n] @ (U * np.sqrt(np.maximum(w, 0))).T
            ok[gi] = bool(t.max() <= np.quantile(Z.max(1), 1 - alpha))
        else:
            ok[gi] = bool((t <= crit_b).all())
    return span(ok)


def point_set(P, bound, demand_alpha=None):
    """Keep theta if m_hat - z * sd_demand <= bound (pathwise route; demand projection if demand_alpha)."""
    n = len(P.mA)
    ok = np.zeros(len(TH), bool)
    z = 0.0 if demand_alpha is None else norm.ppf(1 - demand_alpha / n)
    for gi, th in enumerate(TH):
        m, S = P.at(th, hac=False, demand=True, cross=False)
        sd = np.sqrt(np.maximum(np.diag(S), 0))
        ok[gi] = bool((m - z * sd <= bound[:, gi]).all())
    return span(ok)


def main():
    args = sys.argv[1:]
    spec = args[args.index('--spec') + 1] if '--spec' in args else 'raw_d0'
    seller = args[args.index('--seller') + 1] if '--seller' in args else 'GadgetPickup'
    ndraw = int(args[args.index('--draws') + 1]) if '--draws' in args else 200
    t0 = time.time()
    s = add_freshness(pd.read_parquet(PILOT_OUT / 'raw_reference_panel.parquet').reset_index(drop=True))
    X, names, mean, sd, beta = check_design_matches_saved(s, spec)
    nd = len(names) + 1
    kw = {k: v for k, v in SPECS[spec].items() if k != 'product_week'}
    Hinv, q = penalized_hessian_inverse(X, beta, nd)
    V, vinfo = twoway_cluster_cov(X, s, beta, nd, Hinv, q)
    dates, PSI = date_influence(X, s, beta, nd, Hinv, q)
    print(f'setup {time.time() - t0:.1f}s, V {vinfo}', flush=True)

    m = Seller(s, X, beta, nd, SPECS[spec], sd, seller, dates)
    g, T = m.g, m.T
    PSI_s = PSI[m.days]
    day_dates = dates[m.days]
    first_half = day_dates < SPLIT
    sold = g.sale.eq(1).to_numpy()
    mp, mr = float(m.price[sold].mean()), float(m.ref[sold].mean())

    def lern(iv):
        return None if iv is None else [round(100 * (1 - iv[1] * mr / mp), 2), round(100 * (1 - iv[0] * mr / mp), 2)]

    # cells (SKU-day) and the day index of each cell
    cell_key = (g.sku.astype(str) + '|' + g.date_str).to_numpy()
    ccodes, cinv = np.unique(cell_key, return_inverse=True)
    nC = len(ccodes)
    C = sp.csr_matrix((np.ones(len(g)), (cinv, np.arange(len(g)))), shape=(nC, len(g)))
    row_day = np.searchsorted(m.days, m.dpos)
    cell_day = np.zeros(nC, int)
    cell_day[cinv] = row_day

    # experience on both clocks (branch longevity record)
    lon = json.loads((OUT / 'branch_longevity.json').read_text())[seller]
    H_listing = float(lon['implied_presample_listing_days'])
    H_cell = H_listing * nC / len(g)
    days_window = (pd.Timestamp(lon['ratings_last'][0]) - pd.Timestamp(lon['ratings_first'][0])).days
    H_day = float(lon['implied_presample_sales']) / (lon['terminal_sales_between'] / days_window)

    # comparator shift vectors (as in the learning layer)
    den = g.internal_centered.to_numpy(dtype=float)
    valid = np.isfinite(den) & (den > 0)
    lr = np.where(valid, np.log(m.price / np.where(valid, den, 1.0)), 0.0)
    overall = float(lr[valid].mean())
    comps = {'phi': [np.full(len(g), k) for k in K8],
             'ext_day': [np.exp(overall + np.log1p(k) - m.level_row['int']) - 1 for k in LEVELS]}
    base = m.ev(shift=0.0, key='base')
    q0 = expit(m.z0)
    out = dict(date='2026-11-09', spec=spec, seller=seller, status='diagnostic; not a proposed headline',
               rows=int(len(g)), days=int(T), cells=int(nC), sales=int(sold.sum()),
               H=dict(cell=round(H_cell), listing=round(H_listing), day=round(H_day, 1)),
               mean_price_sold=mp, mean_ref_sold=mr, covariance=vinfo, comparators={})
    log9 = np.log(D_ACT)
    arrays = {}
    for cname, shifts in comps.items():
        n = len(shifts)
        A, B, GA, GB, Rrow, CRrow, deltas = [], [], [], [], [q0 * m.price], [q0 * m.ref], []
        for sh in shifts:
            e = m.ev(shift=sh)
            A.append(e['R'] - base['R']); B.append(e['CR'] - base['CR'])
            GA.append((e['gR'] - base['gR']) / T); GB.append((e['gCR'] - base['gCR']) / T)
            zz, _ = continuous(g, shift=sh, **kw)
            dl = (zz - m.dense0) / sd
            deltas.append(dl)
            qq = expit(m.z0 + dl @ beta[1:nd])
            Rrow.append(m.price * (1 + sh) * qq); CRrow.append(m.ref * qq)
        A, B, GA, GB = map(np.array, (A, B, GA, GB))
        Rrow, CRrow = np.array(Rrow), np.array(CRrow)
        P = Pieces(A, B, GA, GB, V, PSI_s)
        Rc, CRc = np.asarray(Rrow @ C.T), np.asarray(CRrow @ C.T)              # (n+1) x cells
        Rd, CRd = np.asarray(Rrow @ m.D.T), np.asarray(CRrow @ m.D.T)          # (n+1) x days

        # allowances per day (one row per moment; identical across moments), on the full theta grid
        G = len(TH)
        sumKc, maxKc, meanKday, maxKday, sqKc = (np.zeros(G) for _ in range(5))
        Kc08 = Kday08 = None
        for gi, th in enumerate(TH):
            Pic, Pid = Rc - th * CRc, Rd - th * CRd
            Kc = Pic.max(0) - Pic.min(0)
            Kd = Pid.max(0) - Pid.min(0)
            sumKc[gi], maxKc[gi], meanKday[gi], maxKday[gi], sqKc[gi] = Kc.sum(), Kc.max(), Kd.mean(), Kd.max(), (Kc ** 2).sum()
            if abs(th - .8) < 1e-9:
                Kc08, Kday08 = Kc.copy(), Kd.copy()
        tau_c, tau_d = nC, T
        eps = {}
        eps['E_current'] = 2 * np.sqrt(log9 / H_cell) * sumKc / T
        eps['E_day_localrange'] = 2 * np.sqrt(log9 / H_day) * meanKday
        eps['E_day_commonK'] = 2 * np.sqrt(log9 / H_day) * maxKday
        eps['E_cell_commonK'] = 2 * np.sqrt(log9 / H_cell) * (nC / T) * maxKc
        # Q20-S expectation route: window sum of per-round bounds, fixed-window allocation delta_H / tau
        dH = .025
        fc = q20_round_factor(np.arange(H_cell + 1, H_cell + tau_c + 1), D_ACT, dH / tau_c).sum()
        fd = q20_round_factor(np.arange(H_day + 1, H_day + tau_d + 1), D_ACT, dH / tau_d).sum()
        eps['Q20exp_cell_commonK'] = maxKc * fc / T
        eps['Q20exp_day_commonK'] = maxKday * fd / T
        eps['Q20exp_cell_localK'] = sumKc / nC * fc / T          # average local range as if it were common
        # Q20-W realized route: add the martingale term K sqrt(2 tau ln(d/delta_M)), delta_M = .0125
        dM = .0125
        fc_r = q20_round_factor(np.arange(H_cell + 1, H_cell + tau_c + 1), D_ACT, .0125 / tau_c).sum()
        fd_r = q20_round_factor(np.arange(H_day + 1, H_day + tau_d + 1), D_ACT, .0125 / tau_d).sum()
        bound = {}
        bound['Q20real_cell_commonK'] = maxKc * (fc_r + np.sqrt(2 * tau_c * np.log(D_ACT / dM))) / T
        bound['Q20real_day_commonK'] = maxKday * (fd_r + np.sqrt(2 * tau_d * np.log(D_ACT / dM))) / T
        c_aw = np.sqrt(2 * np.log(D_ACT / ALPHA))
        bound['AW_current'] = np.sqrt(sqKc) * (np.sqrt(2 * np.log(D_ACT * (H_cell + nC))) + c_aw) / T
        c_aw2 = np.sqrt(2 * np.log(D_ACT / (ALPHA / 2)))
        bound['AW_budget'] = np.sqrt(sqKc) * (np.sqrt(2 * np.log(D_ACT * (H_cell + nC))) + c_aw2) / T
        sqKday = np.array([(lambda Pid: ((Pid.max(0) - Pid.min(0)) ** 2).sum())(Rd - th * CRd) for th in TH])
        # arrays for offline evaluation of any allowance formula without re-evaluating demand
        arrays[cname] = dict(TH=TH, mA=P.mA, mB=P.mB, Haa=P.Haa, Hab=P.Hab, Hbb=P.Hbb, Daa=P.Daa, Dab=P.Dab, Dbb=P.Dbb,
                             Caa=P.Caa, Cab=P.Cab, Cba=P.Cba, Cbb=P.Cbb, T=T, nC=nC, H_cell=H_cell, H_day=H_day,
                             H_listing=H_listing, sumKc=sumKc, maxKc=maxKc, meanKday=meanKday, maxKday=maxKday,
                             sqKc=sqKc, sqKday=sqKday, mean_price_sold=mp, mean_ref_sold=mr,
                             A=A, B=B, day_dates=day_dates.astype(str))
        E = {k: np.tile(v, (n, 1)) for k, v in eps.items()}
        Bd = {k: np.tile(v, (n, 1)) for k, v in bound.items()}
        sets = {}
        sets['fit'] = span(np.array([bool((P.at(th, False, False)[0] <= 0).all()) for th in TH]))
        sets['SS_bonf'] = test_set(P)
        sets['SS_bonf_cross'] = test_set(P, cross=True)
        sets['E_current_bonf'] = test_set(P, E['E_current'])
        sets['E_current_maxt'] = test_set(P, E['E_current'], maxt=True)
        sets['E_current_bonf_cross'] = test_set(P, E['E_current'], cross=True)                  # F11
        sets['E_current_maxt_cross'] = test_set(P, E['E_current'], cross=True, maxt=True)       # F11
        for k in ['E_day_localrange', 'E_day_commonK', 'E_cell_commonK', 'Q20exp_cell_commonK', 'Q20exp_day_commonK',
                  'Q20exp_cell_localK']:
            sets[f'{k}_bonf_cross'] = test_set(P, E[k], cross=True, alpha=ALPHA if not k.startswith('Q20') else ALPHA - dH)
        sets['AW_current_point'] = point_set(P, Bd['AW_current'])
        sets['AW_budget_demandproj'] = point_set(P, Bd['AW_budget'], demand_alpha=ALPHA / 2)            # F13
        sets['Q20real_cell_commonK_demandproj'] = point_set(P, Bd['Q20real_cell_commonK'], demand_alpha=ALPHA / 2)
        sets['Q20real_day_commonK_demandproj'] = point_set(P, Bd['Q20real_day_commonK'], demand_alpha=ALPHA / 2)
        # half samples: fitted sets, and Bonferroni with the half-sample HAC term only (no demand term)
        for lab, msk in [('first_half', first_half), ('second_half', ~first_half)]:
            Ph = Pieces(A, B, GA, GB, V, PSI_s, days_mask=msk)
            sets[f'{lab}_fit'] = span(np.array([bool((Ph.at(th, False, False)[0] <= 0).all()) for th in TH]))
            sets[f'{lab}_SS_bonf_hac_only'] = test_set(Ph, demand=False)
        mid = int(np.argmin(np.abs(TH - .8)))
        daily_profit08 = float((base['R'] - .8 * base['CR']).mean())
        rec = dict(n_moments=n, sets_theta=sets, lerner={k: lern(v) for k, v in sets.items()},
                   allowance_share_of_daily_profit_at_theta08={k: round(float(v[mid] / daily_profit08), 4) for k, v in {**eps, **bound}.items()},
                   ranges_at_theta08=dict(cell=dict(mean=float(Kc08.mean()), median=float(np.median(Kc08)), p90=float(np.quantile(Kc08, .9)),
                                                    p99=float(np.quantile(Kc08, .99)), max=float(Kc08.max())),
                                          day=dict(mean=float(Kday08.mean()), max=float(Kday08.max()),
                                                   sum_cell_ranges_per_day_mean=float(sumKc[mid] / T)),
                                          daily_expected_profit=daily_profit08),
                   q20_window_factors=dict(cell_expectation=float(fc), day_expectation=float(fd),
                                           cell_realized_martingale=float(np.sqrt(2 * tau_c * np.log(D_ACT / dM))),
                                           day_realized_martingale=float(np.sqrt(2 * tau_d * np.log(D_ACT / dM))),
                                           placeholder_cell=float(2 * np.sqrt(log9 / H_cell) * tau_c),
                                           placeholder_day=float(2 * np.sqrt(log9 / H_day) * tau_d)))
        # F08: exact discrete window sum (worst-case order within a day) against the integral formula, theta = .8
        if cname == 'phi':
            order = np.lexsort((-Kc08, cell_day))                 # chronological days, larger ranges first within a day
            srounds = H_cell + 1 + np.arange(nC)
            exact = float((Kc08[order] / np.sqrt(srounds)).sum())
            integral = float(2 * (np.sqrt(H_cell + nC) - np.sqrt(H_cell)) / nC * Kc08.sum())
            start = float(Kc08.sum() / np.sqrt(H_cell))
            rec['F08_window_sum_theta08'] = dict(exact_weighted_worst_order=exact, integral_average_times_sum=integral,
                                                 window_start_envelope=start, integral_over_exact=integral / exact)
        # F18: unweighted sold-unit Lerner at the E(H) and SS endpoints
        ul = {}
        for k in ['SS_bonf', 'E_current_bonf']:
            iv = sets[k]
            if iv:
                ul[k] = [round(100 * float(np.mean(1 - iv[1] * m.ref[sold] / m.price[sold])), 2),
                         round(100 * float(np.mean(1 - iv[0] * m.ref[sold] / m.price[sold])), 2)]
        rec['F18_unweighted_unit_lerner'] = ul
        # F12: allowance uncertainty through demand, by parametric draws of beta (phi only)
        if cname == 'phi' and ndraw > 0:
            t2 = time.time()
            w, U = np.linalg.eigh((V + V.T) / 2)
            L = U * np.sqrt(np.maximum(w, 0))
            rng = np.random.default_rng(91109)
            sub = np.flatnonzero((TH >= .6) & (TH <= 1.0))
            mdraw = np.zeros((ndraw, n, len(sub)))
            edraw = np.zeros((ndraw, len(sub)))
            shifts_all = [0.0] + list(K8)
            dl_all = [np.zeros_like(deltas[0])] + deltas
            for b in range(ndraw):
                bb = beta + L @ rng.standard_normal(len(beta))
                z0b = m.XX @ bb
                R_rows, CR_rows = [], []
                for sh, dl in zip(shifts_all, dl_all):
                    qb = expit(z0b + dl @ bb[1:nd])
                    R_rows.append(m.price * (1 + sh) * qb); CR_rows.append(m.ref * qb)
                R_rows, CR_rows = np.array(R_rows), np.array(CR_rows)
                Rdb, CRdb = np.asarray(R_rows @ m.D.T), np.asarray(CR_rows @ m.D.T)
                Rcb, CRcb = np.asarray(R_rows @ C.T), np.asarray(CR_rows @ C.T)
                Ab, Bb = (Rdb[1:] - Rdb[0]).mean(1), (CRdb[1:] - CRdb[0]).mean(1)
                for j, gi in enumerate(sub):
                    th = TH[gi]
                    mdraw[b, :, j] = Ab - th * Bb
                    Pic = Rcb - th * CRcb
                    edraw[b, j] = 2 * np.sqrt(log9 / H_cell) * (Pic.max(0) - Pic.min(0)).sum() / T
            var_m = mdraw.var(0)                                       # n x sub
            var_me = (mdraw - edraw[:, None, :]).var(0)
            extra = np.zeros((n, len(TH))); extra_m = np.zeros((n, len(TH)))
            extra[:, sub] = var_me; extra_m[:, sub] = var_m
            outside = np.setdiff1d(np.arange(len(TH)), sub)
            extra[:, outside] = np.nan

            def set_with(varblock):
                ok = np.zeros(len(TH), bool)
                crit = norm.ppf(1 - ALPHA / n)
                for gi in sub:
                    th = TH[gi]
                    mm, Sh = P.at(th, hac=True, demand=False)
                    sdv = np.sqrt(np.diag(Sh) + varblock[:, gi])
                    ok[gi] = bool(((mm - E['E_current'][:, gi]) / sdv <= crit).all())
                return span(ok)
            rec['F12'] = dict(draws=ndraw, theta_subgrid=[float(TH[sub[0]]), float(TH[sub[-1]])],
                              set_draw_var_moment_only=set_with(extra_m), set_draw_var_moment_minus_allowance=set_with(extra),
                              sd_ratio_at_theta08=np.round(np.sqrt(var_me[:, np.argmin(np.abs(TH[sub] - .8))] /
                                                                   var_m[:, np.argmin(np.abs(TH[sub] - .8))]), 3).tolist(),
                              allowance_cv_at_theta08=float(edraw[:, np.argmin(np.abs(TH[sub] - .8))].std() /
                                                            edraw[:, np.argmin(np.abs(TH[sub] - .8))].mean()),
                              seconds=round(time.time() - t2, 1))
            rec['F12']['lerner'] = {k: lern(rec['F12'][k]) for k in ['set_draw_var_moment_only', 'set_draw_var_moment_minus_allowance']}
        out['comparators'][cname] = rec
        L_ = rec['lerner']
        print(f'{cname}: fit {L_["fit"]} SS {L_["SS_bonf"]} SSx {L_["SS_bonf_cross"]} | E {L_["E_current_bonf"]} Ex {L_["E_current_bonf_cross"]} '
              f'| Eday {L_["E_day_localrange_bonf_cross"]} EdayK {L_["E_day_commonK_bonf_cross"]} EcellK {L_["E_cell_commonK_bonf_cross"]} '
              f'| Q20exp cellK {L_["Q20exp_cell_commonK_bonf_cross"]} dayK {L_["Q20exp_day_commonK_bonf_cross"]} '
              f'| AW {L_["AW_current_point"]} AWd {L_["AW_budget_demandproj"]} | Q20real cellK {L_["Q20real_cell_commonK_demandproj"]} '
              f'dayK {L_["Q20real_day_commonK_demandproj"]} | halves {L_["first_half_fit"]} {L_["second_half_fit"]}', flush=True)
        print('   shares', rec['allowance_share_of_daily_profit_at_theta08'], flush=True)
    out['hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     [Path(__file__), Path(__file__).with_name('swappa_common_20261108.py'),
                      Path(__file__).with_name('swappa_core_inference_20261108.py'),
                      PILOT_OUT / 'raw_reference_panel.parquet', coef_path(spec), OUT / 'branch_longevity.json']}
    out['seconds'] = round(time.time() - t0, 1)
    EVID.parent.mkdir(parents=True, exist_ok=True)
    EVID.write_text(json.dumps(out, indent=1, default=float))
    np.savez_compressed(OUT / f'reconciliation_diag_arrays_{spec}_{seller.replace(" ", "_")}.npz',
                        **{f'{c}__{k}': np.asarray(v) for c, d in arrays.items() for k, v in d.items()})
    print('RECONCILIATION_DIAG_COMPLETE', out['seconds'], flush=True)


if __name__ == '__main__':
    main()
