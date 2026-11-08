"""Learning-consistent alternatives to the steady-state assumption (8 November 2026, v2).

The revival's Assumption A2 (stationary play, zero long-run regret) turns the paper's limit
result into population inequalities. This script computes what the paper's own finite-sample
logic delivers instead, using two things the data show: how long each seller has been learning
(experience H from rating counts, branch `longevity.json`) and at what granularity it decides
(one price per SKU-day; `decision_granularity.csv`).

Comparators (deviation classes), each a set of row-level price rescalings:
  phi       proportional shifts of the seller's actual prices (the revival's A3; 8 moments);
  ext_day   the paper's external regret on a daily level action: hold the seller's day-level
            markup over the internal market index fixed at l' (9 levels incl. its own mean);
  ext_cell  external regret on the SKU-day decision: hold every SKU-day cell's markup fixed at a
            uniform l' (9 levels), which also questions the cross-SKU dispersion of markups;
  ctx_gen   the paper's CONTEXTUAL external regret with the context = phone generation group
            (older/SE, 11-12, 13-15): a fixed markup per context (3 x 9 moments);
  ctx_model contextual external regret with the context = device model (9 moments per model
            with at least 300 listing-days).

Finite-sample layers, all requiring only a stationary ENVIRONMENT (the paper's own i.i.d.
assumption), never stationary play:
  SS   steady state, eps = 0 (the revival's A2), Bonferroni over the moments;
  E(H) experience-indexed relaxed obedience: per-round expected regret of a certified
       full-information learner started H rounds earlier at most 2 K_round sqrt(ln d / H)
       (the branch's convention; constant for the methods author); clocks SKU-day (the data's)
       and listing-day; contextual versions use the context's own H and rounds;
  AW   any-window pathwise envelope (strongly adaptive full-information learner, no H, no
       stationarity at all): window regret <= sqrt(sum_rounds K_round^2) [sqrt(2 ln(d T_life))
       + sqrt(2 ln(d/alpha))].
  Bandit variant of E(H) on realized own profit (listing-day clock), the sparse-outcome case.
Tests: Bonferroni and least-favorable max-t over the comparator's moments (SS and E(H)); AW on
window sums without sampling error.

Usage: python swappa_learning_layer_20261108.py [--spec raw_d0] [--sellers all|focal4] [--eps-mode start|window]
  --eps-mode start  (default) per-round allowance evaluated at the window's first round: c K / sqrt(H);
  --eps-mode window exact window average of the per-round allowance c K / sqrt(H + s), s = 1..N,
                    i.e. c K 2 (sqrt(H + N) - sqrt(H)) / N; H = 0 recovers the restart (prefix) calibration
                    and H -> infinity the steady state. Output then carries the suffix _epswindow.
Output: docs/evidence/swappa_learning_layer_20261108.json and a printed table.
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
from scipy.stats import norm, spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swappa_common_20261108 import (ROOT, PILOT_OUT, SPECS, MIN_SALES, continuous, check_design_matches_saved,  # noqa: E402
                                    penalized_hessian_inverse, twoway_cluster_cov, add_freshness, bartlett_lrcov)
from swappa_core_inference_20261108 import Seller, K8, TH  # noqa: E402

OUT = ROOT / 'output' / 'swappa_core_20261108'
EVID = ROOT / 'docs' / 'evidence' / 'swappa_learning_layer_20261108.json'
LEVELS = [0.0] + K8
ALPHA, LAGS = .05, 7
DRAWS = np.random.default_rng(20261108).standard_normal((6000, 400))
FOCAL4 = ['GadgetPickup', 'AMS Traders Inc.', 'UPGRADE SOLUTION INC', 'We Buy Mobile Corp']


def span(mask):
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)
    return [float(TH[idx[0]]), float(TH[idx[-1]]), bool((np.diff(idx) > 1).any())]


def gen_group(dev):
    g = pd.Series(dev).str.replace('apple-iphone-', '', regex=False).str.extract(r'^(\d+)')[0].astype(float).to_numpy()
    return np.where(g >= 13, 'g13_15', np.where(g >= 11, 'g11_12', 'older_se'))


def joint_pieces(A, B, GA, GB, V, T):
    """Moment means, HAC and delta-method blocks for n daily series."""
    return dict(mA=A.mean(1), mB=B.mean(1), Haa=bartlett_lrcov(A.T, A.T, LAGS), Hab=bartlett_lrcov(A.T, B.T, LAGS),
                Hbb=bartlett_lrcov(B.T, B.T, LAGS), Daa=GA @ V @ GA.T, Dab=GA @ V @ GB.T, Dbb=GB @ V @ GB.T, T=T)


def sig_at(P, th):
    Sig = (P['Haa'] - th * (P['Hab'] + P['Hab'].T) + th ** 2 * P['Hbb']) / P['T'] + P['Daa'] - th * (P['Dab'] + P['Dab'].T) + th ** 2 * P['Dbb']
    return P['mA'] - th * P['mB'], Sig


def sets_for(P, eps_by_theta, use_maxt=True):
    """Bonferroni and max-t sets for moments m(theta) - eps(theta) <= 0; eps: n x G or None."""
    n = len(P['mA'])
    crit_b = norm.ppf(1 - ALPHA / n)
    ok_b = np.zeros(len(TH), bool); ok_m = np.zeros(len(TH), bool)
    for gi, th in enumerate(TH):
        m, Sig = sig_at(P, th)
        e = 0.0 if eps_by_theta is None else eps_by_theta[:, gi]
        sd = np.sqrt(np.maximum(np.diag(Sig), 1e-300))
        t = (m - e) / sd
        ok_b[gi] = bool((t <= crit_b).all())
        if use_maxt:
            corr = Sig / np.outer(sd, sd)
            w, U = np.linalg.eigh(corr)
            Z = DRAWS[:, :n] @ (U * np.sqrt(np.maximum(w, 0))).T
            ok_m[gi] = bool(t.max() <= np.quantile(Z.max(1), 1 - ALPHA))
    return span(ok_b), (span(ok_m) if use_maxt else None)


def main():
    args = sys.argv[1:]
    spec = args[args.index('--spec') + 1] if '--spec' in args else 'raw_d0'
    mode = args[args.index('--sellers') + 1] if '--sellers' in args else 'all'
    eps_mode = args[args.index('--eps-mode') + 1] if '--eps-mode' in args else 'start'
    assert eps_mode in ('start', 'window')
    evid = EVID if eps_mode == 'start' else EVID.with_name(EVID.stem + '_epswindow.json')

    def expf(H, N):
        """Experience factor multiplying c K: 1/sqrt(H) at the window start, or its exact average over the window."""
        if eps_mode == 'start':
            return 1.0 / np.sqrt(max(H, 1.0))
        N = max(float(N), 1.0); H = max(float(H), 0.0)
        return 2.0 * (np.sqrt(H + N) - np.sqrt(H)) / N
    t0 = time.time()
    s = add_freshness(pd.read_parquet(PILOT_OUT / 'raw_reference_panel.parquet').reset_index(drop=True))
    X, names, mean, sd, beta = check_design_matches_saved(s, spec)
    nd = len(names) + 1
    kw = {k: v for k, v in SPECS[spec].items() if k != 'product_week'}
    Hinv, q = penalized_hessian_inverse(X, beta, nd)
    V, _ = twoway_cluster_cov(X, s, beta, nd, Hinv, q)
    dates = np.sort(s.date_str.unique())
    counts = s.groupby('Seller').sale.sum()
    sellers = counts[counts >= MIN_SALES].sort_values(ascending=False).index.tolist()
    if mode == 'focal4':
        sellers = [x for x in sellers if x in FOCAL4]
    longevity = json.loads((OUT / 'branch_longevity.json').read_text())
    convergence = json.loads((OUT / 'branch_convergence.json').read_text())
    gran = pd.read_csv(OUT / 'decision_granularity.csv').set_index('seller')
    out = dict(date='2026-11-08', spec=spec, eps_mode=eps_mode, levels=LEVELS, sellers={})
    for name in sellers:
        t1 = time.time()
        m = Seller(s, X, beta, nd, SPECS[spec], sd, name, dates)
        g, T = m.g, m.T
        rows_per_day = len(g) / T
        # decision cells (SKU-day) and levels
        cell_key = (g.sku.astype(str) + '|' + g.date_str).to_numpy()
        ccodes, cinv = np.unique(cell_key, return_inverse=True)
        nC = len(ccodes)
        C = sp.csr_matrix((np.ones(len(g)), (cinv, np.arange(len(g)))), shape=(nC, len(g)))
        den = g.internal_centered.to_numpy(dtype=float)
        valid = np.isfinite(den) & (den > 0)
        lr = np.where(valid, np.log(m.price / np.where(valid, den, 1.0)), 0.0)
        cnt = np.asarray(C @ valid.astype(float)).ravel()
        overall = float(lr[valid].mean())
        cell_level = np.where(cnt > 0, np.asarray(C @ lr).ravel() / np.maximum(cnt, 1), overall)
        row_cell_level = cell_level[cinv]
        row_day_level = m.level_row['int']
        # contexts
        gen = gen_group(g.device_id.to_numpy())
        model = g.device_id.to_numpy()
        model_counts = pd.Series(model).value_counts()
        models = [x for x in model_counts.index if model_counts[x] >= 300]
        lon = longevity[name]
        H_listing = float(lon['implied_presample_listing_days'])
        H_cell = H_listing * nC / len(g)
        base = m.ev(shift=0.0, key='base')
        base_rows = expit(m.z0)
        sold = g.sale.eq(1).to_numpy()
        mp, mr = float(m.price[sold].mean()), float(m.ref[sold].mean())

        def lern(iv):
            return None if iv is None else [round(100 * (1 - iv[1] * mr / mp), 2), round(100 * (1 - iv[0] * mr / mp), 2)]

        def level_shift(k, ref_level, mask=None):
            sh = np.exp(overall + np.log1p(k) - ref_level) - 1 if not isinstance(ref_level, dict) else None
            return sh

        # comparator definitions: list of (label, context_mask, shift vector)
        comps = {}
        comps['phi'] = [(f'shift{k:+.3f}', None, np.full(len(g), k)) for k in K8]
        comps['ext_day'] = [(f'lvl{k:+.3f}', None, np.exp(overall + np.log1p(k) - row_day_level) - 1) for k in LEVELS]
        comps['ext_cell'] = [(f'lvl{k:+.3f}', None, np.exp(overall + np.log1p(k) - row_cell_level) - 1) for k in LEVELS]
        ctx_gen = []
        for c in ['older_se', 'g11_12', 'g13_15']:
            mask = gen == c
            if mask.sum() < 300:
                continue
            mean_c = float(lr[mask & valid].mean()) if (mask & valid).any() else overall
            for k in LEVELS:
                sh = np.where(mask, np.exp(mean_c + np.log1p(k) - row_cell_level) - 1, 0.0)
                ctx_gen.append((f'{c}|lvl{k:+.3f}', mask, sh))
        comps['ctx_gen'] = ctx_gen
        ctx_model = []
        for c in models:
            mask = model == c
            mean_c = float(lr[mask & valid].mean()) if (mask & valid).any() else overall
            for k in LEVELS:
                sh = np.where(mask, np.exp(mean_c + np.log1p(k) - row_cell_level) - 1, 0.0)
                ctx_model.append((f'{c}|lvl{k:+.3f}', mask, sh))
        comps['ctx_model'] = ctx_model
        rec = dict(sales=int(sold.sum()), days=T, cells=int(nC), rows=int(len(g)), cells_per_day=round(nC / T, 1),
                   implied_years=lon['implied_years_at_window_pace'], H=dict(sku_day=round(H_cell), listing_day=round(H_listing)),
                   n_models=len(models), comparators={})
        daily_profit_fn = lambda th: float((base['R'] - th * base['CR']).mean())  # noqa: E731
        for cname, items in comps.items():
            if not items:
                continue
            A, B, GA, GB = [], [], [], []
            Rrow, CRrow = [base_rows * m.price], [base_rows * m.ref]
            for label, mask, sh in items:
                e = m.ev(shift=sh)
                A.append(e['R'] - base['R']); B.append(e['CR'] - base['CR'])
                GA.append((e['gR'] - base['gR']) / T); GB.append((e['gCR'] - base['gCR']) / T)
                zz, _ = continuous(g, shift=sh, **kw)
                qq = expit(m.z0 + ((zz - m.dense0) / sd) @ beta[1:nd])
                Rrow.append(m.price * (1 + sh) * qq); CRrow.append(m.ref * qq)
            A, B, GA, GB = map(np.array, (A, B, GA, GB))
            Rrow, CRrow = np.array(Rrow), np.array(CRrow)       # (n+1) x rows: actual plus each comparator price
            P = joint_pieces(A, B, GA, GB, V, T)
            n = len(items)
            d_eff = len(LEVELS)                                 # actions per decision: the nine levels
            # per-round payoff ranges over the action set, by clock and (for contextual) by context
            Rc, CRc = Rrow @ C.T, CRrow @ C.T                  # (n+1) x cells
            Rc, CRc = np.asarray(Rc), np.asarray(CRc)
            masks = [it[1] for it in items]
            ctx_of_moment = [None if mk is None else tuple(np.flatnonzero(mk)[:1]) for mk in masks]
            # cell membership of contexts
            cell_ctx = {}
            for i, mk in enumerate(masks):
                if mk is None:
                    cell_ctx[i] = None
                else:
                    cm = np.asarray(C @ mk.astype(float)).ravel() > 0
                    cell_ctx[i] = cm
            epsE_cell = np.zeros((n, len(TH))); epsE_row = np.zeros((n, len(TH))); epsB_row = np.zeros((n, len(TH)))
            Baw_cell = np.zeros((n, len(TH))); Baw_row = np.zeros((n, len(TH)))
            for gi, th in enumerate(TH):
                Pic = Rc - th * CRc
                Pir = Rrow - th * CRrow
                Kc_all = Pic.max(0) - Pic.min(0)               # cells: range over actual + all comparator prices
                Kr_all = Pir.max(0) - Pir.min(0)
                Lr = np.maximum(0, m.price * 1.10 - th * m.ref)
                for i in range(n):
                    cm, rm = cell_ctx[i], masks[i]
                    if cm is None:
                        Kc, Kr, L, Hc, Hr, Nc, Nr = Kc_all, Kr_all, Lr, H_cell, H_listing, nC, len(g)
                    else:
                        Kc, Kr, L = Kc_all[cm], Kr_all[rm], Lr[rm]
                        shc, shr = cm.mean(), rm.mean()
                        Hc, Hr, Nc, Nr = H_cell * shc, H_listing * shr, cm.sum(), rm.sum()
                    epsE_cell[i, gi] = 2 * np.sqrt(np.log(d_eff)) * expf(Hc, Nc) * Kc.sum() / T
                    epsE_row[i, gi] = 2 * np.sqrt(np.log(d_eff)) * expf(Hr, Nr) * Kr.sum() / T
                    epsB_row[i, gi] = 2 * np.sqrt(d_eff * np.log(d_eff)) * expf(Hr, Nr) * L.sum() / T
                    c_aw = np.sqrt(2 * np.log(d_eff / ALPHA))
                    Baw_cell[i, gi] = np.sqrt((Kc ** 2).sum()) * (np.sqrt(2 * np.log(d_eff * (Hc + Nc))) + c_aw) / T
                    Baw_row[i, gi] = np.sqrt((Kr ** 2).sum()) * (np.sqrt(2 * np.log(d_eff * (Hr + Nr))) + c_aw) / T
            use_maxt = n <= 400
            ss_b, ss_m = sets_for(P, None, use_maxt)
            eh_c_b, eh_c_m = sets_for(P, epsE_cell, use_maxt)
            eh_r_b, eh_r_m = sets_for(P, epsE_row, use_maxt)
            bd_b, _ = sets_for(P, epsB_row, False)
            # any-window: pathwise on window means, no sampling term
            mm = np.array([sig_at(P, th)[0] for th in TH]).T   # n x G
            aw_c = span((mm <= Baw_cell).all(0)); aw_r = span((mm <= Baw_row).all(0))
            fit = span((mm <= 0).all(0))
            mid = int(np.argmin(np.abs(TH - .8)))
            rec['comparators'][cname] = dict(
                n_moments=n, sets_theta=dict(fit=fit, SS_bonf=ss_b, SS_maxt=ss_m, E_skuday_bonf=eh_c_b, E_skuday_maxt=eh_c_m,
                                             E_listing_bonf=eh_r_b, E_listing_maxt=eh_r_m, Bandit_listing_bonf=bd_b, AW_skuday=aw_c, AW_listing=aw_r),
                allowance_share_of_daily_profit_at_theta08=dict(E_skuday=round(float(epsE_cell[:, mid].mean() / daily_profit_fn(.8)), 4),
                                                                AW_skuday=round(float(Baw_cell[:, mid].mean() / daily_profit_fn(.8)), 4),
                                                                E_listing=round(float(epsE_row[:, mid].mean() / daily_profit_fn(.8)), 4),
                                                                Bandit_listing=round(float(epsB_row[:, mid].mean() / daily_profit_fn(.8)), 4)))
            rec['comparators'][cname]['lerner'] = {k: lern(v) for k, v in rec['comparators'][cname]['sets_theta'].items()}
            L = rec['comparators'][cname]['lerner']
            print(f"{name[:18]:18s} {cname:9s} n={n:3d} | fit {L['fit']} SS {L['SS_bonf']} | E sku {L['E_skuday_bonf']} E list {L['E_listing_bonf']} "
                  f"bandit {L['Bandit_listing_bonf']} | AW sku {L['AW_skuday']} AW list {L['AW_listing']} | eps/profit "
                  f"{rec['comparators'][cname]['allowance_share_of_daily_profit_at_theta08']}", flush=True)
        conv = convergence.get(name, {}).get('internal', {})
        kstars = [v['kstar_pct'] for v in conv.values() if v.get('kstar_pct') is not None]
        rec['behavior'] = dict(monthly_kstar_pct=kstars, mean_abs_kstar=round(float(np.mean(np.abs(kstars))), 3) if kstars else None,
                               sd_kstar=round(float(np.std(kstars)), 3) if len(kstars) > 1 else None,
                               across_day_sd_mean_level=float(gran.loc[name[:20], 'across_day_sd_of_mean_level']),
                               within_day_cross_sku_sd=float(gran.loc[name[:20], 'within_day_cross_sku_sd']))
        rec['seconds'] = round(time.time() - t1, 1)
        out['sellers'][name] = rec
        print(f"   done {name[:18]} yrs {rec['implied_years']} H_cell {rec['H']['sku_day']} models {len(models)} |k*| {rec['behavior']['mean_abs_kstar']} ({rec['seconds']}s)", flush=True)
    yrs = np.array([out['sellers'][n]['implied_years'] for n in out['sellers']])
    for key in ['mean_abs_kstar', 'sd_kstar', 'across_day_sd_mean_level']:
        vals = np.array([out['sellers'][n]['behavior'][key] if out['sellers'][n]['behavior'][key] is not None else np.nan for n in out['sellers']])
        ok = np.isfinite(vals)
        if ok.sum() >= 4:
            rho, p = spearmanr(yrs[ok], vals[ok])
            out[f'spearman_years_vs_{key}'] = dict(rho=round(float(rho), 3), p=round(float(p), 3), n=int(ok.sum()))
            print('Spearman(years,', key, ')', round(rho, 3), 'p', round(p, 3))
    out['seconds'] = round(time.time() - t0, 1)
    evid.write_text(json.dumps(out, indent=1, default=float))
    print('LEARNING_LAYER_COMPLETE', out['seconds'], flush=True)


if __name__ == '__main__':
    main()
