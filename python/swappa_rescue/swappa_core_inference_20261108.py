"""Core inference for the Swappa revival, 8 November 2026.

For every seller with at least 300 terminal sales and every saved demand specification
(native-workbook references, 6 November fits), compute the cost-ratio sets implied by
steady-state no regret against level deviations, and repair or extend the inference in four
ways that the 6 November memo (items M1, M3, M6) asked for:

1. Inference. Keep the Bonferroni test as run (one-sided, alpha/n over n shifts, HAC(7)/T plus
   the two-way-clustered demand delta method) and add (a) a least-favorable max-t test whose
   critical value is the 95th percentile of the maximum of n standardized normals with the
   estimated joint correlation of the moments (valid because every moment is nonpositive at the
   truth; it exploits that the shifts in one direction are nearly collinear), and (b) the same
   test with a generalized-moment-selection refinement (moments with t below -sqrt(ln T) treated
   as slack).
2. Cross-covariance (M3). Add the omitted covariance between the daily payoff moments and the
   demand estimate, through the date-aggregated influence functions of the logit, with the same
   Bartlett kernel as the HAC term. Report its share of the moment variance.
3. Comparator (M1 / Q18). Besides the proportional level shifts (Phi-regret), compute the sets
   under the paper's main notion applied to a level action: the comparator holds the seller's
   price level fixed at l' every day, so day d's prices are rescaled by l'/l_d, where l_d is the
   seller's geometric mean of p/r that day (reference price r; also the internal market index).
4. Feasible prices (M6). Integer-rounded level deviations with owned-peer means recomputed
   from the rounded prices.

Usage: python swappa_core_inference_20261108.py [--specs raw_d0,raw_d1,raw_d6] [--sellers focal|all]
Outputs: output/swappa_core_20261108/<spec>_core.json (+ daily series npz per seller),
         docs/evidence/swappa_core_inference_20261108.json (compact, with hashes).
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
from swappa_common_20261108 import (ROOT, PILOT_OUT, SPECS, MIN_SALES, FOCAL, continuous, coef_path,  # noqa: E402
                                    check_design_matches_saved, penalized_hessian_inverse,
                                    twoway_cluster_cov, date_influence, bartlett_lrcov, add_freshness)

OUT = ROOT / 'output' / 'swappa_core_20261108'
EVID = ROOT / 'docs' / 'evidence' / 'swappa_core_inference_20261108.json'
TH = np.round(np.arange(.2, 1.20001, .0025), 4)
K8 = [-.10, -.05, -.025, -.01, .01, .025, .05, .10]
G1 = sorted(K8 + [-.005, .005])
ALPHA, LAGS, NDRAW = .05, 7, 20000
RNG_DRAWS = np.random.default_rng(20261108).standard_normal((NDRAW, 12))


def span(mask, grid=TH):
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)
    return [float(grid[idx[0]]), float(grid[idx[-1]]), bool((np.diff(idx) > 1).any())]


class Seller:
    """One seller's rows under one demand specification; evaluates counterfactual prices."""

    def __init__(self, s, X, beta, nd, kw, sd, name, dates):
        kw = {k: v for k, v in kw.items() if k != 'product_week'}   # design-only flag
        ix = s.Seller.eq(name).to_numpy()
        self.g = g = s.loc[ix].reset_index(drop=True)
        self.XX = X[ix].tocsr()
        self.beta, self.nd, self.kw, self.sd = beta, nd, kw, sd
        self.z0 = self.XX @ beta
        self.dense0, _ = continuous(g, **kw)
        self.price = g.Price.to_numpy(dtype=float)
        self.ref = g.ref.to_numpy()
        self.peer = g.peer_mean.to_numpy()
        self.T_all = len(dates)
        # day index aligned with the panel-wide date vector (for the influence functions)
        self.dpos = pd.Categorical(g.date_str, categories=dates).codes
        self.days = np.sort(np.unique(self.dpos))
        self.T = len(self.days)
        dc = np.searchsorted(self.days, self.dpos)
        self.D = sp.csr_matrix((np.ones(len(g)), (dc, np.arange(len(g)))), shape=(self.T, len(g)))
        sold = g.sale.eq(1).to_numpy()
        self.mean_price_sold = float(self.price[sold].mean())
        self.mean_ref_sold = float(self.ref[sold].mean())
        self.sales = int(sold.sum())
        # daily levels for the fixed-level comparator
        self.level, self.level_row = {}, {}
        for key, col in [('ref', 'ref'), ('int', 'internal_centered'), ('rmin', 'rival_min')]:
            den = g[col].to_numpy(dtype=float)
            valid = np.isfinite(den) & (den > 0)
            lr = np.where(valid, np.log(self.price / np.where(valid, den, 1.0)), 0.0)
            cnt = np.asarray(self.D @ valid.astype(float)).ravel()
            daymean = np.asarray(self.D @ lr).ravel() / np.maximum(cnt, 1)
            # days with no valid denominator (no rival) keep the overall level: no rescaling
            overall = float(lr[valid].mean())
            daymean = np.where(cnt > 0, daymean, overall)
            self.level_row[key] = daymean[dc]
            self.level[key] = dict(overall=overall, daily_sd=float(np.std(daymean)),
                                   daily_range=[float(daymean.min()), float(daymean.max())],
                                   share_rows_valid=float(valid.mean()))
        self.cache = {}

    def ev(self, shift=None, price_override=None, peer_override=None, key=None):
        if key is not None and key in self.cache:
            return self.cache[key]
        if price_override is None:
            zz, _ = continuous(self.g, shift=shift, **self.kw)
            p1 = self.price * (1 + shift)
        else:
            zz, _ = continuous(self.g, price_override=price_override, peer_override=peer_override, **self.kw)
            p1 = np.asarray(price_override, dtype=float)
        delta = (zz - self.dense0) / self.sd
        q = expit(self.z0 + delta @ self.beta[1:self.nd])
        w = q * (1 - q)
        out = dict(R=np.asarray(self.D @ (p1 * q)).ravel(), CR=np.asarray(self.D @ (self.ref * q)).ravel(),
                   S=np.asarray(self.D @ q).ravel())
        for k2, v in [('gR', p1 * w), ('gCR', self.ref * w)]:
            gr = np.asarray(self.XX.T @ v).ravel()
            gr[1:self.nd] += delta.T @ v
            out[k2] = gr
        if key is not None:
            self.cache[key] = out
        return out

    def rounded(self, k):
        """Integer-rounded level deviation: p -> max(round((1+k)p), 1); owned-peer means from the
        rounded own prices in the same market, with the few own offers outside the risk set
        (censored rows) scaled continuously."""
        g = self.g
        p1 = np.maximum(np.round(self.price * (1 + k)), 1.0)
        mk = g.market_id.to_numpy()
        codes, inv = np.unique(mk, return_inverse=True)
        sum_p = np.bincount(inv, weights=self.price, minlength=len(codes))
        sum_p1 = np.bincount(inv, weights=p1, minlength=len(codes))
        own_total = g.own_total.to_numpy(dtype=float)
        own_n = g.own_n.to_numpy(dtype=float)
        total_new = sum_p1[inv] + (1 + k) * (own_total - sum_p[inv])
        peer1 = np.where(own_n > 1, (total_new - p1) / np.maximum(own_n - 1, 1), np.nan)
        return p1, peer1


def comparator_series(m: Seller, kind):
    """Daily gain series for a comparator. Returns labels, A (n x T), B (n x T), GA, GB (n x k)."""
    base = m.ev(shift=0.0, key='base')
    rows, labels = [], []
    if kind in ('phi_base', 'phi_g1'):
        for k in (K8 if kind == 'phi_base' else G1):
            rows.append(m.ev(shift=k, key=('phi', k)))
            labels.append(f'shift:{k:+.4f}')
    elif kind in ('ext_ref', 'ext_int', 'ext_rmin'):
        key = {'ext_ref': 'ref', 'ext_int': 'int', 'ext_rmin': 'rmin'}[kind]
        lev, row = m.level[key], m.level_row[key]
        for k in [0.0] + K8:
            target = lev['overall'] + np.log1p(k)          # fixed log level l'
            shift = np.exp(target - row) - 1                 # l'/l_d - 1 per row
            rows.append(m.ev(shift=shift))
            labels.append(f'level:{k:+.4f}')
    elif kind == 'phi_base_rounded':
        for k in K8:
            p1, peer1 = m.rounded(k)
            rows.append(m.ev(price_override=p1, peer_override=peer1))
            labels.append(f'rshift:{k:+.4f}')
    else:
        raise ValueError(kind)
    A = np.array([r['R'] - base['R'] for r in rows])
    B = np.array([r['CR'] - base['CR'] for r in rows])
    GA = np.array([r['gR'] - base['gR'] for r in rows]) / m.T
    GB = np.array([r['gCR'] - base['gCR'] for r in rows]) / m.T
    return labels, A, B, GA, GB


def lf_critical(corr, ndraw_cols, slack=None):
    """95th percentile of max_k Z_k, Z ~ N(0, corr); moments flagged in `slack` are excluded."""
    n = corr.shape[0]
    w, U = np.linalg.eigh(corr)
    Z = RNG_DRAWS[:, :n] @ (U * np.sqrt(np.maximum(w, 0))).T
    if slack is not None:
        Z = Z[:, ~slack] if (~slack).any() else Z[:, :1] * 0 - np.inf
    return float(np.quantile(Z.max(1), 1 - ALPHA))


def infer(labels, A, B, GA, GB, V, PSI_s, T):
    """Sets over the theta grid under several tests. PSI_s: T x k influence of the seller's days."""
    n = len(labels)
    mA, mB = A.mean(1), B.mean(1)
    Haa, Hab, Hbb = bartlett_lrcov(A.T, A.T, LAGS), bartlett_lrcov(A.T, B.T, LAGS), bartlett_lrcov(B.T, B.T, LAGS)
    Daa, Dab, Dbb = GA @ V @ GA.T, GA @ V @ GB.T, GB @ V @ GB.T
    # cross term: daily payoff moment series vs daily influence on the delta-method term
    UA, UB = PSI_s @ GA.T, PSI_s @ GB.T                      # T x n each
    Caa, Cab, Cba, Cbb = (bartlett_lrcov(A.T, UA, LAGS), bartlett_lrcov(A.T, UB, LAGS),
                          bartlett_lrcov(B.T, UA, LAGS), bartlett_lrcov(B.T, UB, LAGS))
    # alternative demand variance from the date-aggregated influence functions (diagnostic)
    IFaa, IFab, IFbb = (bartlett_lrcov(UA, UA, LAGS) * T, bartlett_lrcov(UA, UB, LAGS) * T,
                        bartlett_lrcov(UB, UB, LAGS) * T)
    crit_bonf = norm.ppf(1 - ALPHA / n)
    kappa = np.sqrt(np.log(T))
    res = dict(labels=labels, n=n, crit_bonferroni=float(crit_bonf), kappa_gms=float(kappa))
    ok = {k: np.zeros(len(TH), bool) for k in ['fit', 'bonf', 'maxt', 'maxt_gms', 'bonf_cross', 'maxt_cross']}
    crit_track = {k: [] for k in ['maxt', 'maxt_gms', 'maxt_cross']}
    share_cross, share_demand, share_if_ratio = [], [], []
    tmid = None
    for ti, th in enumerate(TH):
        m = mA - th * mB
        Sig_h = (Haa - th * (Hab + Hab.T) + th ** 2 * Hbb) / T
        Sig_d = Daa - th * (Dab + Dab.T) + th ** 2 * Dbb
        Cross = Caa - th * (Cab + Cba) + th ** 2 * Cbb
        Cross = Cross + Cross.T
        Sig = Sig_h + Sig_d
        Sig_x = Sig + Cross
        sd = np.sqrt(np.maximum(np.diag(Sig), 1e-300))
        sd_x = np.sqrt(np.maximum(np.diag(Sig_x), 1e-300))
        t, t_x = m / sd, m / sd_x
        ok['fit'][ti] = bool((m <= 0).all())
        ok['bonf'][ti] = bool((t <= crit_bonf).all())
        ok['bonf_cross'][ti] = bool((t_x <= crit_bonf).all())
        corr = Sig / np.outer(sd, sd)
        c = lf_critical(corr, n)
        ok['maxt'][ti] = bool(t.max() <= c)
        slack = t < -kappa
        c_g = lf_critical(corr, n, slack) if (~slack).any() else -np.inf
        ok['maxt_gms'][ti] = bool(((t[~slack].max() if (~slack).any() else -np.inf) <= c_g))
        corr_x = Sig_x / np.outer(sd_x, sd_x)
        c_x = lf_critical(corr_x, n)
        ok['maxt_cross'][ti] = bool(t_x.max() <= c_x)
        crit_track['maxt'].append(c)
        crit_track['maxt_gms'].append(c_g)
        crit_track['maxt_cross'].append(c_x)
        if ok['fit'][ti]:
            share_cross.append(np.diag(Cross) / np.diag(Sig_x))
            share_demand.append(np.diag(Sig_d) / np.diag(Sig))
            IF = IFaa - th * (IFab + IFab.T) + th ** 2 * IFbb
            share_if_ratio.append(np.sqrt(np.diag(IF) / np.maximum(np.diag(Sig_d), 1e-300)))
            tmid = t if tmid is None else tmid
    res['sets_theta'] = {k: span(v) for k, v in ok.items()}

    def t_at(th):
        m = mA - th * mB
        Sig = (Haa - th * (Hab + Hab.T) + th ** 2 * Hbb) / T + Daa - th * (Dab + Dab.T) + th ** 2 * Dbb
        return np.round(m / np.sqrt(np.maximum(np.diag(Sig), 1e-300)), 2).tolist(), np.round(m, 3).tolist()
    res['t_at'] = t_at
    res['crit_range'] = {k: [float(np.min(v)), float(np.max(v))] for k, v in crit_track.items()}
    if share_cross:
        res['variance_shares_in_fit_set'] = dict(
            cross_term_share=np.round(np.mean(share_cross, 0), 4).tolist(),
            demand_share=np.round(np.mean(share_demand, 0), 4).tolist(),
            if_over_twoway_sd_ratio=np.round(np.mean(share_if_ratio, 0), 3).tolist())
    # which moments bind the Bonferroni endpoints
    for side, key in [('lower', 0), ('upper', 1)]:
        sv = res['sets_theta']['bonf']
        if sv:
            th = sv[key] - (.0025 if key == 0 else -.0025)
            m = mA - th * mB
            Sig = (Haa - th * (Hab + Hab.T) + th ** 2 * Hbb) / T + Daa - th * (Dab + Dab.T) + th ** 2 * Dbb
            t = m / np.sqrt(np.maximum(np.diag(Sig), 1e-300))
            res[f'binding_{side}'] = labels[int(np.argmax(t))]
    return res, dict(mA=mA, mB=mB)


def lerner(iv, m):
    if iv is None:
        return None
    return [round(100 * (1 - iv[1] * m.mean_ref_sold / m.mean_price_sold), 2),
            round(100 * (1 - iv[0] * m.mean_ref_sold / m.mean_price_sold), 2)]


PANELS = {'raw': 'raw_reference_panel.parquet', 'pilot': 'pilot_panel.parquet'}
PANEL = 'raw'


def run_spec(tag, sellers_mode):
    kw = SPECS[tag]
    t0 = time.time()
    s = add_freshness(pd.read_parquet(PILOT_OUT / PANELS[PANEL]).reset_index(drop=True))
    X, names, mean, sd, beta = check_design_matches_saved(s, tag)
    nd = len(names) + 1
    Hinv, q = penalized_hessian_inverse(X, beta, nd)
    V, vinfo = twoway_cluster_cov(X, s, beta, nd, Hinv, q)
    dates, PSI = date_influence(X, s, beta, nd, Hinv, q)
    T_all = len(dates)
    counts = s.groupby('Seller').sale.sum()
    sellers = FOCAL if sellers_mode == 'focal' else counts[counts >= MIN_SALES].sort_values(ascending=False).index.tolist()
    out = dict(spec=tag, panel=PANELS[PANEL], design=kw, rows=int(len(s)), sales=int(s.sale.sum()), dates=T_all,
               covariance=vinfo, seconds_setup=round(time.time() - t0, 1), sellers={})
    print(f'[{tag}] setup {out["seconds_setup"]}s, V clipped {vinfo}', flush=True)
    for name in sellers:
        t1 = time.time()
        m = Seller(s, X, beta, nd, kw, sd, name, dates)
        PSI_s = PSI[m.days]                                  # the seller's days only
        rec = dict(sales=m.sales, rows=len(m.g), days=m.T, mean_price_sold=m.mean_price_sold,
                   mean_ref_sold=m.mean_ref_sold, levels=m.level, comparators={})
        series = {}
        for kind in ['phi_base', 'phi_g1', 'ext_ref', 'ext_int', 'ext_rmin', 'phi_base_rounded']:
            labels, A, B, GA, GB = comparator_series(m, kind)
            r, mm = infer(labels, A, B, GA, GB, V, PSI_s, m.T)
            r['lerner'] = {k: lerner(v, m) for k, v in r['sets_theta'].items()}
            r['moment_means_at_fit_mid'] = None
            fit_iv = r['sets_theta']['fit']
            if fit_iv:
                thm = .5 * (fit_iv[0] + fit_iv[1])
                r['moment_means_at_fit_mid'] = dict(theta=thm, m=np.round(mm['mA'] - thm * mm['mB'], 3).tolist())
            rec['comparators'][kind] = r
            series[kind] = dict(A=A, B=B, GA=GA, GB=GB, labels=np.array(labels))
        # every comparator's t-statistics and moment means at the Phi-base fitted midpoint
        fit_phi = rec['comparators']['phi_base']['sets_theta']['fit']
        if fit_phi:
            thm = .5 * (fit_phi[0] + fit_phi[1])
            for kind, r in rec['comparators'].items():
                tt, mm_ = r['t_at'](thm)
                r['t_at_phi_fit_mid'] = dict(theta=thm, t=tt, m=mm_)
        for r in rec['comparators'].values():
            r.pop('t_at', None)
        # identity checks: zero shift, rounded zero-shift equals base
        base = m.ev(shift=0.0, key='base')
        p0, peer0 = m.rounded(0.0)
        r0 = m.ev(price_override=p0, peer_override=peer0)
        rec['checks'] = dict(zero_shift_R_equals_fit=bool(np.allclose(base['R'], np.asarray(m.D @ (m.price * expit(m.z0))).ravel())),
                             rounded_zero_equals_base=bool(np.allclose(r0['R'], base['R']) and np.allclose(r0['CR'], base['CR'])),
                             direct_profit_identity=None)
        # direct row profit vs aggregated moments at theta = .8, shift +1%
        zz, _ = continuous(m.g, shift=.01, **m.kw)
        q1 = expit(m.z0 + ((zz - m.dense0) / sd) @ beta[1:nd])
        q0 = expit(m.z0)
        direct = np.sum((m.price * 1.01 - .8 * m.ref) * q1 - (m.price - .8 * m.ref) * q0)
        e1 = m.ev(shift=.01, key=('phi', .01))
        agg = np.sum(e1['R'] - base['R'] - .8 * (e1['CR'] - base['CR']))
        rec['checks']['direct_profit_identity'] = float(abs(direct - agg))
        rec['seconds'] = round(time.time() - t1, 1)
        out['sellers'][name] = rec
        np.savez_compressed(OUT / f'{tag}_{name.replace(" ", "_").replace("/", "_")}_series.npz',
                            days=m.days, dates=dates[m.days], price_sold=m.mean_price_sold, ref_sold=m.mean_ref_sold,
                            **{f'{k}_{f}': v[f] for k, v in series.items() for f in ['A', 'B', 'GA', 'GB', 'labels']})
        pb = rec['comparators']['phi_base']
        pe = rec['comparators']['ext_ref']
        print(f'[{tag}] {name[:22]:22s} sales {m.sales:5d} | phi bonf {pb["lerner"]["bonf"]} maxt {pb["lerner"]["maxt"]} '
              f'gms {pb["lerner"]["maxt_gms"]} cross {pb["lerner"]["maxt_cross"]} | ext fit {pe["lerner"]["fit"]} '
              f'bonf {pe["lerner"]["bonf"]} maxt {pe["lerner"]["maxt"]} | rounded bonf '
              f'{rec["comparators"]["phi_base_rounded"]["lerner"]["bonf"]} | {rec["seconds"]}s', flush=True)
        if fit_phi:
            for kind in ['ext_ref', 'ext_int', 'ext_rmin']:
                print(f'      {kind:8s} t at phi-mid: {rec["comparators"][kind]["t_at_phi_fit_mid"]["t"]}', flush=True)
    (OUT / f'{tag}_core.json').write_text(json.dumps(out, indent=1, default=float))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    specs = args[args.index('--specs') + 1].split(',') if '--specs' in args else ['raw_d0', 'raw_d1', 'raw_d2', 'raw_d6']
    sellers_mode = args[args.index('--sellers') + 1] if '--sellers' in args else 'all'
    global PANEL
    PANEL = args[args.index('--panel') + 1] if '--panel' in args else 'raw'
    evid_path = EVID if PANEL == 'raw' else EVID.with_name(EVID.stem + f'_{PANEL}panel.json')
    started = time.time()
    results = {}
    for tag in specs:
        cached = OUT / f'{tag}_core.json'
        if '--resume' in args and cached.exists():
            results[tag] = json.loads(cached.read_text())
            print(f'[{tag}] loaded cached core JSON', flush=True)
        else:
            results[tag] = run_spec(tag, sellers_mode)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [Path(__file__), Path(__file__).with_name('swappa_common_20261108.py'),
               PILOT_OUT / PANELS[PANEL], *[coef_path(t) for t in specs]]}
    compact = dict(date='2026-11-08', status='local computation; inference provisional until the Unity block bootstrap is read',
                   theta_grid=[.2, 1.2, .0025], alpha=ALPHA, hac_lags=LAGS, draws=NDRAW, shifts_base=K8, shifts_g1=G1,
                   hashes=hashes, seconds=round(time.time() - started, 1), results={})
    for tag, out in results.items():
        compact['results'][tag] = {name: {k: dict(sets_theta=c['sets_theta'], lerner=c['lerner'], crit_range=c['crit_range'],
                                                  variance_shares=c.get('variance_shares_in_fit_set'),
                                                  binding=[c.get('binding_lower'), c.get('binding_upper')],
                                                  t_at_phi_fit_mid=c.get('t_at_phi_fit_mid'))
                                          for k, c in rec['comparators'].items()}
                                   for name, rec in out['sellers'].items()}
        compact['results'][tag]['_covariance'] = out['covariance']
        compact['results'][tag]['_levels'] = {name: rec['levels'] for name, rec in out['sellers'].items()}
        compact['results'][tag]['_checks'] = {name: rec['checks'] for name, rec in out['sellers'].items()}
    compact['panel'] = PANELS[PANEL]
    evid_path.parent.mkdir(parents=True, exist_ok=True)
    evid_path.write_text(json.dumps(compact, indent=1, default=float))
    print('CORE_INFERENCE_COMPLETE', round(time.time() - started, 1), 's', flush=True)


if __name__ == '__main__':
    main()
