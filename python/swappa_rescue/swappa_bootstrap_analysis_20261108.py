"""Analyze the Unity moving-block bootstrap of the Swappa level-deviation moments (8 Nov 2026).

Reads the task npz files written by `swappa_block_bootstrap_20261108.py` (pulled to
output/swappa_core_20261108/boot/) and the full-sample pieces (recomputed here from the saved
6 November fits), and reports for each spec and seller:

- dispersion check: bootstrap standard deviation of m*_k(theta) against the full-sample analytic
  standard error se_k(theta) (HAC(7)/T plus the two-way-clustered delta method), at the fitted
  midpoint and at the Bonferroni endpoints; and for the price coefficients of the logit;
- percentile-t least-favorable critical value c*(theta) = 95th percentile over replications of
  max_k [m*_k(theta) - m_hat_k(theta)] / se*_k(theta), the resulting bootstrap set
  {theta: max_k m_hat_k(theta)/se_k(theta) <= c*(theta)}, and its Lerner index;
- size check at the full-sample null point closest to binding (theta_0 in the fitted set with
  the largest max_k t_k): rejection rates of the Bonferroni and normal least-favorable max-t
  tests when applied to the bootstrap moments at theta_0 (the bootstrap truth there is the
  full-sample moment vector, which is nonpositive);
- coverage of the full-sample fitted midpoint by each replication's Bonferroni set.

Usage: python swappa_bootstrap_analysis_20261108.py [--boot DIR]
Output: docs/evidence/swappa_bootstrap_analysis_20261108.json
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swappa_common_20261108 import (ROOT, PILOT_OUT, SPECS, MIN_SALES, check_design_matches_saved,  # noqa: E402
                                    penalized_hessian_inverse, twoway_cluster_cov)
from swappa_block_bootstrap_20261108 import seller_series, moments_and_se, TH, K8  # noqa: E402
from swappa_core_inference_20261108 import lf_critical  # noqa: E402

EVID = ROOT / 'docs' / 'evidence' / 'swappa_bootstrap_analysis_20261108.json'


def key_of(name):
    return name.replace(' ', '_').replace('/', '_').replace(',', '').replace('.', '')


def span(mask):
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)
    return [float(TH[idx[0]]), float(TH[idx[-1]]), bool((np.diff(idx) > 1).any())]


def main():
    args = sys.argv[1:]
    bdir = Path(args[args.index('--boot') + 1]) if '--boot' in args else ROOT / 'output' / 'swappa_core_20261108' / 'boot'
    out = dict(date='2026-11-08', boot_dir=str(bdir), results={})
    s = pd.read_parquet(PILOT_OUT / 'raw_reference_panel.parquet').reset_index(drop=True)
    dates = np.sort(s.date_str.unique())
    T = len(dates)
    counts = s.groupby('Seller').sale.sum()
    sellers = counts[counts >= MIN_SALES].sort_values(ascending=False).index.tolist()
    for spec in ['raw_d0', 'raw_d6']:
        files = sorted(glob.glob(str(bdir / f'{spec}_task*.npz')))
        if not files:
            continue
        zs = [np.load(f, allow_pickle=True) for f in files]
        coefs = np.concatenate([z['coefs'] for z in zs])
        B = len(coefs)
        # full-sample pieces
        kw = SPECS[spec]
        ckw = {k: v for k, v in kw.items() if k != 'product_week'}
        X, names, mean, sd, beta = check_design_matches_saved(s, spec)
        nd = len(names) + 1
        Hinv, q = penalized_hessian_inverse(X, beta, nd)
        V, _ = twoway_cluster_cov(X, s, beta, nd, Hinv, q)
        se_coef = np.sqrt(np.diag(V))[1:nd] / sd
        rec = dict(replications=B, tasks=len(files), price_coefficients={}, sellers={})
        for j, n in enumerate(names[:4]):
            rec['price_coefficients'][n] = dict(full_sample=float(beta[1 + j] / sd[j]), analytic_se_twoway=float(se_coef[j]),
                                                boot_mean=float(coefs[:, j].mean()), boot_sd=float(coefs[:, j].std(ddof=1)),
                                                ratio_boot_over_analytic=float(coefs[:, j].std(ddof=1) / se_coef[j]))
        # the above-rival-minimum slope is the sum of the two collinear terms; report it too
        ssum = coefs[:, 0] + coefs[:, 1]
        rec['price_coefficients']['lp_ref_plus_gap_up'] = dict(full_sample=float((beta[1] / sd[0]) + (beta[2] / sd[1])),
                                                              analytic_se_twoway=float(np.sqrt(V[1, 1] / sd[0] ** 2 + V[2, 2] / sd[1] ** 2 + 2 * V[1, 2] / (sd[0] * sd[1]))),
                                                              boot_mean=float(ssum.mean()), boot_sd=float(ssum.std(ddof=1)))
        posidx_all = pd.Categorical(s.date_str, categories=dates).codes
        for name in sellers:
            k = key_of(name)
            if f'm__{k}' not in zs[0].files:
                continue
            ms = np.concatenate([z[f'm__{k}'] for z in zs])     # B x n x G
            ses = np.concatenate([z[f'se__{k}'] for z in zs])
            okb = np.isfinite(ms).all(axis=(1, 2))
            ms, ses = ms[okb], ses[okb]
            ix = s.Seller.eq(name).to_numpy()
            g = s.loc[ix].reset_index(drop=True)
            A, Bm, GA, GB = seller_series(g, X[ix].tocsr(), beta, nd, ckw, sd, posidx_all[ix], T)
            m_hat, se_hat = moments_and_se(A, Bm, GA, GB, V, T)      # n x G
            t_hat = m_hat / se_hat
            n = len(K8)
            crit_b = norm.ppf(1 - .05 / n)
            fit = (m_hat <= 0).all(0)
            bonf = (t_hat <= crit_b).all(0)
            # bootstrap dispersion vs analytic se
            sd_boot = ms.std(0, ddof=1)                            # n x G
            ratio = sd_boot / se_hat
            # percentile-t LF critical value per theta and the bootstrap set
            stat_star = ((ms - m_hat[None]) / ses).max(1)         # B x G
            c_star = np.quantile(stat_star, .95, axis=0)           # G
            boot_set = (t_hat.max(0) <= c_star)
            # normal LF critical values for comparison (full-sample correlation)
            # (recompute the joint covariance for the correlation; cheap)
            from swappa_common_20261108 import bartlett_lrcov
            Haa, Hab, Hbb = bartlett_lrcov(A.T, A.T), bartlett_lrcov(A.T, Bm.T), bartlett_lrcov(Bm.T, Bm.T)
            Daa, Dab, Dbb = GA @ V @ GA.T, GA @ V @ GB.T, GB @ V @ GB.T
            c_norm = np.zeros(len(TH))
            for ti, th in enumerate(TH):
                Sig = (Haa - th * (Hab + Hab.T) + th ** 2 * Hbb) / T + Daa - th * (Dab + Dab.T) + th ** 2 * Dbb
                sdv = np.sqrt(np.maximum(np.diag(Sig), 1e-300))
                c_norm[ti] = lf_critical(Sig / np.outer(sdv, sdv), n)
            maxt_set = (t_hat.max(0) <= c_norm)
            # size check at the null point closest to binding
            if fit.any():
                cand = np.flatnonzero(fit)
                i0 = cand[np.argmax(t_hat.max(0)[cand])]
                tz = ms[:, :, i0] / ses[:, :, i0]
                rej_bonf = float(np.mean(tz.max(1) > crit_b))
                rej_norm = float(np.mean(tz.max(1) > c_norm[i0]))
                rej_boot = float(np.mean(tz.max(1) > c_star[i0]))
                mid = cand[len(cand) // 2]
                cover_mid = float(np.mean((ms[:, :, mid] / ses[:, :, mid] <= crit_b).all(1)))
                null_point = dict(theta=float(TH[i0]), max_t_full=float(t_hat.max(0)[i0]),
                                  reject_bonferroni=rej_bonf, reject_normal_maxt=rej_norm, reject_percentile_t=rej_boot,
                                  bonferroni_covers_fit_mid=cover_mid)
                sel = np.flatnonzero(fit)
                ratio_fit = np.round(ratio[:, sel].mean(1), 3).tolist()
            else:
                null_point, ratio_fit = None, None
            sold = g.sale.eq(1).to_numpy()
            mp, mr = float(g.Price[sold].mean()), float(g.ref[sold].mean())

            def lern(iv):
                return None if iv is None else [round(100 * (1 - iv[1] * mr / mp), 2), round(100 * (1 - iv[0] * mr / mp), 2)]
            eb = span(bonf)
            ratio_end = None
            if eb:
                ii = [int(np.argmin(np.abs(TH - eb[0]))), int(np.argmin(np.abs(TH - eb[1])))]
                ratio_end = [np.round(ratio[:, i], 3).tolist() for i in ii]
            rec['sellers'][name] = dict(
                replications_used=int(okb.sum()),
                sets_theta=dict(fit=span(fit), bonferroni=eb, normal_maxt=span(maxt_set), percentile_t_boot=span(boot_set)),
                lerner=dict(fit=lern(span(fit)), bonferroni=lern(eb), normal_maxt=lern(span(maxt_set)), percentile_t_boot=lern(span(boot_set))),
                critical_values=dict(bonferroni=float(crit_b), normal_maxt_range=[float(c_norm.min()), float(c_norm.max())],
                                     percentile_t_range=[float(c_star.min()), float(c_star.max())],
                                     percentile_t_at_fit=(float(np.mean(c_star[fit])) if fit.any() else None)),
                boot_sd_over_analytic_se=dict(in_fit_set_by_shift=ratio_fit, at_bonferroni_endpoints_by_shift=ratio_end, shifts=K8),
                null_point_size_check=null_point)
            print(spec, name[:22].ljust(22), 'B', int(okb.sum()), '| lerner bonf', lern(eb), 'maxt', lern(span(maxt_set)), 'boot', lern(span(boot_set)),
                  '| ratio', ratio_fit, '| null', null_point and {k: round(v, 3) for k, v in null_point.items()}, flush=True)
        out['results'][spec] = rec
        print(spec, 'price coefficients', json.dumps(rec['price_coefficients'], indent=0), flush=True)
    EVID.write_text(json.dumps(out, indent=1, default=float))
    print('BOOTSTRAP_ANALYSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
