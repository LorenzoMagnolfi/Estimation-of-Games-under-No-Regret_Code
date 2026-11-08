"""Batch-markdown event study on the Swappa risk-set panel (8 November 2026).

Purpose: validate the fitted logit's own-price response with within-seller time variation,
a different source from the cross-seller variation that identifies the logit. Design D4 of the
6 November demand note, implemented at the daily-hazard level:

- Events. For a seller with at least 300 sales and a date d, the continuing listings are its
  listing-days at d whose previous observation is d-1. A batch markdown is a date where at
  least MIN_CUT continuing listings are cut and they are at least MIN_SHARE of the continuing
  listings, with a median cut of at least MIN_SIZE (log points). Batch markups are defined
  symmetrically. Events are defined from prices alone, before any sales outcome is read.
- Treated units: the listings cut (raised) on d, followed day by day while they stay active
  (a listing that leaves without a terminal sale is simply no longer at risk).
- Controls: other sellers' listing-days in the same SKUs (model x storage x condition) as the
  treated listings, over the same event window; and, within the seller, its continuing
  listings on d that were not repriced (when the batch is partial).
- Outcome: the daily terminal-sale hazard of the at-risk listings, by event day from -7 to +6.
- Estimand: difference-in-differences of log hazards, post (days 0..2 and 0..6) versus pre
  (days -7..-2, and separately -1), treated versus control, divided by the treated listings'
  mean log price change. This is an arc elasticity. The same event windows are scored with
  the fitted logit (saved fit raw_d0 and raw_d6) so the model's implied effect for the same
  listings and the same price changes can be compared with the data.
- Inference: event-level bootstrap (resample events with replacement, 2,000 draws).

Caveat (stated in the output): markdown timing may respond to the seller's private
information about demand; the pre-event hazard path is reported so that a dip before the cut
can be seen. This is a validation exercise, not an instrument.

Usage: python swappa_markdown_events_20261108.py [--min-cut 10] [--min-share 0.25]
Outputs: output/swappa_core_20261108/markdown_events.json
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
from scipy.special import expit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swappa_common_20261108 import ROOT, PILOT_OUT, SPECS, MIN_SALES, continuous, check_design_matches_saved  # noqa: E402

OUT = ROOT / 'output' / 'swappa_core_20261108'
PRE, POST3, POST7, POST36 = list(range(-7, -1)), [0, 1, 2], list(range(0, 7)), [3, 4, 5, 6]
WINDOW = list(range(-7, 7))


def find_events(s, sellers, min_cut, min_share, min_size, direction):
    cont = s[s.fresh_previous & s.Seller.isin(sellers)].copy()
    cont['dlp'] = np.log(cont.Price / cont.previous_price)
    cont['moved'] = cont.dlp < -1e-9 if direction == 'cut' else cont.dlp > 1e-9
    agg = cont.groupby(['Seller', 'date']).agg(n_cont=('dlp', 'size'), n_moved=('moved', 'sum'),
                                               med_move=('dlp', lambda x: float(np.median(np.abs(x[x != 0]))) if (x != 0).any() else 0.0))
    ev = agg[(agg.n_moved >= min_cut) & (agg.n_moved / agg.n_cont >= min_share) & (agg.med_move >= min_size)].reset_index()
    return ev, cont


def hazard_table(s, seller, skus, excluded_rival_codes, d):
    """Daily stock hazards by event day for three groups of listing-days in the window:
    treated = the seller's listing-days in the event SKUs (its stock in those products, pre and
    post; a listing active at d cannot have sold before d, so the pre-period must be the stock,
    not the treated listings themselves); rival = other sellers' listing-days in the event SKUs,
    excluding rival listings repriced by more than one percent on d; own_untreated = the
    seller's listing-days in its other SKUs."""
    rel = (s.date - d).dt.days
    win = (rel >= -7) & (rel <= 6)
    in_sku = s.sku.isin(skus)
    own = s.Seller.eq(seller)
    groups = {'treated': win & own & in_sku,
              'rival': win & ~own & in_sku & ~s.Code.isin(excluded_rival_codes),
              'own_untreated': win & own & ~in_sku}
    out = {}
    for label, mask in groups.items():
        g = s.loc[mask]
        if len(g) == 0:
            out[label] = None
            continue
        tab = g.assign(rel=rel[mask]).groupby('rel').sale.agg(['sum', 'size'])
        out[label] = {int(r): [float(tab.loc[r, 'sum']), int(tab.loc[r, 'size'])] for r in tab.index}
    return out


def pooled_rate(tables, label, days):
    num = sum(t[label][dd][0] for t in tables if t[label] for dd in days if dd in t[label])
    den = sum(t[label][dd][1] for t in tables if t[label] for dd in days if dd in t[label])
    return num / den if den else np.nan


def did(tables, label_ctrl, pre, post):
    """DiD of log hazards: [log h_T(post) - log h_T(pre)] - [log h_C(post) - log h_C(pre)]."""
    hT1, hT0 = pooled_rate(tables, 'treated', post), pooled_rate(tables, 'treated', pre)
    hC1, hC0 = pooled_rate(tables, label_ctrl, post), pooled_rate(tables, label_ctrl, pre)
    if min(hT1, hT0, hC1, hC0) <= 0 or np.isnan([hT1, hT0, hC1, hC0]).any():
        return np.nan
    return float(np.log(hT1 / hT0) - np.log(hC1 / hC0))


def main():
    args = sys.argv[1:]
    min_cut = int(args[args.index('--min-cut') + 1]) if '--min-cut' in args else 10
    min_share = float(args[args.index('--min-share') + 1]) if '--min-share' in args else 0.25
    min_size = float(args[args.index('--min-size') + 1]) if '--min-size' in args else 0.01
    t0 = time.time()
    s = pd.read_parquet(PILOT_OUT / 'raw_reference_panel.parquet').reset_index(drop=True)
    counts = s.groupby('Seller').sale.sum()
    sellers = counts[counts >= MIN_SALES].index.tolist()
    fits = {}
    for tag in ['raw_d0', 'raw_d6']:
        X, names, mean, sd, beta = check_design_matches_saved(s, tag)
        fits[tag] = dict(X=X, nd=len(names) + 1, sd=sd, beta=beta, kw={k: v for k, v in SPECS[tag].items() if k != 'product_week'})
    results = {}
    for direction in ['cut', 'raise']:
        events, cont = find_events(s, sellers, min_cut, min_share, min_size, direction)
        rec = dict(n_events=int(len(events)), events_by_seller=events.groupby('Seller').size().to_dict(), per_event=[])
        tables, dlp_all, model_d0, model_d6, w_events = [], [], [], [], []
        for _, e in events.iterrows():
            seller, d = e.Seller, e.date
            rows_d = cont[(cont.Seller == seller) & (cont.date == d)]
            treated = rows_d[rows_d.moved]
            skus = set(treated.sku)
            moved_rivals = set(cont[(cont.date == d) & (cont.Seller != seller) & (cont.dlp.abs() > 0.01)].Code)
            tab = hazard_table(s, seller, skus, moved_rivals, d)
            tables.append(tab)
            dlp = float(treated.dlp.mean())
            dlp_all.append(dlp)
            # model-implied effect: the seller's post stock (days 0..2) in the event SKUs at actual
            # prices versus prices scaled back by the event's mean cut (a level shift, peers too)
            post_rows = s.index[(s.Seller == seller) & (s.sku.isin(skus)) & (s.date >= d) & (s.date <= d + pd.Timedelta(days=2))]
            me = {}
            for tag, f in fits.items():
                g = s.loc[post_rows]
                if len(g) == 0:
                    me[tag] = np.nan
                    continue
                XX = f['X'][post_rows]
                z0 = XX @ f['beta']
                dense0, _ = continuous(g, **f['kw'])
                zz, _ = continuous(g, shift=np.exp(-dlp) - 1, **f['kw'])
                delta = (zz - dense0) / f['sd']
                q_act = expit(z0)
                q_pre = expit(z0 + delta @ f['beta'][1:f['nd']])
                me[tag] = float(np.log(q_act.sum() / q_pre.sum()))
            model_d0.append(me['raw_d0'])
            model_d6.append(me['raw_d6'])
            w_events.append(len(treated))
            n_riv = sum(v[1] for v in tab['rival'].values()) if tab['rival'] else 0
            n_own = sum(v[1] for v in tab['own_untreated'].values()) if tab['own_untreated'] else 0
            rec['per_event'].append(dict(seller=seller, date=str(d.date()), n_treated=int(len(treated)), n_continuing=int(e.n_cont),
                                         mean_dlp=round(dlp, 4), n_rival_listing_days=n_riv, n_own_other_sku_listing_days=n_own,
                                         model_log_hazard_change_d0=round(me['raw_d0'], 4), model_log_hazard_change_d6=round(me['raw_d6'], 4)))
        if not tables:
            results[direction] = rec
            continue
        dlp_all = np.array(dlp_all)
        # pooled hazards by event day
        rec['hazard_by_event_day'] = {lab: {dd: round(pooled_rate(tables, lab, [dd]), 5) for dd in WINDOW} for lab in ['treated', 'rival', 'own_untreated']}
        rec['mean_log_price_change_treated'] = float(np.average(dlp_all, weights=w_events))
        rec['model_log_hazard_change'] = dict(raw_d0=float(np.nansum(np.array(model_d0) * w_events) / np.sum(w_events)),
                                              raw_d6=float(np.nansum(np.array(model_d6) * w_events) / np.sum(w_events)))
        est = {}
        rng = np.random.default_rng(20261108)
        for ctrl in ['rival', 'own_untreated']:
            for prelab, pre in [('pre_-7_-2', PRE), ('pre_-1', [-1])]:
                for postlab, post in [('post_0_2', POST3), ('post_0_6', POST7), ('post_3_6', POST36)]:
                    point = did(tables, ctrl, pre, post)
                    boots = []
                    for _ in range(2000):
                        pick = rng.integers(0, len(tables), len(tables))
                        boots.append(did([tables[i] for i in pick], ctrl, pre, post))
                    boots = np.array(boots)
                    boots = boots[np.isfinite(boots)]
                    mean_dlp = float(np.average(dlp_all, weights=w_events))
                    est[f'{ctrl}|{prelab}|{postlab}'] = dict(did_log_hazard=None if np.isnan(point) else round(point, 4),
                                                             se_boot=round(float(boots.std()), 4) if len(boots) else None,
                                                             ci95=[round(float(np.quantile(boots, .025)), 4), round(float(np.quantile(boots, .975)), 4)] if len(boots) else None,
                                                             arc_elasticity=None if np.isnan(point) else round(point / mean_dlp, 3),
                                                             n_boot_finite=int(len(boots)))
        rec['estimates'] = est
        # per-event DiD (rival control, pre -7..-2, post 0..2) against the event's own cut size:
        # DiD_e = a + b * dlp_e; b is the elasticity, a is an effect common to every repricing
        # event whatever its size (attention, timing). Weighted by treated listings.
        per = []
        for tab, dlp, w in zip(tables, dlp_all, w_events):
            v3 = did([tab], 'rival', PRE, POST3)
            v36 = did([tab], 'rival', PRE, POST36)
            per.append((dlp, v3, v36, w))
        per = np.array(per, dtype=float)
        okp = np.isfinite(per[:, 1])
        reg = {}
        for col, lab in [(1, 'post_0_2'), (2, 'post_3_6')]:
            ok = np.isfinite(per[:, col])
            x, yv, w = per[ok, 0], per[ok, col], per[ok, 3]
            Xr = np.column_stack([np.ones(ok.sum()), x])
            W = np.sqrt(w)
            coef, *_ = np.linalg.lstsq(Xr * W[:, None], yv * W, rcond=None)
            # event bootstrap for the slope and intercept
            bs = []
            for _ in range(2000):
                pick = rng.integers(0, ok.sum(), ok.sum())
                c_, *_ = np.linalg.lstsq(Xr[pick] * W[pick, None], yv[pick] * W[pick], rcond=None)
                bs.append(c_)
            bs = np.array(bs)
            bins = {}
            for lo, hi, name in [(-1, -.05, 'cut>5%'), (-.05, -.025, 'cut 2.5-5%'), (-.025, 0, 'cut<2.5%')]:
                sel = (x > lo) & (x <= hi)
                if sel.sum():
                    bins[name] = dict(events=int(sel.sum()), mean_dlp=round(float(np.average(x[sel], weights=w[sel])), 4),
                                      mean_did=round(float(np.average(yv[sel], weights=w[sel])), 4))
            reg[lab] = dict(events=int(ok.sum()), intercept=round(float(coef[0]), 4), intercept_se=round(float(bs[:, 0].std()), 4),
                            slope_elasticity=round(float(coef[1]), 3), slope_se=round(float(bs[:, 1].std()), 3), by_cut_size=bins)
        rec['per_event_regression'] = reg
        results[direction] = rec
        print(direction, 'events', len(events), 'mean dlp', round(rec['mean_log_price_change_treated'], 4),
              'model d0', round(rec['model_log_hazard_change']['raw_d0'], 4), 'd6', round(rec['model_log_hazard_change']['raw_d6'], 4), flush=True)
        for k, v in est.items():
            print('   ', k, v['did_log_hazard'], 'se', v['se_boot'], 'elast', v['arc_elasticity'], flush=True)
    results['_meta'] = dict(min_cut=min_cut, min_share=min_share, min_size=min_size, sellers=sellers, seconds=round(time.time() - t0, 1),
                            caveat='Markdown timing may respond to private demand information; validation, not an instrument.')
    (OUT / f'markdown_events_c{min_cut}_s{min_share}.json').write_text(json.dumps(results, indent=1, default=float))
    print('EVENTS_COMPLETE', round(time.time() - t0, 1), flush=True)


if __name__ == '__main__':
    main()
