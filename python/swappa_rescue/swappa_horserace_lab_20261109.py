"""Laboratory with known truth for the horse race against the equilibrium benchmark (9 November 2026).

The environment is a seller's actual Swappa window: its SKU-day cells with their states, its observed prices as the
relative-price profile (lattice level 40), demand raw_d0 taken as the truth, cost c = theta* r (Decluttr quote). The
seller chooses a price level each day from A9 = {0, +-1, +-2.5, +-5, +-10 percent} (lattice offsets) with the learner
of 08_CLOSING_PROOFS.md: the regularized leader with entropy (Hedge), c = 1, constant scale kappa_L equal to the
root-mean-square cell span at theta*, step eta_s = sqrt(ln 9 / s) per cell, full feedback once a day, one level drawn
per day for all of the seller's cells. Presample experience is H cells, drawn as whole days from the window's day
distribution.

Data-generating processes (`--dgp`, comma-separated; default all):
  S        stationary: presample and window days i.i.d. from the window's days, H measured (Assumption S holds)
  S16      as S with H/16;   S784   as S with H = 784 (Certified Cell's experience);   S0   fresh learner (H = 0)
  C        window days in calendar order (the actual sequence), presample i.i.d.
  OFFp/OFFm  as S with theta* moved so that the payoff-maximizing lattice level is +7 (resp. -7), between grid points
  SHb      continuing learner after a change: presample under theta_pre = theta*(1 + b), window under theta*,
           b in {-0.05, -0.03, -0.02, -0.01, +0.01, +0.02, +0.03, +0.05} (labels SHm5 ... SHp5), and b = -+0.03 with
           H/16 (SHm3h16, SHp3h16: a younger learner that partly re-adapts in the window); Assumption S fails at the boundary

Estimators on each simulated window (same code as on the data, `swappa_horserace_core_20261109.py`): the equilibrium
first-order condition (point and 95 percent interval), the steady-state inequalities, and the test of record with the
measured H. Variance: lag-0 sample variance plus the real-data demand delta-method term (demand is known in the lab,
so this term only widens every interval; it is kept because a practitioner would report it); a second set of
coverage numbers drops it. Outputs per DGP: coverage of theta*, the equilibrium point's error in Lerner points, set
widths, and the realized window regret against the eight level shifts at theta* as a share of expected profit.

Usage: python swappa_horserace_lab_20261109.py [--seller GadgetPickup] [--reps 500] [--dgp S,C,...] [--json OUT]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swappa_horserace_core_20261109 import (TH, MID, A9, J8, DELTA_H, ALPHA, day_tables, span_stats,  # noqa: E402
                                            allowance_record, demand_terms)

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / 'output' / 'swappa_horserace_20261109'
LONGEVITY = ROOT / 'output' / 'swappa_core_20261108' / 'branch_longevity.json'
SHIFTS = {'SHm5': (-.05, 1), 'SHm3': (-.03, 1), 'SHm2': (-.02, 1), 'SHm1': (-.01, 1), 'SHp1': (.01, 1), 'SHp2': (.02, 1),
          'SHp3': (.03, 1), 'SHp5': (.05, 1), 'SHm3h16': (-.03, 16), 'SHp3h16': (.03, 16)}


def lattice_opt(Rd, CRd, th):
    return int(np.argmax((Rd - th * CRd).mean(0))) - MID


def theta_for_opt(Rd, CRd, target):
    """Grid theta whose payoff-maximizing lattice level (full lattice, window mean) is closest to `target`."""
    best = min(TH, key=lambda th: (abs(lattice_opt(Rd, CRd, th) - target), abs(th - .8)))
    return float(best)


def simulate(rng, Rd, CRd, cpd, theta_true, theta_pre, kappa_L, H, reps, order):
    """Hedge levels for `reps` windows. Returns day indices (reps x T) and lattice levels (reps x T)."""
    T = Rd.shape[0]
    cols = MID + A9
    P_win = Rd[:, cols] - theta_true * CRd[:, cols]                  # T x 9
    P_pre = Rd[:, cols] - theta_pre * CRd[:, cols]
    Bbar = cpd.mean()
    n_pre_days = int(round(H / Bbar))
    cum = np.zeros((reps, len(A9)))
    n = np.zeros(reps)
    if n_pre_days > 0:
        pre = rng.integers(0, T, size=(reps, n_pre_days))
        cum += P_pre[pre].sum(1)
        n += cpd[pre].sum(1)
    days = np.tile(np.arange(T), (reps, 1)) if order == 'calendar' else rng.integers(0, T, size=(reps, T))
    lev = np.zeros((reps, T), int)
    eta0 = np.sqrt(np.log(len(A9)))
    for t in range(T):
        eta = eta0 / np.sqrt(n + 1) / kappa_L
        logit = eta[:, None] * (cum - cum.max(1, keepdims=True))
        p = np.exp(logit)
        p /= p.sum(1, keepdims=True)
        u = rng.random(reps)
        a = (p.cumsum(1) < u[:, None]).sum(1)
        a = np.minimum(a, len(A9) - 1)
        lev[:, t] = MID + A9[a]
        d = days[:, t]
        cum += P_win[d]
        n += cpd[d]
    return days, lev


def evaluate(Rd, CRd, days, lev, theta_true, eps, dem, alpha=ALPHA):
    """Vectorized estimators over replications. Returns per-replication records."""
    reps, T = lev.shape
    g = lambda M, off: M[days, lev + off]  # noqa: E731
    R0, C0 = g(Rd, 0), g(CRd, 0)
    A = np.stack([g(Rd, j) - R0 for j in J8], 1)                       # reps x 8 x T
    B = np.stack([g(CRd, j) - C0 for j in J8], 1)
    dR, dC = g(Rd, 1) - g(Rd, -1), g(CRd, 1) - g(CRd, -1)              # reps x T
    ratio = C0.sum(1) / R0.sum(1)                                       # expected revenue-weighted sold ratio
    th = TH[None, None, :]
    # equilibrium
    thN = dR.sum(1) / dC.sum(1)
    mD = dR.mean(1)[:, None] - TH[None] * dC.mean(1)[:, None]          # reps x nth
    vRR, vCC = dR.var(1, ddof=1), dC.var(1, ddof=1)
    vRC = ((dR - dR.mean(1, keepdims=True)) * (dC - dC.mean(1, keepdims=True))).sum(1) / (T - 1)
    vD = (vRR[:, None] - 2 * TH[None] * vRC[:, None] + TH[None] ** 2 * vCC[:, None]) / T
    gD = dem['gdR'][None] - TH[:, None] * dem['gdC'][None]             # nth x k
    vdem_D = np.einsum('tk,kl,tl->t', gD, dem['V'], gD)
    z2 = norm.ppf(1 - alpha / 2)
    eq_ok = np.abs(mD) <= z2 * np.sqrt(vD + vdem_D[None])
    eq_ok_nod = np.abs(mD) <= z2 * np.sqrt(vD)
    # inequalities
    mA, mB = A.mean(2), B.mean(2)
    Ac, Bc = A - mA[..., None], B - mB[..., None]
    vAA, vBB, vAB = (Ac ** 2).sum(2) / (T - 1), (Bc ** 2).sum(2) / (T - 1), (Ac * Bc).sum(2) / (T - 1)
    m = mA[..., None] - th * mB[..., None]                              # reps x 8 x nth
    v = (vAA[..., None] - 2 * th * vAB[..., None] + th ** 2 * vBB[..., None]) / T
    G = dem['GA'][:, None, :] - TH[None, :, None] * dem['GB'][:, None, :]   # 8 x nth x k
    vdem = np.einsum('itk,kl,itl->it', G, dem['V'], G)[None]           # 1 x 8 x nth
    crit_st = norm.ppf(1 - alpha / 8)
    crit_rc = norm.ppf(1 - (alpha - DELTA_H) / 8)
    st_ok = ((m / np.sqrt(v + vdem)) <= crit_st).all(1)
    rc_ok = (((m - eps[None, None]) / np.sqrt(v + vdem)) <= crit_rc).all(1)
    st_ok_nod = ((m / np.sqrt(v)) <= crit_st).all(1)
    rc_ok_nod = (((m - eps[None, None]) / np.sqrt(v)) <= crit_rc).all(1)
    gi = int(np.argmin(np.abs(TH - theta_true)))
    # realized window regret against the eight shifts at the truth, share of expected profit
    gains = mA - theta_true * mB
    prof = R0.mean(1) - theta_true * C0.mean(1)
    regret_share = np.maximum(gains.max(1), 0) / prof

    def width(ok):
        L = 100 * (1 - TH[None] * ratio[:, None])
        Lm = np.where(ok, L, np.nan)
        return np.nanmax(Lm, 1) - np.nanmin(Lm, 1)
    true_L = 100 * (1 - theta_true * ratio)
    eq_L = 100 * (1 - thN * ratio)
    return dict(
        cover_eq=eq_ok[:, gi], cover_steady=st_ok[:, gi], cover_record=rc_ok[:, gi],
        cover_eq_nodemand=eq_ok_nod[:, gi], cover_steady_nodemand=st_ok_nod[:, gi], cover_record_nodemand=rc_ok_nod[:, gi],
        empty_steady=~st_ok.any(1), empty_record=~rc_ok.any(1), empty_eq=~eq_ok.any(1),
        eq_err=eq_L - true_L, true_L=true_L, width_eq=width(eq_ok), width_steady=width(st_ok), width_record=width(rc_ok),
        regret_share=regret_share, mean_level_pct=100 * (np.exp(0.005 * (lev - MID)) - 1).mean(1))


def summarize(rec):
    s = {}
    for k, v in rec.items():
        v = np.asarray(v, float)
        if k.startswith(('cover', 'empty')):
            s[k] = float(np.nanmean(v))
        else:
            s[k] = dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v)), p05=float(np.nanpercentile(v, 5)),
                        p50=float(np.nanpercentile(v, 50)), p95=float(np.nanpercentile(v, 95)))
    return s


def main():
    args = sys.argv[1:]
    seller = args[args.index('--seller') + 1] if '--seller' in args else 'GadgetPickup'
    reps = int(args[args.index('--reps') + 1]) if '--reps' in args else 500
    dgps = args[args.index('--dgp') + 1].split(',') if '--dgp' in args else ['S', 'S16', 'S784', 'S0', 'C', 'OFFp', 'OFFm', *SHIFTS]
    seed = int(args[args.index('--seed') + 1]) if '--seed' in args else 20261109
    tdir = TABLES
    out_json = Path(args[args.index('--json') + 1]) if '--json' in args else tdir / f'horserace_lab_{seller.replace(" ", "_")}.json'
    lon = json.loads(LONGEVITY.read_text())
    z = dict(np.load(tdir / f'tables_raw_d0_{seller.replace(" ", "_")}.npz', allow_pickle=False))
    dem_all = dict(np.load(tdir / 'demand_raw_d0.npz', allow_pickle=False))
    dem = demand_terms(z, 'dec', dem_all['V'])
    Rd, CRd = day_tables(z, 'dec')
    T = Rd.shape[0]
    cpd = z['cells_per_day'].astype(float)
    H_meas = float(lon[seller]['implied_presample_listing_days']) * float(z['nC']) / float(z['n_rows'])
    sp_ = span_stats(z, 'dec')
    dR, dC = Rd[:, MID + 1] - Rd[:, MID - 1], CRd[:, MID + 1] - CRd[:, MID - 1]
    th_data = float(TH[np.argmin(np.abs(TH - dR.sum() / dC.sum()))])
    truths = {'base': th_data, 'OFFp': theta_for_opt(Rd, CRd, 7), 'OFFm': theta_for_opt(Rd, CRd, -7)}
    print(f'{seller}: T={T}, H={H_meas:.0f}, theta_data={th_data}, lattice optimum at theta_data {lattice_opt(Rd, CRd, th_data)}, '
          f'OFFp theta {truths["OFFp"]} (opt {lattice_opt(Rd, CRd, truths["OFFp"])}), OFFm theta {truths["OFFm"]} '
          f'(opt {lattice_opt(Rd, CRd, truths["OFFm"])})', flush=True)
    rng = np.random.default_rng(seed)
    res = dict(seller=seller, reps=reps, seed=seed, H_measured=H_meas, theta_data=th_data, truths=truths, dgps={})
    for dg in dgps:
        t0 = time.time()
        th_true = truths['OFFp'] if dg == 'OFFp' else truths['OFFm'] if dg == 'OFFm' else th_data
        th_pre = th_true * (1 + SHIFTS[dg][0]) if dg in SHIFTS else th_true
        H = {'S16': H_meas / 16, 'S784': 784.0, 'S0': 0.0}.get(dg, H_meas / SHIFTS[dg][1] if dg in SHIFTS else H_meas)
        order = 'calendar' if dg == 'C' else 'iid'
        gi = int(np.argmin(np.abs(TH - th_true)))
        kappa_L = float(sp_['kappa'][gi])
        days, lev = simulate(rng, Rd, CRd, cpd, th_true, th_pre, kappa_L, H, reps, order)
        eps_meas = allowance_record(sp_, cpd, H)
        rec = evaluate(Rd, CRd, days, lev, th_true, eps_meas, dem)
        s = summarize(rec)
        s.update(theta_true=th_true, theta_pre=th_pre, H=H, order=order, kappa_L=kappa_L,
                 allowance_share_at_truth=float(eps_meas[gi] / sp_['profit'][gi]),
                 level_hist={int(k): float(v) for k, v in zip(*np.unique(lev - MID, return_counts=True))})
        tot = sum(s['level_hist'].values())
        s['level_hist'] = {k: v / tot for k, v in s['level_hist'].items()}
        res['dgps'][dg] = s
        print(f"{dg:5s} H={H:8.0f} cover eq {s['cover_eq']:.3f} (no dem {s['cover_eq_nodemand']:.3f}) steady {s['cover_steady']:.3f} "
              f"record {s['cover_record']:.3f} | eq err {s['eq_err']['mean']:+.2f} sd {s['eq_err']['sd']:.2f} | regret "
              f"{100 * s['regret_share']['mean']:.2f}% [{100 * s['regret_share']['p05']:.2f}, {100 * s['regret_share']['p95']:.2f}] "
              f"| widths eq {s['width_eq']['mean']:.2f} st {s['width_steady']['mean']:.2f} rc {s['width_record']['mean']:.2f} "
              f"| mean level {s['mean_level_pct']['mean']:+.2f}% | {time.time() - t0:.1f}s", flush=True)
    out_json.write_text(json.dumps(res, indent=1))
    print('HORSERACE_LAB_COMPLETE', out_json)


if __name__ == '__main__':
    main()
