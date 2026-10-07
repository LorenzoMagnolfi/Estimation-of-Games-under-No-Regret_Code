"""Per-period regret footprint for all 48 roles from a period env npz (no parquet needed).

For each role: min-regret effective value lambda* over fixed-multiplier deviations, delta* (relative regret at lambda*),
actual median multiplier, multiplier path signature (constant / capped / PID-like / other), budget exhaustion step,
win rate, and per-quintile regret shares. Same conventions as regret_lambda.py and classify_paths.py.
Usage: python footprint_period.py <env_npz> <out_csv>
"""
import sys

import numpy as np
import pandas as pd

E = np.array([1.0, 0.80, 0.48]); RES = 0.01
env = np.load(sys.argv[1]); out = sys.argv[2]
pv, bd, rv, step = env["pvals"], env["bids"], env["rivals"], env["step"]
roles, budget, cpa, cat, period = env["roles"], env["budget"], env["cpa"], env["category"], int(env["period"])
alphas = np.exp(np.linspace(np.log(20), np.log(400), 31))
lam_grid = np.exp(np.linspace(np.log(20), np.log(400), 61))


def outcome(b, r):
    r1, r2, r3 = r[:, 0], r[:, 1], r[:, 2]
    s1 = b > r1; s2 = (~s1) & (b > r2); s3 = (~s1) & (~s2) & (b > r3)
    price = np.maximum(np.where(s1, r1, np.where(s2, r2, np.where(s3, r3, 0.0))), RES)
    e = np.where(s1, E[0], np.where(s2, E[1], np.where(s3, E[2], 0.0)))
    return e, price


rows = []
for ii, i in enumerate(roles):
    v = pv[ii].astype(np.float64); b = bd[ii].astype(np.float64); r = rv[ii].astype(np.float64)
    act = (b > 0) & (v > 0)
    v, b, r, st = v[act], b[act], r[act], step[act]
    if len(v) < 1000:
        continue
    e_a, p_a = outcome(b, r)
    ev_a, ec_a = (e_a * v).mean(), (e_a * p_a).mean()
    ev_m = np.zeros(len(alphas)); ec_m = np.zeros(len(alphas))
    for j, a in enumerate(alphas):
        e, p = outcome(a * v, r)
        ev_m[j], ec_m[j] = (e * v).mean(), (e * p).mean()
    u_a = lam_grid * ev_a - ec_a
    u_m = lam_grid[:, None] * ev_m[None, :] - ec_m[None, :]
    best = u_m.max(axis=1); delta = (best - u_a) / np.maximum(best, 1e-12); ok = best > 0
    js = int(np.argmin(np.where(ok, delta, np.inf)))
    lam_star, d_star = lam_grid[js], float(delta[js])
    # path signature
    ap = pd.Series(b / v).groupby(st).median()
    a = ap.values; lc = np.diff(np.log(a[a > 0]))
    zero = np.mean(np.abs(lc) < 1e-6) if len(lc) else 1.0
    pid = np.mean((np.abs(lc - np.log(1.2)) < 1e-3) | (np.abs(lc - np.log(0.7)) < 1e-3)) if len(lc) else 0.0
    capped = np.mean(np.abs(a / cpa[ii] - 1.5) < 1e-6)
    if zero > 0.98: cls = "constant"
    elif pid + zero > 0.9 and pid > 0.05: cls = "PID-like"
    elif capped > 0.5: cls = "capped 1.5xCPA"
    else: cls = "other (LP/RL)"
    active_steps = np.unique(st)
    exhaust = int(active_steps.max()) + 1 if active_steps.max() < 47 else 48
    win = float((e_a > 0).mean())
    rows.append(dict(period=period, role=int(i), category=int(cat[ii]), budget=float(budget[ii]), cpa=float(cpa[ii]), n_active=len(v),
                     exhaust_step=exhaust, win_rate=win, alpha_med=float(np.median(b / v)), alpha_over_cpa=float(np.median(b / v) / cpa[ii]),
                     lam_star=lam_star, lam_star_over_cpa=lam_star / cpa[ii], delta_star=d_star, cls=cls,
                     path_cv=float(a.std() / a.mean()) if a.mean() > 0 else np.nan, mean_pvalue=float(v.mean()),
                     r1_med=float(np.median(r[:, 0])), r3_med=float(np.median(r[:, 2]))))
df = pd.DataFrame(rows)
df.to_csv(out, index=False)
print(f"period {period}: {len(df)} roles; delta*<0.1 share {(df.delta_star < 0.1).mean():.2f}; classes {df.cls.value_counts().to_dict()}; "
      f"median lambda*/CPA {df.lam_star_over_cpa.median():.2f}; exhaust<48: {(df.exhaust_step < 48).sum()}")
