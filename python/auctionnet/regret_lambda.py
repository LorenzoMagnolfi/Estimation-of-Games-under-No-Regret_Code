"""E2-lite: min-regret effective value per conversion (lambda) per advertiser, and contextual check by pValue bucket.

Deviation class: fixed absolute multipliers alpha (bid = alpha * pValue) on a grid, the same for every lambda.
Regret(lambda) = max_alpha mean u(alpha; lambda) - mean u(actual bids; lambda), u = e_slot (lambda pValue - price).
Reports lambda* = argmin of relative regret (NST delta = regret / best), delta* at lambda*, and by pValue quintile
the best alpha and the regret share, to check whether one multiplier rationalizes all contexts.
Usage: python regret_lambda.py <results_dir>
"""
import os
import sys

import numpy as np
import pandas as pd

res = sys.argv[1]
E = np.array([1.0, 0.80, 0.48])
RES = 0.01
d = pd.read_parquet(os.path.join(res, "period_ranked.parquet"),
                    columns=["advertiserNumber", "advertiserCategoryIndex", "CPAConstraint", "budget", "oppid", "pValue",
                             "bid", "rank"])
top = d[d["rank"] <= 4][["oppid", "rank", "bid", "advertiserNumber"]]
tb = top.pivot(index="oppid", columns="rank", values="bid"); tb.columns = [f"b{int(c)}" for c in tb.columns]
ta = top.pivot(index="oppid", columns="rank", values="advertiserNumber"); ta.columns = [f"a{int(c)}" for c in ta.columns]
alphas = np.exp(np.linspace(np.log(20), np.log(400), 31))
lam_grid = np.exp(np.linspace(np.log(20), np.log(400), 61))


def outcome(b, r1, r2, r3):
    s1 = b > r1
    s2 = (~s1) & (b > r2)
    s3 = (~s1) & (~s2) & (b > r3)
    price = np.maximum(np.where(s1, r1, np.where(s2, r2, np.where(s3, r3, 0.0))), RES)
    e = np.where(s1, E[0], np.where(s2, E[1], np.where(s3, E[2], 0.0)))
    return e, price


rows = []
buck_rows = []
for i in sorted(d.advertiserNumber.unique()):
    di = d[d.advertiserNumber == i].set_index("oppid").sort_index().join(tb).join(ta)
    B = di[["b1", "b2", "b3", "b4"]].values
    A = di[["a1", "a2", "a3", "a4"]].values
    Bs = -np.sort(-np.where(A == i, -1.0, B), axis=1)[:, :3]
    r1, r2, r3 = Bs[:, 0], Bs[:, 1], Bs[:, 2]
    v = di.pValue.values
    act = (di.bid.values > 0) & (v > 0)
    v, r1, r2, r3, bact = v[act], r1[act], r2[act], r3[act], di.bid.values[act]
    n = len(v)
    # outcomes of the actual bids and of each fixed multiplier (independent of lambda)
    e_a, p_a = outcome(bact, r1, r2, r3)
    ev_a, ec_a = (e_a * v).mean(), (e_a * p_a).mean()
    ev_m = np.zeros(len(alphas)); ec_m = np.zeros(len(alphas))
    ev_mb = np.zeros((5, len(alphas))); ec_mb = np.zeros((5, len(alphas)))
    q = np.quantile(v, [0.2, 0.4, 0.6, 0.8]); bucket = np.searchsorted(q, v)
    ev_ab = np.array([(e_a * v)[bucket == k].mean() for k in range(5)])
    ec_ab = np.array([(e_a * p_a)[bucket == k].mean() for k in range(5)])
    for j, a in enumerate(alphas):
        e, p = outcome(a * v, r1, r2, r3)
        ev_m[j], ec_m[j] = (e * v).mean(), (e * p).mean()
        for k in range(5):
            m = bucket == k
            ev_mb[k, j], ec_mb[k, j] = (e * v)[m].mean(), (e * p)[m].mean()
    # regret as a function of lambda (payoffs are linear in lambda)
    u_a = lam_grid * ev_a - ec_a
    u_m = lam_grid[:, None] * ev_m[None, :] - ec_m[None, :]
    best = u_m.max(axis=1)
    reg = best - u_a
    delta = reg / np.maximum(best, 1e-12)
    ok = best > 0
    j_star = int(np.argmin(np.where(ok, delta, np.inf)))
    lam_star = lam_grid[j_star]
    cpa = float(di.CPAConstraint.iloc[0])
    # contextual check at lambda*: best alpha per bucket and per-bucket delta
    ub = lam_star * ev_mb - ec_mb
    ua_b = lam_star * ev_ab - ec_ab
    best_alpha_b = alphas[ub.argmax(axis=1)]
    delta_b = (ub.max(axis=1) - ua_b) / np.maximum(ub.max(axis=1), 1e-12)
    # absolute regret contribution by bucket (each bucket has 1/5 of the active opportunities) and the best alpha
    # restricted to the pooled best alpha's neighbourhood: is the pooled best within 1% of the bucket's best?
    reg_b = np.maximum(ub.max(axis=1) - ua_b, 0.0) / 5.0
    reg_share_b = reg_b / max(reg_b.sum(), 1e-12)
    j_pool = int(u_m[j_star].argmax())
    pooled_ok_b = (ub[:, j_pool] >= ub.max(axis=1) - 0.01 * np.abs(ub.max(axis=1)) - 1e-9)
    alpha_act = np.median(bact / v)
    rows.append(dict(advertiser=i, category=int(di.advertiserCategoryIndex.iloc[0]), cpa=cpa, budget=float(di.budget.iloc[0]),
                     n_active=n, alpha_actual=alpha_act, alpha_over_cpa=alpha_act / cpa, lam_star=lam_star,
                     lam_star_over_cpa=lam_star / cpa, delta_star=float(delta[j_star]),
                     delta_at_cpa=float(delta[np.argmin(np.abs(lam_grid - cpa))]),
                     best_alpha_at_lamstar=float(alphas[u_m[j_star].argmax()]),
                     lam_interval_delta10=(float(lam_grid[ok & (delta <= 0.10)].min()) if (ok & (delta <= 0.10)).any() else np.nan,
                                           float(lam_grid[ok & (delta <= 0.10)].max()) if (ok & (delta <= 0.10)).any() else np.nan),
                     best_alpha_by_bucket=np.round(best_alpha_b, 1).tolist(), delta_by_bucket=np.round(delta_b, 3).tolist(),
                     regret_share_by_bucket=np.round(reg_share_b, 3).tolist(), pooled_alpha_ok_by_bucket=pooled_ok_b.tolist(),
                     n_buckets_pooled_ok=int(pooled_ok_b.sum())))
    buck_rows.append(dict(advertiser=i, **{f"alpha_b{k}": best_alpha_b[k] for k in range(5)}, **{f"delta_b{k}": delta_b[k] for k in range(5)},
                          **{f"share_b{k}": reg_share_b[k] for k in range(5)}, n_ok=int(pooled_ok_b.sum())))
    print(f"adv {i:2d} cat {int(di.advertiserCategoryIndex.iloc[0])} cpa {cpa:5.0f} alpha_act {alpha_act:6.1f} ({alpha_act / cpa:4.2f}xCPA) "
          f"lam* {lam_star:6.1f} ({lam_star / cpa:4.2f}xCPA) delta* {delta[j_star]:.3f} delta@CPA {rows[-1]['delta_at_cpa']:.3f} "
          f"best alpha@lam* {rows[-1]['best_alpha_at_lamstar']:6.1f} | regret share by pValue quintile {np.round(reg_share_b, 2).tolist()} pooled-alpha optimal in {int(pooled_ok_b.sum())}/5 buckets")
out = pd.DataFrame(rows)
out.to_csv(os.path.join(res, "regret_lambda.csv"), index=False)
print("\nsummary over 48 advertisers")
print(out[["alpha_over_cpa", "lam_star_over_cpa", "delta_star", "delta_at_cpa"]].describe().round(3).to_string())
print("share with delta* < 0.05:", round((out.delta_star < 0.05).mean(), 3), "; < 0.10:", round((out.delta_star < 0.10).mean(), 3),
      "; < 0.40:", round((out.delta_star < 0.40).mean(), 3))
print("share with lambda* within 10% of the actual multiplier:", round((np.abs(np.log(out.lam_star / out.alpha_actual)) < 0.1).mean(), 3))
bb = pd.DataFrame(buck_rows)
spread = bb[[f"alpha_b{k}" for k in range(5)]].max(axis=1) / bb[[f"alpha_b{k}" for k in range(5)]].min(axis=1)
print("best-alpha spread across pValue buckets (max/min): median", round(spread.median(), 2), " 90th pct", round(spread.quantile(.9), 2))
print("median regret share by pValue quintile (low to high):", bb[[f"share_b{k}" for k in range(5)]].median().round(3).to_dict())
print("advertisers for which the pooled best alpha is within 1% of the bucket optimum in all 5 buckets:", int((bb.n_ok == 5).sum()), "; in >=4:", int((bb.n_ok >= 4).sum()))
