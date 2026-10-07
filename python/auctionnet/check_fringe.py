"""Check the fringe assumption of Design A on all periods: are the rivals' bids independent of the focal context?

The simulation draws the focal context (predicted conversion probability) and the rival triple independently: the rival
triple is resampled uniformly from the focal role's pooled recorded triples. In the data the two can co-move, for
instance through the time of day (the mean predicted value drifts during the day and so do the rivals' bids).
Two outputs:
  1. <out>/fringe_corr.csv, for every period and every role: Pearson correlation between the role's predicted value and
     the highest rival bid r1 (and r3), overall and after removing step (time-of-day) means; mean r1 in the top
     quintile of the role's predicted value relative to the mean r1.
  2. <out>/fringe_tables.csv, for each focal role of results/rules/selection.csv, pooled over periods as in sim_env2.py:
     the expected payoff table u(k, j) = E[e (lambda0 v_j - price)] of multiplier k at grid context j computed two ways,
     'resampled' (rival triples drawn independently of the context, as in the simulation) and 'paired' (the rival
     triples that actually co-occurred with predicted values in context bin j). The difference between the two is the
     error the independence assumption introduces in the environment.
Usage: python check_fringe.py <auctionnet_root> <out_dir>
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from provenance import code_commit

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
envd = os.path.join(root, "env"); da = os.path.join(root, "results", "designA")
files = sorted(glob.glob(os.path.join(envd, "period-*_env.npz")), key=lambda f: int(os.path.basename(f).split("-")[1].split("_")[0]))
sel = pd.read_csv(os.path.join(root, "results", "rules", "selection.csv"))
focal = [int(r) for r in sel.role]
t0 = time.time()


def demean_by(x, g):
    s = pd.Series(x).groupby(g).transform("mean").values
    return x - s


rows = []
pooled = {r: dict(v=[], R=[]) for r in focal}
for f in files:
    e = np.load(f)
    period = int(e["period"]); roles = e["roles"]; step = e["step"].astype(np.int64)
    PV, RV = e["pvals"], e["rivals"]                 # decompress once per file
    for ii, role in enumerate(roles):
        v = PV[ii].astype(np.float64); R = RV[ii].astype(np.float64)
        keep = (v > 0) & np.all(np.isfinite(R), axis=1)
        v, Rk, st = v[keep], R[keep], step[keep]
        r1, r3 = Rk[:, 0], Rk[:, 2]
        q80 = np.quantile(v, 0.8)
        rows.append(dict(period=period, role=int(role), n=int(keep.sum()),
                         corr_v_r1=float(np.corrcoef(v, r1)[0, 1]), corr_v_r3=float(np.corrcoef(v, r3)[0, 1]),
                         corr_v_r1_within_step=float(np.corrcoef(demean_by(v, st), demean_by(r1, st))[0, 1]),
                         corr_v_r3_within_step=float(np.corrcoef(demean_by(v, st), demean_by(r3, st))[0, 1]),
                         r1_topquint_rel=float(r1[v >= q80].mean() / r1.mean()), r3_topquint_rel=float(r3[v >= q80].mean() / r3.mean())))
        if int(role) in pooled:
            pooled[int(role)]["v"].append(v); pooled[int(role)]["R"].append(Rk)
    print(f"period {period}: done ({time.time() - t0:.0f}s)", flush=True)
corr = pd.DataFrame(rows)
corr.to_csv(os.path.join(out, "fringe_corr.csv"), index=False)


def outcome(b, R, E, RES):
    r1, r2, r3 = R[:, 0], R[:, 1], R[:, 2]
    s1 = b > r1; s2 = (~s1) & (b > r2); s3 = (~s1) & (~s2) & (b > r3)
    price = np.maximum(np.where(s1, r1, np.where(s2, r2, np.where(s3, r3, 0.0))), RES)
    e = np.where(s1, E[0], np.where(s2, E[1], np.where(s3, E[2], 0.0)))
    return e, price


trows = []
for role in focal:
    env = np.load(os.path.join(da, f"env_role{role}.npz"))
    alphas, vgrid, lam0, edges, E, RES = env["alphas"], env["vgrid"], float(env["lam0"]), env["edges"], env["E"], float(env["RES"])
    v = np.concatenate(pooled[role]["v"]); R = np.concatenate(pooled[role]["R"])
    j_of = np.clip(np.searchsorted(edges, np.log(v), side="right") - 1, 0, len(vgrid) - 1)
    for j, vj in enumerate(vgrid):
        inbin = j_of == j
        for k, a in enumerate(alphas):
            e_all, p_all = outcome(a * vj, R, E, RES)
            u_all = e_all * (lam0 * vj - p_all)
            trows.append(dict(role=role, context=j + 1, v=vj, mult=k + 1, mult_over_lam=a / lam0, n_bin=int(inbin.sum()),
                              u_resampled=float(u_all.mean()), u_paired=float(u_all[inbin].mean()) if inbin.any() else np.nan,
                              win_resampled=float((e_all > 0).mean()), win_paired=float((e_all[inbin] > 0).mean()) if inbin.any() else np.nan))
    print(f"role {role}: payoff tables done ({time.time() - t0:.0f}s)", flush=True)
tab = pd.DataFrame(trows)
tab.to_csv(os.path.join(out, "fringe_tables.csv"), index=False)
commit, dirty = code_commit()
with open(os.path.join(out, "MANIFEST.json"), "w") as f:
    json.dump(dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), code_commit=commit, code_dirty=dirty,
                   env_files=[os.path.basename(x) for x in files], focal_roles=focal, seconds=round(time.time() - t0)), f, indent=1)
print("FRINGE_DONE", f"({time.time() - t0:.0f}s)")
