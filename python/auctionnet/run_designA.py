"""Orchestrator for Design A (definitive run). Stages, each skipped if its outputs exist:
  1. footprints: footprint_period.py for every period env file            -> results/footprints/footprint_period-<k>.csv
  2. rules:      select_roles.py on all footprints                          -> results/rules/selection.csv (+ _roles.csv)
  3. env:        sim_env2.py for each selected role (pooled periods)        -> results/designA/env_role<r>.npz
  4. runs:       sim_run.py hedge to N_MAIN (checkpoints) and exp3 to N_BANDIT, seed 1, per role
  5. inference:  sim_infer.py (regimes P and O, full grid) per run
  6. coverage:   sim_coverage.py hedge, N_COV, N_SEEDS seeds per role (coarse grid)
  7. figures:    sim_plots.py per run
A manifest (JSON) records env file hashes, git commit of the code repo, parameters, and timestamps.
Usage: python run_designA.py <auctionnet_root> [stage_max=7]
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

root = sys.argv[1]
stage_max = int(sys.argv[2]) if len(sys.argv) > 2 else 7
here = os.path.dirname(os.path.abspath(__file__))
py = sys.executable
N_MAIN, N_BANDIT, N_COV, N_SEEDS, NV = "1e8", "1e7", "1e7", 20, 7
envd = os.path.join(root, "env"); res = os.path.join(root, "results")
fp_dir = os.path.join(res, "footprints"); rules_dir = os.path.join(res, "rules"); da = os.path.join(res, "designA")
for d in [fp_dir, rules_dir, da, os.path.join(da, "figs")]:
    os.makedirs(d, exist_ok=True)
log = []


def run(cmd, tag):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    log.append(dict(tag=tag, cmd=" ".join(cmd), seconds=round(time.time() - t0), rc=r.returncode))
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit(f"stage failed: {tag}")
    return r.stdout


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


env_files = sorted(glob.glob(os.path.join(envd, "period-*_env.npz")))
print(f"{len(env_files)} period env files")
# 1. footprints
if stage_max >= 1:
    for f in env_files:
        k = os.path.basename(f).replace("_env.npz", "")
        out = os.path.join(fp_dir, f"footprint_{k}.csv")
        if not os.path.exists(out):
            print(run([py, os.path.join(here, "footprint_period.py"), f, out], f"footprint {k}").strip().splitlines()[-1])
# 2. rules
sel_csv = os.path.join(rules_dir, "selection.csv")
if stage_max >= 2:
    print(run([py, os.path.join(here, "select_roles.py"), os.path.join(fp_dir, "footprint_period-*.csv"), sel_csv], "rules"))
import pandas as pd
sel = pd.read_csv(sel_csv)
# 3. environments
runs = []
for _, s in sel.iterrows():
    role, lam0 = int(s.role), float(s.lambda0)
    env_out = os.path.join(da, f"env_role{role}.npz")
    if stage_max >= 3 and not os.path.exists(env_out):
        print(run([py, os.path.join(here, "sim", "sim_env2.py"), env_out, str(role), f"{lam0:.4f}", str(NV)] + env_files, f"env role {role}"))
    runs.append((role, lam0, env_out))
# 4-7 per role
for role, lam0, env_out in runs:
    for learner, N in [("hedge", N_MAIN), ("exp3", N_BANDIT)]:
        run_out = os.path.join(da, f"run_role{role}_{learner}_{N}.npz"); pref = os.path.join(da, f"inf_role{role}_{learner}_{N}")
        if stage_max >= 4 and not os.path.exists(run_out):
            print(run([py, os.path.join(here, "sim", "sim_run.py"), env_out, run_out, learner, N, "1"], f"run role {role} {learner}").strip().splitlines()[-1])
        if stage_max >= 5 and not os.path.exists(pref + "_regimeO.csv"):
            run([py, os.path.join(here, "sim", "sim_infer.py"), env_out, run_out, pref, "PO"], f"infer role {role} {learner}")
            print(f"inference done: role {role} {learner}")
        if stage_max >= 7 and os.path.exists(pref + "_regimeO.csv"):
            run([py, os.path.join(here, "sim", "sim_plots.py"), env_out, pref, os.path.join(da, "figs", f"role{role}_{learner}_{N}")], f"plots role {role} {learner}")
    if stage_max >= 6:
        cov_dir = os.path.join(da, f"coverage_role{role}_hedge_{N_COV}")
        if not os.path.exists(os.path.join(cov_dir, "coverage_hedge_P.csv")):
            print(run([py, os.path.join(here, "sim", "sim_coverage.py"), env_out, cov_dir, "hedge", N_COV, str(N_SEEDS), "101"], f"coverage role {role}").strip().splitlines()[-12:])
# manifest
try:
    commit = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "python/auctionnet"], capture_output=True, text=True).stdout.strip() != ""
except Exception:
    commit, dirty = "unknown", True
manifest = dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), code_commit=commit, code_dirty=dirty, params=dict(N_MAIN=N_MAIN, N_BANDIT=N_BANDIT, N_COV=N_COV, N_SEEDS=N_SEEDS, NV=NV),
                env_files={os.path.basename(f): sha(f) for f in env_files}, selection=sel.to_dict(orient="records"), log=log)
with open(os.path.join(da, "MANIFEST.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print("manifest written")
