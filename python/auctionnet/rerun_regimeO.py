"""Rerun Regime O with the corrected tolerance (sim_infer.py of 2026-10-07) on every Design A run of one focal role.

Runs covered: the main Hedge run to 1e8 and the EXP3 run to 1e7 (environment env_role<r>.npz); the robustness variants
and seeds 2 and 3 (results/designA/robust/role<r>/...); the 20 coverage seeds to 1e7 on the coarse grid. Regime P is not
recomputed (it is unchanged). Outputs go to results/designA/regimeO_v2/role<r>/ and never overwrite the files of
2026-09-23/24. Every step is skipped if its output exists. A manifest records the code commit.
Usage: python rerun_regimeO.py <auctionnet_root> <role_index 0..5>
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd

from provenance import code_commit

root, ridx = sys.argv[1], int(sys.argv[2])
here = os.path.dirname(os.path.abspath(__file__))
py = sys.executable
da = os.path.join(root, "results", "designA")
role = int(pd.read_csv(os.path.join(root, "results", "rules", "selection.csv")).role.iloc[ridx])
out = os.path.join(da, "regimeO_v2", f"role{role}")
os.makedirs(os.path.join(out, "figs"), exist_ok=True)
env_main = os.path.join(da, f"env_role{role}.npz")
log = []


def run(cmd, tag):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    log.append(dict(tag=tag, cmd=" ".join(cmd), seconds=round(time.time() - t0), rc=r.returncode))
    print(f"[{tag}] {round(time.time() - t0)} s rc {r.returncode}", flush=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:], flush=True)
        raise SystemExit(f"step failed: {tag}")


jobs = [("hedge_1e8", env_main, os.path.join(da, f"run_role{role}_hedge_1e8.npz"), "full"),
        ("exp3_1e7", env_main, os.path.join(da, f"run_role{role}_exp3_1e7.npz"), "full")]
rob = os.path.join(da, "robust", f"role{role}")
for vdir in sorted(glob.glob(os.path.join(rob, "*", "run_hedge_1e8.npz"))):
    v = os.path.basename(os.path.dirname(vdir))
    env_v = os.path.join(rob, v, "env.npz") if os.path.exists(os.path.join(rob, v, "env.npz")) else env_main
    jobs.append((f"{v}_hedge_1e8", env_v, vdir, "full"))
cov = os.path.join(da, f"coverage_role{role}_hedge_1e7")
for f in sorted(glob.glob(os.path.join(cov, "run_hedge_*.npz"))):
    s = os.path.basename(f).replace("run_hedge_", "").replace(".npz", "")
    jobs.append((f"cov{s}_hedge_1e7", env_main, f, "coarse"))
for tag, env, runf, grid in jobs:
    pref = os.path.join(out, tag)
    if not os.path.exists(pref + "_regimeO.csv"):
        run([py, os.path.join(here, "sim", "sim_infer.py"), env, runf, pref, "O", grid], f"infer {tag}")
# figures for the two main runs: sim_plots.py reads the (unchanged) Regime P file next to the new Regime O files
for tag, learner in [("hedge_1e8", "hedge_1e8"), ("exp3_1e7", "exp3_1e7")]:
    p_old = os.path.join(da, f"inf_role{role}_{learner}_regimeP.csv")
    shutil.copyfile(p_old, os.path.join(out, f"{tag}_regimeP.csv"))
    run([py, os.path.join(here, "sim", "sim_plots.py"), env_main, os.path.join(out, tag), os.path.join(out, "figs", f"role{role}_{tag}")], f"plots {tag}")
commit, dirty = code_commit()
with open(os.path.join(out, "MANIFEST.json"), "w") as fh:
    json.dump(dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), role=role, code_commit=commit, code_dirty=dirty,
                   jobs=[dict(tag=t, env=os.path.basename(e) if os.path.dirname(e) == da else os.path.relpath(e, da), run=os.path.relpath(r_, da), grid=g)
                         for t, e, r_, g in jobs], log=log), fh, indent=1)
print("REGIMEO_V2_DONE", flush=True)
