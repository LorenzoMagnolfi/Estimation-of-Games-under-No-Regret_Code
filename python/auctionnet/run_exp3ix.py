"""EXP3-IX (certified bandit family B) for Design A, one focal role: run, inference, coverage, figures.

Steps (each skipped if its output exists): (1) sim_run.py exp3ix to 1e7 (seed 1) with the main environment; (2) sim_infer.py
on it, regimes P and O, full grid, rules exact/empirical/none/tailB/momentB; (3) sim_coverage.py exp3ix, 20 seeds to 1e7,
coarse grid (all applicable rules); (4) sim_plots.py on the main run. Outputs go to results/designA/exp3ix_v1/role<r>/ and
never overwrite earlier files. A manifest records the code commit. The plain EXP3 runs of September are untouched.
Usage: python run_exp3ix.py <auctionnet_root> <role_index 0..5>
"""
import json
import os
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
out = os.path.join(da, "exp3ix_v1", f"role{role}")
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
    return r.stdout


runf = os.path.join(out, "run_exp3ix_1e7.npz")
if not os.path.exists(runf):
    text = run([py, os.path.join(here, "sim", "sim_run.py"), env_main, runf, "exp3ix", "1e7", "1"], "run exp3ix 1e7")
    with open(os.path.join(out, "run_exp3ix_1e7_stdout.txt"), "w") as fh:
        fh.write(text)
pref = os.path.join(out, "exp3ix_1e7")
if not os.path.exists(pref + "_regimeO.csv"):
    text = run([py, os.path.join(here, "sim", "sim_infer.py"), env_main, runf, pref, "PO", "full", "all", "exact,empirical,none,tailB,momentB"], "infer exp3ix 1e7")
    with open(pref + "_stdout.txt", "w") as fh:
        fh.write(text)
cov = os.path.join(out, "coverage_exp3ix_1e7")
if not os.path.exists(os.path.join(cov, "coverage_exp3ix_P.csv")):
    run([py, os.path.join(here, "sim", "sim_coverage.py"), env_main, cov, "exp3ix", "1e7", "20"], "coverage exp3ix 20 seeds")
run([py, os.path.join(here, "sim", "sim_plots.py"), env_main, pref, os.path.join(out, "figs", f"role{role}_exp3ix_1e7")], "plots exp3ix 1e7")
commit, dirty = code_commit()
with open(os.path.join(out, "MANIFEST.json"), "w") as fh:
    json.dump(dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), role=role, code_commit=commit, code_dirty=dirty,
                   learner="exp3ix (family B: IX estimator, gamma = eta/2, eta = sqrt(ln d/(d s)), uniform prior)",
                   rules="tailB, momentB (V05 eqs. s4Btail, s4Bmoment); Regime O with conservative count and rival comparison term",
                   env=os.path.basename(env_main), log=log), fh, indent=1)
print("EXP3IX_V1_DONE", flush=True)
