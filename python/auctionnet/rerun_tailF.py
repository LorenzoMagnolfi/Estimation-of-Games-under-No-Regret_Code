"""Rerun Design A inference with the direct-tail rule 'tailF' (sim_infer.py of 2026-11-02) on the Hedge runs of one focal role.

Runs covered: the main Hedge run to 1e8 (both regimes, full grid, all checkpoints; rules tailF, markov and none are computed
in the same files so that the comparison is at identical checkpoints) and the 20 coverage seeds to 1e7 (both regimes, coarse
grid, rule tailF only). The EXP3 runs are not in the certified bandit family and are not rerun. Outputs go to
results/designA/tailF_v1/role<r>/ and never overwrite earlier files. Every step is skipped if its output exists. A manifest
records the code commit. Regime P needs no new simulation: the rule only changes the per-context tolerance.
Usage: python rerun_tailF.py <auctionnet_root> <role_index 0..5>
"""
import glob
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
out = os.path.join(da, "tailF_v1", f"role{role}")
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


jobs = [("hedge_1e8", env_main, os.path.join(da, f"run_role{role}_hedge_1e8.npz"), "full", "tailF,markov,none")]
cov = os.path.join(da, f"coverage_role{role}_hedge_1e7")
for f in sorted(glob.glob(os.path.join(cov, "run_hedge_*.npz"))):
    s = os.path.basename(f).replace("run_hedge_", "").replace(".npz", "")
    jobs.append((f"cov{s}_hedge_1e7", env_main, f, "coarse", "tailF"))
for tag, env, runf, grid, rules in jobs:
    pref = os.path.join(out, tag)
    if not os.path.exists(pref + "_regimeO.csv"):
        text = run([py, os.path.join(here, "sim", "sim_infer.py"), env, runf, pref, "PO", grid, "all", rules], f"infer {tag}")
        with open(pref + "_stdout.txt", "w") as fh:
            fh.write(text)
run([py, os.path.join(here, "sim", "sim_plots.py"), env_main, os.path.join(out, "hedge_1e8"), os.path.join(out, "figs", f"role{role}_hedge_1e8")], "plots hedge_1e8")
commit, dirty = code_commit()
with open(os.path.join(out, "MANIFEST.json"), "w") as fh:
    json.dump(dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), role=role, code_commit=commit, code_dirty=dirty,
                   rule="tailF (V05 eq. s4Finputs; Regime O with conservative count and rival comparison term)",
                   jobs=[dict(tag=t, env=os.path.basename(e), run=os.path.relpath(r_, da), grid=g, rules=ru) for t, e, r_, g, ru in jobs], log=log), fh, indent=1)
print("TAILF_V1_DONE", flush=True)
