"""Paired-environment robustness row for Design A (R1.3), one focal role: each auction draws one recorded opportunity and
keeps its own context and rival triple, instead of a context from the fitted lognormal and rivals drawn independently.

Steps (each skipped if its output exists): (1) sim_env2.py rebuilds the role's environment with the per-opportunity context
bins (env_role<r>_paired.npz; same lattice, lambda0 and periods as the main environment); (2) sim_run.py hedge to 1e8 in mode
'paired' (seed 1); (3) sim_infer.py regimes P and O, full grid, rules exact/empirical/none/markov/tailF/momentF (Regime O
reports the feasibility of the fitted lognormal truth and of the empirical context masses, the DGP's truth here);
(4) sim_coverage.py hedge, 20 seeds to 1e7, coarse grid, mode 'paired'; (5) sim_plots.py. Outputs in
results/designA/paired_v1/role<r>/, never overwriting earlier files; a manifest records the code commit.
Usage: python run_paired.py <auctionnet_root> <role_index 0..5>
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
sel = pd.read_csv(os.path.join(root, "results", "rules", "selection.csv"))
role, lam0 = int(sel.role.iloc[ridx]), float(sel.lambda0.iloc[ridx])
out = os.path.join(da, "paired_v1", f"role{role}")
os.makedirs(os.path.join(out, "figs"), exist_ok=True)
env_files = sorted(glob.glob(os.path.join(root, "env", "period-*_env.npz")))
NV, N_MAIN, N_COV, N_SEEDS = 7, "1e8", "1e7", 20
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


env_paired = os.path.join(out, f"env_role{role}_paired.npz")
if not os.path.exists(env_paired):
    text = run([py, os.path.join(here, "sim", "sim_env2.py"), env_paired, str(role), f"{lam0:.4f}", str(NV)] + env_files, "env paired")
    with open(os.path.join(out, "env_stdout.txt"), "w") as fh:
        fh.write(text)
runf = os.path.join(out, f"run_hedge_{N_MAIN}_paired.npz")
if not os.path.exists(runf):
    text = run([py, os.path.join(here, "sim", "sim_run.py"), env_paired, runf, "hedge", N_MAIN, "1", "paired"], f"run hedge {N_MAIN} paired")
    with open(os.path.join(out, f"run_hedge_{N_MAIN}_paired_stdout.txt"), "w") as fh:
        fh.write(text)
pref = os.path.join(out, f"hedge_{N_MAIN}_paired")
if not os.path.exists(pref + "_regimeO.csv"):
    text = run([py, os.path.join(here, "sim", "sim_infer.py"), env_paired, runf, pref, "PO", "full", "all", "exact,empirical,none,markov,tailF,momentF"], "infer paired")
    with open(pref + "_stdout.txt", "w") as fh:
        fh.write(text)
cov = os.path.join(out, f"coverage_hedge_{N_COV}_paired")
if not os.path.exists(os.path.join(cov, "coverage_hedge_P.csv")):
    run([py, os.path.join(here, "sim", "sim_coverage.py"), env_paired, cov, "hedge", N_COV, str(N_SEEDS), "101", "paired"], "coverage paired 20 seeds")
run([py, os.path.join(here, "sim", "sim_plots.py"), env_paired, pref, os.path.join(out, "figs", f"role{role}_hedge_{N_MAIN}_paired")], "plots paired")
commit, dirty = code_commit()
with open(os.path.join(out, "MANIFEST.json"), "w") as fh:
    json.dump(dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), role=role, lambda0=lam0, code_commit=commit, code_dirty=dirty,
                   mode="paired: one recorded opportunity per auction, own context bin and own rival triple (sim_run.py mode paired)",
                   rules="exact, empirical, none, markov, tailF, momentF; Regime O also reports feasibility of the empirical context masses",
                   env=os.path.basename(env_paired), env_files=[os.path.basename(f) for f in env_files], log=log), fh, indent=1)
print("PAIRED_V1_DONE", flush=True)
