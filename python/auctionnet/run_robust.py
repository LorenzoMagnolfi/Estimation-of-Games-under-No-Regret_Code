"""Robustness and comparison runs for Design A, one focal role per invocation (Slurm array over the six roles).

All variants are relative to the definitive specification of run_designA.py (12 multipliers on 0.5-1.5 x lambda0,
7 contexts at ratio rho^5, lambda0 from the pooled rule, Hedge to N_MAIN, seed 1, full-grid inference in both regimes):
  lam075, lam150 : lambda0 x 0.75 and x 1.5 (the environment's value per conversion)
  lamcat         : category-specific lambda0 (column lambda0_category of selection.csv)
  nm8, nm16      : 8 and 16 multiplier levels; the context ratio stays near 1.65 (ctx_steps 3 and 7)
  nv5, nv9       : 5 and 9 contexts on the default lattice
  wide           : 13 levels on 0.25-2.5 x lambda0, ctx_steps 3 (the grid on which the no-regret content is visible)
Also: seeds 2 and 3 of the definitive specification at N_MAIN, and the STZ-style subsampling comparison (sim_stz.py) on the
definitive run with the 20 coverage runs at N_COV as subsamples.
Outputs under results/designA/robust/role<r>/ with a MANIFEST.json per role. Every step is skipped if its output exists.
Usage: python run_robust.py <auctionnet_root> <role_index 0..5>
"""
import glob
import json
import os
import subprocess
import sys
import time

import pandas as pd

from provenance import code_commit

root = sys.argv[1]
ridx = int(sys.argv[2])
here = os.path.dirname(os.path.abspath(__file__))
py = sys.executable
N_MAIN, N_COV, NV = "1e8", "1e7", 7
envd = os.path.join(root, "env"); res = os.path.join(root, "results"); da = os.path.join(res, "designA")
sel = pd.read_csv(os.path.join(res, "rules", "selection.csv"))
s = sel.iloc[ridx]
role, lam0, lam_cat = int(s.role), float(s.lambda0), float(s.lambda0_category)
work = os.path.join(da, "robust", f"role{role}")
os.makedirs(work, exist_ok=True)
env_files = sorted(glob.glob(os.path.join(envd, "period-*_env.npz")))
log = []
skipped = []


def run(cmd, tag):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    log.append(dict(tag=tag, cmd=" ".join(cmd), seconds=round(time.time() - t0), rc=r.returncode))
    print(f"[{tag}] {round(time.time() - t0)} s rc {r.returncode}", flush=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:], flush=True)
        raise SystemExit(f"step failed: {tag}")
    return r.stdout


# variant name -> (lambda0, NV, grid spec or None)
variants = {
    "lam075": (0.75 * lam0, NV, None),
    "lam150": (1.50 * lam0, NV, None),
    "lamcat": (lam_cat, NV, None),
    "nm8": (lam0, NV, "8,0.5,1.5,3"),
    "nm16": (lam0, NV, "16,0.5,1.5,7"),
    "nv5": (lam0, 5, None),
    "nv9": (lam0, 9, None),
    "wide": (lam0, NV, "13,0.25,2.5,3"),
}
for name, (lam, nv, grid) in variants.items():
    if name == "lamcat" and abs(lam_cat - lam0) < 1e-9:   # category value equals the pooled one: the run would duplicate the base
        skipped.append(name)
        continue
    vd = os.path.join(work, name); os.makedirs(os.path.join(vd, "figs"), exist_ok=True)
    env_out = os.path.join(vd, "env.npz"); run_out = os.path.join(vd, f"run_hedge_{N_MAIN}.npz"); pref = os.path.join(vd, f"inf_hedge_{N_MAIN}")
    if not os.path.exists(env_out):
        cmd = [py, os.path.join(here, "sim", "sim_env2.py"), env_out, str(role), f"{lam:.4f}", str(nv)]
        if grid:
            cmd.append(grid)
        print(run(cmd + env_files, f"env {name}").strip().splitlines()[-1], flush=True)
    if not os.path.exists(run_out):
        print(run([py, os.path.join(here, "sim", "sim_run.py"), env_out, run_out, "hedge", N_MAIN, "1"], f"run {name}").strip().splitlines()[-1], flush=True)
    if not os.path.exists(pref + "_regimeO.csv"):
        run([py, os.path.join(here, "sim", "sim_infer.py"), env_out, run_out, pref, "PO"], f"infer {name}")
    run([py, os.path.join(here, "sim", "sim_plots.py"), env_out, pref, os.path.join(vd, "figs", f"role{role}_{name}_hedge_{N_MAIN}")], f"plots {name}")

# extra seeds of the definitive specification at N_MAIN
env_main = os.path.join(da, f"env_role{role}.npz")
for seed in (2, 3):
    vd = os.path.join(work, f"seed{seed}"); os.makedirs(vd, exist_ok=True)
    run_out = os.path.join(vd, f"run_hedge_{N_MAIN}.npz"); pref = os.path.join(vd, f"inf_hedge_{N_MAIN}")
    if not os.path.exists(run_out):
        print(run([py, os.path.join(here, "sim", "sim_run.py"), env_main, run_out, "hedge", N_MAIN, str(seed)], f"run seed{seed}").strip().splitlines()[-1], flush=True)
    if not os.path.exists(pref + "_regimeO.csv"):
        run([py, os.path.join(here, "sim", "sim_infer.py"), env_main, run_out, pref, "PO"], f"infer seed{seed}")

# STZ-style comparison on the definitive run, coverage runs as subsamples
stz_pref = os.path.join(work, "stz")
if not os.path.exists(stz_pref + "_sets.csv") and not glob.glob(stz_pref + "*.csv"):
    run([py, os.path.join(here, "sim", "sim_stz.py"), env_main, os.path.join(da, f"run_role{role}_hedge_{N_MAIN}.npz"), N_MAIN,
         os.path.join(da, f"coverage_role{role}_hedge_{N_COV}"), N_COV, stz_pref], "stz")

commit, dirty = code_commit()
with open(os.path.join(work, "MANIFEST.json"), "w") as f:
    json.dump(dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), role=role, lambda0=lam0, lambda0_category=lam_cat, code_commit=commit, code_dirty=dirty,
                   variants={k: dict(lambda0=v[0], NV=v[1], grid=v[2]) for k, v in variants.items() if k not in skipped},
                   skipped=skipped, log=log), f, indent=1)
print("ROBUST_DONE", flush=True)
