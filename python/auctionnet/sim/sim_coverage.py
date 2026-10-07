"""Coverage over seeds for Design A: run the learner for several seeds and record, per checkpoint and slack rule,
whether the truth is in the Regime P interval and feasible in Regime O, plus interval widths and Regime O shares on a
coarse grid.
Usage: python sim_coverage.py <env_npz> <work_dir> <learner> <N_total> <n_seeds> [first_seed=101]
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

env, work, learner, N_total, n_seeds = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
first = int(sys.argv[6]) if len(sys.argv) > 6 else 101
here = os.path.dirname(os.path.abspath(__file__))
py = sys.executable
os.makedirs(work, exist_ok=True)
rowsP, rowsO = [], []
for s in range(first, first + n_seeds):
    run = os.path.join(work, f"run_{learner}_{s}.npz"); pref = os.path.join(work, f"inf_{learner}_{s}")
    if not os.path.exists(run):
        subprocess.run([py, os.path.join(here, "sim_run.py"), env, run, learner, N_total, str(s)], check=True, capture_output=True)
    if not os.path.exists(pref + "_regimeP.csv"):
        subprocess.run([py, os.path.join(here, "sim_infer.py"), env, run, pref, "PO", "coarse"], check=True, capture_output=True)
    P = pd.read_csv(pref + "_regimeP.csv"); P["seed"] = s; rowsP.append(P)
    O = pd.read_csv(pref + "_regimeO.csv"); O["seed"] = s; rowsO.append(O)
    print(f"seed {s} done", flush=True)
P = pd.concat(rowsP); O = pd.concat(rowsO)
P.to_csv(os.path.join(work, f"coverage_{learner}_P.csv"), index=False); O.to_csv(os.path.join(work, f"coverage_{learner}_O.csv"), index=False)
pd.set_option("display.width", 200)
print("\nRegime P: coverage of the truth and median width by N and rule")
print(P.groupby(["rule", "N"]).agg(coverage=("contains_truth", "mean"), width_med=("width", "median"), lo_med=("lam_lo", "median"), hi_med=("lam_hi", "median")).round(3).to_string())
print("\nRegime O: truth feasibility and median feasible share (coarse grid) by N and rule")
print(O.groupby(["rule", "N"]).agg(coverage=("truth_feasible", "mean"), share_med=("share", "median"), mu_lo=("mu_lo", "median"), mu_hi=("mu_hi", "median"),
                                   sig_lo=("sig_lo", "median"), sig_hi=("sig_hi", "median")).round(3).to_string())
