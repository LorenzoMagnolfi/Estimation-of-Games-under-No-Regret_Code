"""Design A on the repaired implementation (close-out of 9 November 2026), one focal role.

Re-inference of the saved Hedge runs (i.i.d. environment, September 2026, and paired environment, 3 November: the main run to
1e8 and the 20 coverage seeds to 1e7) with the audited tables (exact full pool; context-conditional for the paired environment),
and EXP3-IX with the exact reward width (a new trajectory to 1e7 and 20 new coverage seeds), all through sim_infer2.py (C1 to C3
repairs, matched rectangle, normalized joint budget, own-count tails, conditional-count bandit inputs). Nothing is re-learned for
Hedge; the EXP3-IX trajectories are new because the normalizer changed the recursion (I1).
Outputs under results/designA/recon_v2/role<r>/: main runs <tag>_regime{P,O}.csv, _O_<rule>_<N>.npy, _grid.npz, _solver_log.json;
coverage cov_<tag>/inf_<seed>_* and coverage_<tag>_{P,O}.csv; MANIFEST.json. Existing outputs are skipped (role 28's main runs
come from job 11390604), so the script is restartable.
Usage: python run_recon_v2.py <auctionnet_root> <role_index 0..5> [audit_dir=~/NoRegret_runs/auctionnet/audit_20261104]
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
AUD = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/NoRegret_runs/auctionnet/audit_20261104")
here = os.path.dirname(os.path.abspath(__file__))
py = sys.executable
da = os.path.join(root, "results", "designA")
role = int(pd.read_csv(os.path.join(root, "results", "rules", "selection.csv")).role.iloc[ridx])
out = os.path.join(da, "recon_v2", f"role{role}")
os.makedirs(out, exist_ok=True)
aud_iid, aud_pair = os.path.join(AUD, f"audit_role{role}_iid.json"), os.path.join(AUD, f"audit_role{role}_paired.json")
env_iid = os.path.join(da, f"env_role{role}.npz"); run_iid = os.path.join(da, f"run_role{role}_hedge_1e8.npz")
env_pair = os.path.join(da, "paired_v1", f"role{role}", f"env_role{role}_paired.npz")
run_pair = os.path.join(da, "paired_v1", f"role{role}", "run_hedge_1e8_paired.npz")
for f in (aud_iid, aud_pair, env_iid, run_iid, env_pair, run_pair):
    if not os.path.exists(f):
        raise SystemExit(f"missing input: {f}")
SEEDS = list(range(101, 121))
F_RULES_COV = "exact,empirical,none,tailF,momentF,momentF_matched,jointF,tailF_own"
log = []


def run(cmd, tag, quiet=False):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    log.append(dict(tag=tag, seconds=round(time.time() - t0), rc=r.returncode))
    if not quiet or r.returncode != 0:
        print(f"[{tag}] {round(time.time() - t0)} s rc {r.returncode}", flush=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:], flush=True)
        raise SystemExit(f"step failed: {tag}")
    return r.stdout


def infer(env, runf, pref, grid, tables, tag, rules=None, quiet=False):
    if os.path.exists(pref + "_regimeO.csv"):
        return
    cmd = [py, os.path.join(here, "sim", "sim_infer2.py"), env, runf, pref, "regimes=PO", f"grid={grid}", f"tables={tables}"]
    if rules:
        cmd.append(f"rules={rules}")
    text = run(cmd, tag, quiet=quiet)
    with open(pref + "_stdout.txt", "w") as fh:
        fh.write(text)


# ---- main runs (full grid) ----
infer(env_iid, run_iid, os.path.join(out, "hedge_1e8_iid"), "full", aud_iid, "infer hedge 1e8 iid")
infer(env_pair, run_pair, os.path.join(out, "hedge_1e8_paired"), "full", aud_pair, "infer hedge 1e8 paired")
run_x = os.path.join(out, "run_exp3ix_1e7_exact.npz")
v1_x = os.path.join(da, "recon_v1", f"role{role}", "run_exp3ix_1e7_exact.npz")
if not os.path.exists(run_x):
    if os.path.exists(v1_x):
        run_x = v1_x                      # role 28: the 4 November trajectory with the exact width
    else:
        text = run([py, os.path.join(here, "sim", "sim_run.py"), env_iid, run_x, "exp3ix", "1e7", "1", "iid", aud_iid], "run exp3ix 1e7 exact")
        with open(os.path.join(out, "run_exp3ix_1e7_exact_stdout.txt"), "w") as fh:
            fh.write(text)
infer(env_iid, run_x, os.path.join(out, "exp3ix_1e7_exact"), "full", aud_iid, "infer exp3ix 1e7 exact")


# ---- coverage (coarse grid, 20 seeds) ----
def coverage(tag, env, tables, run_of_seed, rules=None, new_runs=None):
    cov = os.path.join(out, f"cov_{tag}"); os.makedirs(cov, exist_ok=True)
    if os.path.exists(os.path.join(cov, f"coverage_{tag}_P.csv")):
        print(f"[coverage {tag}] exists", flush=True); return
    t0 = time.time(); P, O = [], []
    for s in SEEDS:
        runf = run_of_seed(s)
        if not os.path.exists(runf):
            if new_runs is None:
                raise SystemExit(f"missing coverage run: {runf}")
            run(new_runs(s, runf), f"run {tag} seed {s}", quiet=True)
        pref = os.path.join(cov, f"inf_{s}")
        infer(env, runf, pref, "coarse", tables, f"infer {tag} seed {s}", rules=rules, quiet=True)
        p = pd.read_csv(pref + "_regimeP.csv"); p["seed"] = s; P.append(p)
        o = pd.read_csv(pref + "_regimeO.csv"); o["seed"] = s; O.append(o)
    P = pd.concat(P); O = pd.concat(O)
    P.to_csv(os.path.join(cov, f"coverage_{tag}_P.csv"), index=False); O.to_csv(os.path.join(cov, f"coverage_{tag}_O.csv"), index=False)
    log.append(dict(tag=f"coverage {tag}", seconds=round(time.time() - t0), rc=0))
    pd.set_option("display.width", 220)
    print(f"[coverage {tag}] {round(time.time() - t0)} s; Regime P truth retention and median width by rule and N:", flush=True)
    print(P.groupby(["rule", "N"]).agg(retained=("contains_truth", "mean"), width_med=("width", "median")).round(3).to_string(), flush=True)
    O["truth_num"] = O.truth_feasible.map(lambda v: 1.0 if str(v) == "True" else (0.0 if str(v) == "False" else float("nan")))
    O["emp_num"] = O.emp_feasible.map(lambda v: 1.0 if str(v) == "True" else (0.0 if str(v) == "False" else float("nan")))
    print(f"[coverage {tag}] Regime O truth retention, empirical-mass retention, median feasible count and unresolved by rule and N:", flush=True)
    print(O.groupby(["rule", "N"]).agg(truth=("truth_num", "mean"), emp=("emp_num", "mean"), n_feas_med=("n_feasible", "median"), unresolved=("n_unresolved", "sum")).round(3).to_string(), flush=True)


coverage("hedge_iid", env_iid, aud_iid, lambda s: os.path.join(da, f"coverage_role{role}_hedge_1e7", f"run_hedge_{s}.npz"), rules=F_RULES_COV)
coverage("hedge_paired", env_pair, aud_pair, lambda s: os.path.join(da, "paired_v1", f"role{role}", "coverage_hedge_1e7_paired", f"run_hedge_{s}.npz"), rules=F_RULES_COV)
coverage("exp3ix_exact", env_iid, aud_iid, lambda s: os.path.join(out, "cov_exp3ix_exact", f"run_exp3ix_{s}.npz"),
         new_runs=lambda s, runf: [py, os.path.join(here, "sim", "sim_run.py"), env_iid, runf, "exp3ix", "1e7", str(s), "iid", aud_iid])

commit, dirty = code_commit()
with open(os.path.join(out, "MANIFEST.json"), "w") as fh:
    json.dump(dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), role=role, code_commit=commit, code_dirty=dirty,
                   inputs=dict(env_iid=os.path.relpath(env_iid, da), run_iid=os.path.relpath(run_iid, da), env_pair=os.path.relpath(env_pair, da), run_pair=os.path.relpath(run_pair, da),
                               run_exp3ix=os.path.relpath(run_x, da), audits=[os.path.basename(aud_iid), os.path.basename(aud_pair)], coverage_seeds=SEEDS),
                   notes="sim_infer2 with C1 (upward floor), C2 (certified supporting lines), C3 (unresolved statuses), matched rectangle, mode-keyed tables; Hedge trajectories saved, EXP3-IX trajectories new with the exact width",
                   log=log), fh, indent=1)
print(f"RECON_V2_ROLE_DONE role {role}", flush=True)
