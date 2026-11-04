#!/usr/bin/env bash
#SBATCH --job-name=auctionnet-recon28
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32g
#SBATCH --partition=econ,batch
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/auctionnet/logs/%x_%j.log
# Reconciliation batch for role 28 (Niccolò's feedback of 4 November 2026), versioned outputs under results/designA/recon_v1/role28/:
#  0. smoke: sim_infer2 on the paired smoke run with the audited conditional tables, coarse grid, one checkpoint, every rule;
#  1. Hedge, i.i.d. environment: re-inference of the saved 1e8 run with exact full-pool tables (I3), the repaired interval sign (I5),
#     the rectangle, the normalized joint budget (jointF), the ceiling-count tail (tailF) and the own-count tail (tailF_own, Q14);
#  2. Hedge, paired environment: the same with context-conditional tables (I2);
#  3. EXP3-IX, i.i.d.: a NEW trajectory to 1e7 normalized by the exact reward width (I1), then tailB / momentB / jointB /
#     tailBc / momentBc / tailB_own; and sim_infer2 on the OLD trajectory for the record (family-B rules refused).
set -euo pipefail
ROOT=/home/magnolfi.1/NoRegret_data/auctionnet
CODE=/home/magnolfi.1/NoRegret_Code/python/auctionnet
AUD=/home/magnolfi.1/NoRegret_runs/auctionnet/audit_20261104
PY=$HOME/.conda/envs/auctionnet/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
DA=$ROOT/results/designA; OUT=$DA/recon_v1/role28; mkdir -p $OUT
cd "$CODE"
test -f $AUD/audit_role28_iid.json && test -f $AUD/audit_role28_paired.json
echo "=== step 0: smoke"
SM=$HOME/NoRegret_runs/auctionnet/paired_smoke
"$PY" sim/sim_infer2.py $SM/env_paired.npz $SM/run_paired.npz $OUT/smoke_paired regimes=PO grid=coarse checkpoints=100000 tables=$AUD/audit_role28_paired.json ntan=12 | tail -n 12
echo "=== step 1: Hedge iid, re-inference with exact tables"
"$PY" sim/sim_infer2.py $DA/env_role28.npz $DA/run_role28_hedge_1e8.npz $OUT/hedge_1e8_iid regimes=PO grid=full rules=exact,empirical,none,tailF,momentF,jointF,tailF_own tables=$AUD/audit_role28_iid.json | grep "^P \|^O \|sim_infer2"
echo "=== step 2: Hedge paired, re-inference with conditional tables"
"$PY" sim/sim_infer2.py $DA/paired_v1/role28/env_role28_paired.npz $DA/paired_v1/role28/run_hedge_1e8_paired.npz $OUT/hedge_1e8_paired regimes=PO grid=full rules=exact,empirical,none,tailF,momentF,jointF,tailF_own tables=$AUD/audit_role28_paired.json | grep "^P \|^O \|sim_infer2"
echo "=== step 3a: EXP3-IX old trajectory under sim_infer2 (certified rules refused)"
"$PY" sim/sim_infer2.py $DA/env_role28.npz $DA/exp3ix_v1/role28/run_exp3ix_1e7.npz $OUT/exp3ix_1e7_oldnorm regimes=PO grid=full tables=$AUD/audit_role28_iid.json | grep "^P \|^O \|sim_infer2\|not applicable"
echo "=== step 3b: EXP3-IX new trajectory with the exact reward width"
"$PY" sim/sim_run.py $DA/env_role28.npz $OUT/run_exp3ix_1e7_exact.npz exp3ix 1e7 1 iid $AUD/audit_role28_iid.json | tail -n 6
"$PY" sim/sim_infer2.py $DA/env_role28.npz $OUT/run_exp3ix_1e7_exact.npz $OUT/exp3ix_1e7_exact regimes=PO grid=full tables=$AUD/audit_role28_iid.json | grep "^P \|^O \|sim_infer2"
cat > $OUT/MANIFEST.txt <<EOF
recon_v1 role 28, $(date): code $(cat $HOME/NoRegret_Code/python/auctionnet/STAGED_COMMIT.txt 2>/dev/null); audit JSONs $AUD/audit_role28_{iid,paired}.json
inputs: run_role28_hedge_1e8.npz (iid, Sept 2026), paired_v1/role28/run_hedge_1e8_paired.npz (3 Nov), exp3ix_v1/role28/run_exp3ix_1e7.npz (2 Nov, normalizer K); new run_exp3ix_1e7_exact.npz (normalizer width_exact)
EOF
echo RECON28_DONE
