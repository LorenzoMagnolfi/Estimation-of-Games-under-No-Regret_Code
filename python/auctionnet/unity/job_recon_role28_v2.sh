#!/usr/bin/env bash
#SBATCH --job-name=auctionnet-recon28v2
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32g
#SBATCH --partition=econ,batch
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/auctionnet/logs/%x_%j.log
# Close-out of 9 November 2026 (Niccolò's C1 to C3, the matched rectangle, the mode guard): inference only, on the saved trajectories
# of recon_v1 (nothing is re-learned), versioned outputs under results/designA/recon_v2/role28/:
#  0. smoke: sim_infer2 on the paired smoke run (coarse grid, one checkpoint, every applicable rule, 12 tangents);
#  1. Hedge, i.i.d. environment, saved 1e8 run, exact full-pool tables: rectangle, matched rectangle, budget, ceiling tail, own-count tail;
#  2. Hedge, paired environment, saved 1e8 run, context-conditional tables: the same;
#  3. EXP3-IX, the recon_v1 trajectory normalized by the exact width: tailB, momentB, matched, jointB, tailBc, momentBc, tailB_own;
#  4. cell-by-cell comparison of the recon_v2 masks with recon_v1 (compare_recon.py).
set -euo pipefail
ROOT=/home/magnolfi.1/NoRegret_data/auctionnet
CODE=/home/magnolfi.1/NoRegret_Code/python/auctionnet
AUD=/home/magnolfi.1/NoRegret_runs/auctionnet/audit_20261104
PY=$HOME/.conda/envs/auctionnet/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
DA=$ROOT/results/designA; V1=$DA/recon_v1/role28; OUT=$DA/recon_v2/role28; mkdir -p $OUT
cd "$CODE"
test -f $AUD/audit_role28_iid.json && test -f $AUD/audit_role28_paired.json && test -f $V1/run_exp3ix_1e7_exact.npz
echo "=== step 0: smoke"
SM=$HOME/NoRegret_runs/auctionnet/paired_smoke
"$PY" sim/sim_infer2.py $SM/env_paired.npz $SM/run_paired.npz $OUT/smoke_paired regimes=PO grid=coarse checkpoints=100000 tables=$AUD/audit_role28_paired.json ntan=12 | tail -n 14
echo "=== step 1: Hedge iid"
"$PY" sim/sim_infer2.py $DA/env_role28.npz $DA/run_role28_hedge_1e8.npz $OUT/hedge_1e8_iid regimes=PO grid=full tables=$AUD/audit_role28_iid.json | grep "^P \|^O \|sim_infer2"
echo "=== step 2: Hedge paired"
"$PY" sim/sim_infer2.py $DA/paired_v1/role28/env_role28_paired.npz $DA/paired_v1/role28/run_hedge_1e8_paired.npz $OUT/hedge_1e8_paired regimes=PO grid=full tables=$AUD/audit_role28_paired.json | grep "^P \|^O \|sim_infer2"
echo "=== step 3: EXP3-IX exact width (recon_v1 trajectory)"
"$PY" sim/sim_infer2.py $DA/env_role28.npz $V1/run_exp3ix_1e7_exact.npz $OUT/exp3ix_1e7_exact regimes=PO grid=full tables=$AUD/audit_role28_iid.json | grep "^P \|^O \|sim_infer2"
echo "=== step 4: compare with recon_v1"
"$PY" compare_recon.py $V1 $OUT $OUT/diff_v1_v2.csv
cat > $OUT/MANIFEST.txt <<EOF
recon_v2 role 28, $(date): code $(cat $HOME/NoRegret_Code/python/auctionnet/STAGED_COMMIT.txt 2>/dev/null); audit JSONs $AUD/audit_role28_{iid,paired}.json
inputs (unchanged trajectories): run_role28_hedge_1e8.npz (iid, Sept 2026), paired_v1/role28/run_hedge_1e8_paired.npz (3 Nov), recon_v1/role28/run_exp3ix_1e7_exact.npz (4 Nov)
changes: C1 upward floor (no zeroing of positive bounds), C2 analytic supporting lines with certificates, C3 unresolved statuses kept, momentF/B_matched, mode-keyed table guard
EOF
echo RECON28_V2_DONE
