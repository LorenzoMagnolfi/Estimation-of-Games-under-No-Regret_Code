#!/usr/bin/env bash
#SBATCH --job-name=auctionnet-reconv2all
#SBATCH --array=0-5
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32g
#SBATCH --partition=econ,batch
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/auctionnet/logs/%x_%A_%a.log
# Design A, all six focal roles on the repaired implementation (close-out of 9 November 2026): re-inference of the saved Hedge runs
# (i.i.d. and paired; main run and 20 coverage seeds) with the audited tables, and EXP3-IX with the exact reward width (new trajectory
# and 20 new coverage seeds), through sim_infer2.py (run_recon_v2.py). Outputs results/designA/recon_v2/role<r>/.
set -euo pipefail
ROOT=/home/magnolfi.1/NoRegret_data/auctionnet
CODE=/home/magnolfi.1/NoRegret_Code/python/auctionnet
AUD=/home/magnolfi.1/NoRegret_runs/auctionnet/audit_20261104
PY=$HOME/.conda/envs/auctionnet/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
cd "$CODE"
"$PY" run_recon_v2.py "$ROOT" "$SLURM_ARRAY_TASK_ID" "$AUD"
echo RECONV2ALL_JOB_DONE
