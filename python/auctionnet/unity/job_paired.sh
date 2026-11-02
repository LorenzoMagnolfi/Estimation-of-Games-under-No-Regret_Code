#!/usr/bin/env bash
#SBATCH --job-name=auctionnet-paired
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32g
#SBATCH --partition=econ,batch
#SBATCH --array=0-5
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/auctionnet/logs/%x_%A_%a.log
# Paired-environment robustness row (R1.3) for one focal role: rebuild the environment with per-opportunity context bins,
# Hedge to 1e8 in mode paired, inference with the certified tail, 20-seed coverage, figures (run_paired.py).
set -euo pipefail
ROOT=/home/magnolfi.1/NoRegret_data/auctionnet
CODE=/home/magnolfi.1/NoRegret_Code/python/auctionnet
PY=$HOME/.conda/envs/auctionnet/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
cd "$CODE"
"$PY" run_paired.py "$ROOT" "$SLURM_ARRAY_TASK_ID"
echo PAIRED_JOB_DONE
