#!/usr/bin/env bash
#SBATCH --job-name=auctionnet-robust
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=40g
#SBATCH --partition=econ,batch
#SBATCH --array=0-5
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/auctionnet/logs/%x_%A_%a.log
# Design A robustness array: one task per focal role (index into results/rules/selection.csv).
# Variants: lambda0 x0.75/x1.5/category, 8 and 16 multiplier levels, 5 and 9 contexts, wide grid; seeds 2-3 at 1e8; STZ comparison.
set -euo pipefail
ROOT=/home/magnolfi.1/NoRegret_data/auctionnet
CODE=/home/magnolfi.1/NoRegret_Code/python/auctionnet
PY=$HOME/.conda/envs/auctionnet/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
cd "$CODE"
"$PY" run_robust.py "$ROOT" "$SLURM_ARRAY_TASK_ID"
echo ROBUST_DONE
