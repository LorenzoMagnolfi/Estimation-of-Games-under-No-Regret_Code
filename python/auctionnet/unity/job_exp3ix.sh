#!/usr/bin/env bash
#SBATCH --job-name=auctionnet-exp3ix
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32g
#SBATCH --partition=econ,batch
#SBATCH --array=0-5
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/auctionnet/logs/%x_%A_%a.log
# EXP3-IX (certified bandit family B) for one focal role: run to 1e7, inference with tailB/momentB, 20-seed coverage, figures (run_exp3ix.py).
set -euo pipefail
ROOT=/home/magnolfi.1/NoRegret_data/auctionnet
CODE=/home/magnolfi.1/NoRegret_Code/python/auctionnet
PY=$HOME/.conda/envs/auctionnet/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
cd "$CODE"
"$PY" run_exp3ix.py "$ROOT" "$SLURM_ARRAY_TASK_ID"
echo EXP3IX_JOB_DONE
