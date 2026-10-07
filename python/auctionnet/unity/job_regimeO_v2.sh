#!/usr/bin/env bash
#SBATCH --job-name=auctionnet-regimeO-v2
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32g
#SBATCH --partition=econ,batch
#SBATCH --array=0-5
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/auctionnet/logs/%x_%A_%a.log
# Rerun Regime O with the corrected tolerance on every Design A run of one focal role (rerun_regimeO.py).
set -euo pipefail
ROOT=/home/magnolfi.1/NoRegret_data/auctionnet
CODE=/home/magnolfi.1/NoRegret_Code/python/auctionnet
PY=$HOME/.conda/envs/auctionnet/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$CODE"
"$PY" rerun_regimeO.py "$ROOT" "$SLURM_ARRAY_TASK_ID"
echo REGIMEO_V2_JOB_DONE
