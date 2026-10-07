#!/usr/bin/env bash
#SBATCH --job-name=auctionnet-designA
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=48g
#SBATCH --partition=econ,batch
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/auctionnet/logs/%x_%j.log
# Design A definitive run: footprints on all periods, rule-based selection, pooled environments, Hedge/EXP3 runs,
# both inference regimes, 20-seed coverage, figures, manifest. Stage argument optional (default 7 = all).
set -euo pipefail
ROOT=/home/magnolfi.1/NoRegret_data/auctionnet
CODE=/home/magnolfi.1/NoRegret_Code/python/auctionnet
STAGE="${1:-7}"
PY=$HOME/.conda/envs/auctionnet/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
cd "$CODE"
"$PY" run_designA.py "$ROOT" "$STAGE"
echo DESIGNA_DONE
