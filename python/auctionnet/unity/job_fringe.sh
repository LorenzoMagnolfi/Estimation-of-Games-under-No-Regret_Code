#!/usr/bin/env bash
#SBATCH --job-name=auctionnet-fringe
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=48g
#SBATCH --partition=econ,batch
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/auctionnet/logs/%x_%j.log
# Fringe-independence check on all periods (check_fringe.py): correlations for every role, payoff tables for the focal roles.
set -euo pipefail
ROOT=/home/magnolfi.1/NoRegret_data/auctionnet
CODE=/home/magnolfi.1/NoRegret_Code/python/auctionnet
PY=$HOME/.conda/envs/auctionnet/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$CODE"
"$PY" check_fringe.py "$ROOT" "$ROOT/results/fringe"
echo FRINGE_JOB_DONE
