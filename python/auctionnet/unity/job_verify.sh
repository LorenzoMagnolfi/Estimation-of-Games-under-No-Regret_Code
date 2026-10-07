#!/usr/bin/env bash
#SBATCH --job-name=auctionnet-verify
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=64g
#SBATCH --partition=econ,batch
#SBATCH --array=0-20
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/auctionnet/logs/%x_%A_%a.log
# Verify the auction rule on every AuctionNet period (array index 0..20 -> period 7..27), without the ranked parquet.
set -euo pipefail
ROOT=/home/magnolfi.1/NoRegret_data/auctionnet
CODE=/home/magnolfi.1/NoRegret_Code/python/auctionnet
PY=$HOME/.conda/envs/auctionnet/bin/python
PERIOD=$((SLURM_ARRAY_TASK_ID + 7))
mkdir -p "$ROOT/results/verify"
cd "$CODE"
"$PY" verify_auction.py "$ROOT/parquet/period-$PERIOD.parquet" "$ROOT/results/verify" --no-save
echo VERIFY_DONE
