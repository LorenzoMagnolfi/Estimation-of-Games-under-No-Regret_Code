#!/usr/bin/env bash
#SBATCH --job-name=auctionnet-convert
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32g
#SBATCH --partition=econ,batch
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/auctionnet/logs/%x_%j.log
# Convert every period zip to parquet and per-period environment files (skips existing and incomplete zips).
set -euo pipefail
ROOT=/home/magnolfi.1/NoRegret_data/auctionnet
CODE=/home/magnolfi.1/NoRegret_Code/python/auctionnet
PY=$HOME/.conda/envs/auctionnet/bin/python
mkdir -p "$ROOT/parquet" "$ROOT/env"
"$PY" "$CODE/convert_all.py" "$ROOT/raw" "$ROOT/parquet" "$ROOT/env"
echo CONVERT_DONE
