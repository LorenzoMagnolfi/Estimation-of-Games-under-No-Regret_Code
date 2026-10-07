#!/usr/bin/env bash
# Login-node watcher (sleep only, no compute): wait for the download and the mamba environment, then submit the
# conversion job and the Design A job with a Slurm dependency. Usage: nohup bash chain.sh > chain.log 2>&1 &
set -u
ROOT=$HOME/NoRegret_data/auctionnet
CODE=$HOME/NoRegret_Code/python/auctionnet
LOGS=$HOME/NoRegret_runs/auctionnet/logs
mkdir -p "$LOGS"
echo "chain started $(date)"
until grep -q ALLDONE "$ROOT/download.log" 2>/dev/null; do sleep 60; done
echo "download finished $(date)"
until [ -x "$HOME/.conda/envs/auctionnet/bin/python" ]; do sleep 60; done
echo "environment ready $(date)"
incomplete=$(grep -c INCOMPLETE "$ROOT/download.log" || true)
if [ "$incomplete" != "0" ]; then echo "WARNING: $incomplete incomplete zips; conversion will skip them"; fi
cj=$(sbatch --parsable "$CODE/unity/job_convert.sh")
echo "convert job $cj submitted $(date)"
dj=$(sbatch --parsable --dependency=afterok:$cj "$CODE/unity/job_designA.sh")
echo "designA job $dj submitted (afterok:$cj) $(date)"
echo CHAIN_SUBMITTED
