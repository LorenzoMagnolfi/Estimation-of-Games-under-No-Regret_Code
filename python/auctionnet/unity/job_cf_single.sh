#!/usr/bin/env bash
#SBATCH --job-name=auctionnet-cf
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=24g
#SBATCH --partition=econ,batch
#SBATCH --array=0-5
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/auctionnet/logs/%x_%A_%a.log
# Single-learner counterfactual illustration (cf_single.py), one task per focal role of results/rules/selection.csv.
set -euo pipefail
ROOT=/home/magnolfi.1/NoRegret_data/auctionnet
CODE=/home/magnolfi.1/NoRegret_Code/python/auctionnet
PY=$HOME/.conda/envs/auctionnet/bin/python
DA=$ROOT/results/designA
ROLE=$("$PY" -c "import pandas as pd; print(int(pd.read_csv('$ROOT/results/rules/selection.csv').role.iloc[$SLURM_ARRAY_TASK_ID]))")
mkdir -p "$ROOT/results/cf"
cd "$CODE"
"$PY" cf_single.py "$DA/env_role$ROLE.npz" "$DA/run_role${ROLE}_hedge_1e8.npz" "$DA/inf_role${ROLE}_hedge_1e8_regimeP.csv" "$ROOT/results/cf/role$ROLE"
echo CF_JOB_DONE
