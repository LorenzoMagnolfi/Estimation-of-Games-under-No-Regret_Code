#!/usr/bin/env bash
#SBATCH --job-name=noregret-v05-smokeb
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16g
#SBATCH --partition=econ,batch
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/v05lab_20261102/logs/%x_%j.log
# V05 lab batch 7: batch 7 smoke test (learn_mixed, learn_hedge_units, eps_player_cap, cons_N_override, v05_solve_eta).
set -euo pipefail
ml matlab/r2024b
export SEDUMI_DIR=/home/magnolfi.1/NoRegret_tools/sedumi
cd $HOME/NoRegret_Code/matlab/src
matlab -batch "run('/home/magnolfi.1/NoRegret_tools/cvx/cvx_setup.m'); cd('/home/magnolfi.1/NoRegret_Code/matlab/src'); II_CHECK_corrected_bundle_prereqs; II_SMOKE_v05b"
echo "=== V05_SMOKE_B_DONE ==="
