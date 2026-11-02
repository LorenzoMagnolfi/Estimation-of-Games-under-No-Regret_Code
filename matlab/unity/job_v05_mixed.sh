#!/usr/bin/env bash
#SBATCH --job-name=noregret-v05-mixed
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=48g
#SBATCH --partition=econ,batch
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/v05lab_20261102/logs/%x_%j.log
# V05 lab batch 7: SIM-MIXED: Hedge learner against a constant-price and a myopic rival, joint and own regions at 1M and 4M (II_RUN_v05_mixed).
set -euo pipefail
ml matlab/r2024b
export SEDUMI_DIR=/home/magnolfi.1/NoRegret_tools/sedumi
cd $HOME/NoRegret_Code/matlab/src
matlab -batch "run('/home/magnolfi.1/NoRegret_tools/cvx/cvx_setup.m'); cd('/home/magnolfi.1/NoRegret_Code/matlab/src'); II_CHECK_corrected_bundle_prereqs; II_RUN_v05_mixed"
echo "=== V05_MIXED_DONE ==="
