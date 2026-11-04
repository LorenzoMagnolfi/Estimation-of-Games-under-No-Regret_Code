#!/usr/bin/env bash
#SBATCH --job-name=noregret-v05-primal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=48g
#SBATCH --partition=econ,batch
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/v05lab_20261102/logs/%x_%j.log
# V05 lab, reconciliation batch of 4 November 2026: full-coordinate primal feasibility against the reduced-direction lane, the normalized joint budget, and the own-count tail (II_CHECK_v05_primal).
set -euo pipefail
ml matlab/r2024b
export SEDUMI_DIR=/home/magnolfi.1/NoRegret_tools/sedumi
cd $HOME/NoRegret_Code/matlab/src
matlab -batch "run('/home/magnolfi.1/NoRegret_tools/cvx/cvx_setup.m'); cd('/home/magnolfi.1/NoRegret_Code/matlab/src'); II_CHECK_corrected_bundle_prereqs; II_CHECK_v05_primal"
echo "=== V05_PRIMAL_DONE ==="
