#!/usr/bin/env bash
#SBATCH --job-name=noregret-v05-primalv2
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=48g
#SBATCH --partition=econ,batch
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/v05lab_20261102/logs/%x_%j.log
# V05 lab, close-out of 9 November 2026 (Niccolò's C1 to C3): the full-coordinate primal check rerun with upward-replaced floors, analytic
# supporting lines with a certificate, minimum-violation LPs with unresolved statuses preserved, and the matched rectangle (II_CHECK_v05_primal_v2).
set -euo pipefail
ml matlab/r2024b
export SEDUMI_DIR=/home/magnolfi.1/NoRegret_tools/sedumi
cd $HOME/NoRegret_Code/matlab/src
matlab -batch "run('/home/magnolfi.1/NoRegret_tools/cvx/cvx_setup.m'); cd('/home/magnolfi.1/NoRegret_Code/matlab/src'); II_CHECK_corrected_bundle_prereqs; II_CHECK_v05_primal_v2"
echo "=== V05_PRIMAL_V2_DONE ==="
