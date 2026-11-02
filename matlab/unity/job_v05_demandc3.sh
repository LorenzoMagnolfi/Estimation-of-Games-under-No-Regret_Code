#!/usr/bin/env bash
#SBATCH --job-name=noregret-v05-demandc3
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=48g
#SBATCH --partition=econ,batch
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/v05lab_20261102/logs/%x_%j.log
# V05 lab batch 7: joint demand and cost on a wide two-stage cost grid plus the C3 projection with a first-stage demand set (II_RUN_v05_demand_c3).
set -euo pipefail
ml matlab/r2024b
export SEDUMI_DIR=/home/magnolfi.1/NoRegret_tools/sedumi
cd $HOME/NoRegret_Code/matlab/src
matlab -batch "run('/home/magnolfi.1/NoRegret_tools/cvx/cvx_setup.m'); cd('/home/magnolfi.1/NoRegret_Code/matlab/src'); II_CHECK_corrected_bundle_prereqs; II_RUN_v05_demand_c3"
echo "=== V05_DEMAND_C3_DONE ==="
