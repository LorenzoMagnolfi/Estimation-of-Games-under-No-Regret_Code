#!/usr/bin/env bash
#SBATCH --job-name=noregret-v05-realization
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=8g
#SBATCH --partition=econ,batch
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/v05lab_20261102/logs/%x_%j.log
# V05 lab: realization-conditional (persistent-type) data-only sets under the certified tail (II_RUN_v05_realization).
set -euo pipefail
ml matlab/r2024b
export SEDUMI_DIR=/home/magnolfi.1/NoRegret_tools/sedumi
cd $HOME/NoRegret_Code/matlab/src
matlab -batch "cd('/home/magnolfi.1/NoRegret_Code/matlab/src'); II_RUN_v05_realization"
echo "=== V05_REALIZATION_DONE ==="
