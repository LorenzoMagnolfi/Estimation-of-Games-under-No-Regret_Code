#!/usr/bin/env bash
#SBATCH --job-name=swappa-horserace-tables
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=40g
#SBATCH --partition=econ,batch
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/swappa_horserace_20261109/logs/%x_%j.log
# Per-cell payoff tables on the log-price lattice for the horse race against the equilibrium benchmark
# (9 November 2026): 13 sellers, demand raw_d0, Decluttr and internal references.
set -euo pipefail
CODE=/home/magnolfi.1/NoRegret_Code/swappa
RUN=/home/magnolfi.1/NoRegret_runs/swappa_horserace_20261109
PY=$HOME/.conda/envs/auctionnet/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$CODE"
sha256sum python/swappa_rescue/swappa_horserace_tables_20261109.py python/swappa_rescue/swappa_common_20261108.py \
  python/swappa_rescue/swappa_core_inference_20261108.py output/swappa_demand_repair_20261106/raw_reference_panel.parquet \
  output/swappa_demand_repair_20261106/raw_d0_coef.npz
"$PY" python/swappa_rescue/swappa_horserace_tables_20261109.py --spec raw_d0 --sellers all
mkdir -p "$RUN/results"
cp output/swappa_horserace_20261109/*.npz "$RUN/results/"
echo SWAPPA_HORSERACE_TABLES_DONE
