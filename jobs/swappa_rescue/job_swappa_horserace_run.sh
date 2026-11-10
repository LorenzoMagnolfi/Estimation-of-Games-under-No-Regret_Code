#!/usr/bin/env bash
#SBATCH --job-name=swappa-horserace-run
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=12g
#SBATCH --partition=econ,batch
#SBATCH --output=/home/magnolfi.1/NoRegret_runs/swappa_horserace_20261109/logs/%x_%j.log
# Horse race against the equilibrium benchmark (9 November 2026): real-data evaluation for the 13 sellers (two
# references) and the laboratory with known truth for three sellers, on the tables of job 11392145.
set -euo pipefail
CODE=/home/magnolfi.1/NoRegret_Code/swappa
RUN=/home/magnolfi.1/NoRegret_runs/swappa_horserace_20261109
PY=$HOME/.conda/envs/auctionnet/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$CODE"
sha256sum python/swappa_rescue/swappa_horserace_core_20261109.py python/swappa_rescue/swappa_horserace_eval_20261109.py \
  python/swappa_rescue/swappa_horserace_lab_20261109.py output/swappa_horserace_20261109/*.npz
"$PY" python/swappa_rescue/swappa_horserace_eval_20261109.py
for S in "GadgetPickup" "AMS Traders Inc." "Certified Cell, Inc"; do
  "$PY" python/swappa_rescue/swappa_horserace_lab_20261109.py --seller "$S" --reps 500
done
cp output/swappa_horserace_20261109/*.json "$RUN/results/"
echo SWAPPA_HORSERACE_RUN_DONE
