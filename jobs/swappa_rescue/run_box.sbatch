#!/usr/bin/env bash
#SBATCH --account=PAS3582
#SBATCH --partition=cpu
#SBATCH --job-name=swappa-box
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --mail-type=NONE
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
umask 077
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
RUN="$SLURM_SUBMIT_DIR"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/swappa_${SLURM_JOB_ID}_XXXXXX")
tar -xzf "$RUN/package.tar.gz" -C "$WORK"
/users/PAS3582/magnolfi1/envs/pc3/bin/python "$WORK/jobs/swappa_rescue/run_box.py" --task "$SLURM_ARRAY_TASK_ID" --tasks "$RUN/tasks.json" --export "$RUN/results/task_$SLURM_ARRAY_TASK_ID"
