#!/usr/bin/env bash
#
# Run ONE revision simulation runner on Linstat with CVX/SeDuMi on the path and
# a provenance manifest. Designed to be the command handed to `ssubmit`.
#
#   ssubmit --partition=econ-fac --cores=8 --mem=32g \
#     --output=$HOME/noregret_logs/nonparam_%j.out \
#     "bash $HOME/Estimation-of-Games-under-No-Regret_Code/matlab/src/run_one_linstat.sh II_RUN_nonparam_revision"
#
# Env overrides: MATLAB_BIN, CVX_DIR.
#
# NOTE: no `set -e` around the MATLAB call — we want to record the exit code in
# the manifest even when MATLAB errors out.

set -uo pipefail

RUNNER="${1:?usage: run_one_linstat.sh <RUNNER_SCRIPT_NAME>}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/../.." && pwd)"
MATLAB_BIN="${MATLAB_BIN:-/software/matlab/bin/matlab}"
export CVX_DIR="${CVX_DIR:-/home/m/magnolfi/jpe_revision_corrected_20260605_092931/deps/cvx}"

MANIFEST_DIR="${REPO_ROOT}/matlab/output/manifests"
mkdir -p "${MANIFEST_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo nogit)"
DIRTY="$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null | head -1)"
MANIFEST="${MANIFEST_DIR}/${RUNNER}_${STAMP}.manifest"
{
  echo "runner=${RUNNER}"
  echo "git_sha=${SHA}"
  echo "git_dirty=$([ -n "${DIRTY}" ] && echo yes || echo no)"
  echo "host=$(hostname)"
  echo "slurm_job=${SLURM_JOB_ID:-none}"
  echo "started=$(date -Is)"
  echo "cvx_dir=${CVX_DIR}"
  echo "matlab_bin=${MATLAB_BIN}"
} > "${MANIFEST}"
echo "[run_one] ${RUNNER}  sha=${SHA}  manifest=${MANIFEST}"

if [[ ! -x "${MATLAB_BIN}" ]]; then
  echo "ERROR: MATLAB not executable at ${MATLAB_BIN}" >&2
  echo "exit_code=127" >> "${MANIFEST}"
  exit 127
fi

cd "${SRC_DIR}"
# II_CHECK_corrected_bundle_prereqs adds CVX_DIR (genpath) and asserts cvx/sedumi
# are on the path; then the runner executes (its `clear all` does not reset the path).
"${MATLAB_BIN}" -nodisplay -batch "II_CHECK_corrected_bundle_prereqs; ${RUNNER}"
RC=$?

{
  echo "finished=$(date -Is)"
  echo "exit_code=${RC}"
} >> "${MANIFEST}"
echo "[run_one] ${RUNNER} done rc=${RC}"
exit "${RC}"
