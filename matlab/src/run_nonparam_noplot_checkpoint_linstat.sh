#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/m/magnolfi/jpe_revision_corrected_20260605_092931"
MATLAB_BIN="/software/matlab/bin/matlab"
CVX_DIR="${ROOT}/deps/cvx"
LOG="${ROOT}/matlab/output/nonparam_noplot_checkpoint_20260605_1152.nohup.log"
PIDFILE="${ROOT}/matlab/output/nonparam_noplot_checkpoint.pid"

cd "${ROOT}/matlab/src"

env CVX_DIR="${CVX_DIR}" \
  nohup "${MATLAB_BIN}" \
  -batch "II_CHECK_corrected_bundle_prereqs; II_RUN_nonparam_revision_noplot_checkpoint" \
  -prefersoftwareopengl > "${LOG}" 2>&1 &

echo "$!" > "${PIDFILE}"
echo "LAUNCHED_NONPARAM_NOPLOT"
cat "${PIDFILE}"
sleep 2
tail -25 "${LOG}" || true
