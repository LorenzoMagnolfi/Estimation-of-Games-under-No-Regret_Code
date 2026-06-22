#!/usr/bin/env bash
#
# Launch the corrected JPE revision simulation bundle on linstat.
#
# Usage from matlab/src:
#   bash run_corrected_bundle_linstat.sh
#
# Optional environment variables:
#   MATLAB_BIN        MATLAB executable, default /software/matlab/bin/matlab
#   CVX_DIR           directory containing a CVX installation
#   SEDUMI_DIR        directory containing a SeDuMi installation
#   MATLAB_EXTRA_PATH pathsep-separated extra MATLAB paths

set -euo pipefail

MATLAB_BIN="${MATLAB_BIN:-/software/matlab/bin/matlab}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OUTPUT_DIR="../output"
mkdir -p "${OUTPUT_DIR}"

if [[ ! -x "${MATLAB_BIN}" ]]; then
  echo "MATLAB executable not found or not executable: ${MATLAB_BIN}" >&2
  exit 1
fi

echo "Checking corrected bundle prerequisites..."
"${MATLAB_BIN}" -batch "II_CHECK_corrected_bundle_prereqs"

STAMP="$(date +%Y%m%d_%H%M%S)"
NOHUP_LOG="${OUTPUT_DIR}/corrected_revision_bundle_${STAMP}.nohup.log"
PID_FILE="${OUTPUT_DIR}/corrected_revision_bundle.pid"

echo "Launching corrected revision bundle..."
export MATLAB_BIN
if [[ -n "${CVX_DIR:-}" ]]; then export CVX_DIR; fi
if [[ -n "${SEDUMI_DIR:-}" ]]; then export SEDUMI_DIR; fi
if [[ -n "${MATLAB_EXTRA_PATH:-}" ]]; then export MATLAB_EXTRA_PATH; fi
setsid nohup bash -c '
  echo "=== MATLAB bundle process starting ==="
  date
  echo "MATLAB_BIN=${MATLAB_BIN}"
  echo "CVX_DIR=${CVX_DIR:-}"
  "${MATLAB_BIN}" -batch "II_RUN_revision_corrected_bundle"
  status=$?
  echo "=== MATLAB bundle process finished with status ${status} ==="
  date
  exit "${status}"
' > "${NOHUP_LOG}" 2>&1 &
PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "Started MATLAB PID ${PID}"
echo "Nohup log: ${NOHUP_LOG}"
echo "Bundle diary: ${OUTPUT_DIR}/corrected_revision_bundle.log"
echo "Status artifact: ${OUTPUT_DIR}/corrected_revision_bundle_status.mat"
