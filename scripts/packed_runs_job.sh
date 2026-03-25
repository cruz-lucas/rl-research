#!/bin/bash

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: scripts/packed_runs_job.sh MANIFEST.json [PROGRESS.jsonl]"
  exit 1
fi

MANIFEST_PATH="$1"
PROGRESS_FILE="${2:-}"

module load python/3.11 gcc arrow

VENV_DIR="$HOME/.venv/"
source "${VENV_DIR}/bin/activate"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Job Name: ${SLURM_JOB_NAME:-interactive}"
echo "Node: ${SLURMD_NODENAME:-localhost}"
echo "Start Time: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-1}"
echo "Memory: ${SLURM_MEM_PER_NODE:-?} MB"
echo "Manifest: ${MANIFEST_PATH}"
if [ -n "${PROGRESS_FILE}" ]; then
  echo "Progress File: ${PROGRESS_FILE}"
fi

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-$SCRATCH/navix_best}"

cmd=(python scripts/run_packed_job.py "$MANIFEST_PATH")
if [ -n "${PROGRESS_FILE}" ]; then
  cmd+=(--progress-file "$PROGRESS_FILE")
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"
