#!/bin/bash
#SBATCH --job-name=rl_experiment
#SBATCH --account=aip-machado
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --array=0-0               # Override with --array on sbatch command line
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

set -euo pipefail
mkdir -p logs

module load python/3.11 cuda gcc arrow

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Job Name: ${SLURM_JOB_NAME:-interactive}"
echo "Node: ${SLURMD_NODENAME:-localhost}"
echo "Start Time: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-1}"
echo "Memory: ${SLURM_MEM_PER_NODE:-?} MB"

# Positional args:
#   $1: gin config path
#   $@: gin bindings (each becomes `--binding VALUE`)
if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch --array=0-(seeds-1) scripts/job.sh CONFIG.gin [BINDING ...]"
  exit 1
fi

CONFIG_PATH="$1"
shift
BINDINGS=("$@")

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-~/mlruns}"

SEED="${SLURM_ARRAY_TASK_ID:-0}"

cmd=(uv run --offline python -m rl_research.main --config "$CONFIG_PATH" --seed "$SEED")
for b in "${BINDINGS[@]}"; do
  cmd+=(--binding "$b")
done

echo "Command: ${cmd[*]}"
"${cmd[@]}"
