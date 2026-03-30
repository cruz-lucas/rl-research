#!/bin/bash
#SBATCH --job-name=export_results
#SBATCH --account=def-machado
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=/home/%u/scratch/logs/job_%A_%a_%x.out
#SBATCH --error=/home/%u/scratch/logs/job_%A_%a_%x.err

set -euo pipefail

LOG_DIR="$SCRATCH/logs"
mkdir -p "$LOG_DIR"

module load python/3.11 gcc arrow

VENV_DIR="$HOME/.venv/"
source "${VENV_DIR}/bin/activate"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Job Name: ${SLURM_JOB_NAME:-interactive}"
echo "Node: ${SLURMD_NODENAME:-localhost}"
echo "Start Time: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-1}"
echo "Memory: ${SLURM_MEM_PER_NODE:-?} MB"

srun ./scripts/mlruns_archive.sh compress ./mlruns_archive ~/scratch/mlruns/hpo_doorkey_8x8