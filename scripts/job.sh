#!/bin/bash
#SBATCH --job-name=rl_experiment
#SBATCH --account=aip-machado
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-29
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

export MLFLOW_TRACKING_URI=~/mlruns

uv run python run_experiment.py --config "./rl_research/configs/qlearning_doublegoright.gin"