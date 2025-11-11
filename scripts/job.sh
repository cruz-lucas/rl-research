#!/bin/bash
#SBATCH --job-name=doublegoright_mcts_hpo
#SBATCH --account=aip-machado
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --output=logs/hpo_suite_experiment_%j.out
#SBATCH --error=logs/hpo_suite_experiment_%j.err

set -euo pipefail

mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Job Name: ${SLURM_JOB_NAME:-interactive}"
echo "Node: ${SLURMD_NODENAME:-localhost}"
echo "Start Time: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-1}"
echo "Memory: ${SLURM_MEM_PER_NODE:-?} MB"

module load python/3.11 gcc arrow

echo "Launching sweep via rl-run..."
uv run --offline --active rl-run run=doublegoright_fullyobs_mcts_rmax_empirical_sweep

echo "Job completed at: $(date)"
