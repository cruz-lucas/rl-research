#!/bin/bash
#SBATCH --job-name=doublegoright_mcts_hpo
#SBATCH --account=aip-machado
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --output=logs/hpo_suite_experiment_%j.out
#SBATCH --error=logs/hpo_suite_experiment_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"

module load python/3.11 gcc arrow

uv run --offline --active python examples/doublegoright_mcts_hpo.py

echo "Job completed at: $(date)"