#!/bin/bash
#SBATCH --job-name=ri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_long
#SBATCH --output=logs/%x-%j.out
#
# Builds the RAG index (data/rag_index) over all cleaned povarenok recipes, ~10-20 min on a V100.
#   sbatch slurm/rag_index.sbatch                                  # intfloat/multilingual-e5-large
#   sbatch slurm/rag_index.sbatch --encoder BAAI/bge-m3
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
srun python scripts/build_rag_index.py "$@"
