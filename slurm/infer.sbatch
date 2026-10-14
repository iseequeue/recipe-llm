#!/bin/bash
#SBATCH --job-name=recipe-infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_long
#SBATCH --output=logs/%x-%j.out
#
# Batch generation for a file of requests (one per line) -> jsonl.
#   sbatch slurm/infer.sbatch outputs/qwen2.5-1.5b-recipes/final_adapter
#   sbatch slurm/infer.sbatch outputs/qwen2.5-1.5b-recipes/final_adapter my_prompts.txt out.jsonl --temperature 0
#   sbatch slurm/infer.sbatch runs:/<mlflow_run_id>/adapter
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

ADAPTER=${1:?usage: sbatch slurm/infer.sbatch ADAPTER [PROMPTS_FILE] [OUT_JSONL] [infer.py args...]}
PROMPTS=${2:-slurm/example_prompts.txt}
OUT=${3:-outputs/generations-${SLURM_JOB_ID:-local}.jsonl}
shift $(( $# < 3 ? $# : 3 ))

mkdir -p "$(dirname "$OUT")"
srun python infer.py --adapter "$ADAPTER" --input_file "$PROMPTS" --output_file "$OUT" --batch_size 8 "$@"
echo "Written: $OUT"
