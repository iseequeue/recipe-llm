#!/bin/bash
#SBATCH --job-name=recipe-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_long
#SBATCH --output=logs/%x-%j.out
#
# Data pipeline: rule-based cleaning (CPU, ~1 min) + LLM-as-judge scoring (GPU, ~1-2 h for 24k recipes).
#   sbatch slurm/data.sbatch
#   sbatch slurm/data.sbatch --max_candidates 30000 -- --top_k 12000   # build args -- score args
# Resumable: re-submitting continues scoring from the last saved chunk.
# Result: data/clean/train.parquet, data/clean/judge_examples.md, MLflow experiment "recipe-data".
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
JUDGE_MODEL=${JUDGE_MODEL:-yandex/YandexGPT-5-Lite-8B-instruct}

BUILD_ARGS=(); SCORE_ARGS=(); target=BUILD_ARGS
for x in "$@"; do
  if [ "$x" = "--" ]; then target=SCORE_ARGS; continue; fi
  if [ "$target" = BUILD_ARGS ]; then BUILD_ARGS+=("$x"); else SCORE_ARGS+=("$x"); fi
done

if [ ! -f data/clean/candidates.parquet ] || [ ${#BUILD_ARGS[@]} -gt 0 ]; then
  srun python scripts/build_dataset.py --mlflow "${BUILD_ARGS[@]}"
fi
srun python scripts/score_recipes.py --model "$JUDGE_MODEL" --mlflow "${SCORE_ARGS[@]}"
