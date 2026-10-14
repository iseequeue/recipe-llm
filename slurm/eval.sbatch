#!/bin/bash
#SBATCH --job-name=recipe-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_long
#SBATCH --output=logs/%x-%j.out
#
# Evaluate one system on eval/prompts.jsonl (200 requests, greedy decoding; --resume keeps old answers):
#   sbatch slurm/eval.sbatch yandex-base --base_model yandex/YandexGPT-5-Lite-8B-instruct
#   sbatch slurm/eval.sbatch yandex-ft   --adapter outputs/yandexgpt5-recipes/final_adapter
#   sbatch slurm/eval.sbatch qwen-ft     --adapter outputs/qwen2.5-7b-clean-recipes/final_adapter
# Then on the login node: python scripts/compare.py && python scripts/make_sbs.py yandex-base yandex-ft
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
NAME=${1:?usage: sbatch slurm/eval.sbatch NAME (--adapter DIR | --base_model ID) [evaluate.py args]}
shift
srun python scripts/evaluate.py --name "$NAME" "$@"
