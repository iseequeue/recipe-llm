#!/bin/bash
#SBATCH --job-name=recipe-app
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_long
#SBATCH --output=logs/%x-%j.out
#
# Web demo on a GPU node. The public link (https://....gradio.live) appears in the log:
#   sbatch slurm/app.sbatch outputs/yandexgpt5-recipes/final_adapter
#   grep -a gradio.live logs/recipe-app-<JOBID>.out
# Stop it: scancel <JOBID>
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
ADAPTER=${1:?usage: sbatch slurm/app.sbatch ADAPTER_DIR [app.py args...]}
shift
export HF_HUB_OFFLINE=1
# gradio --share needs internet from the node to reach gradio.live
srun python app.py --adapter "$ADAPTER" --share "$@"
