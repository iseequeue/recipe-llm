#!/bin/bash
# RAG experiment as a chain of 1-GPU Slurm jobs (run on the login node, from the project root):
#
#   ri    build the RAG index ──┬─► eybr  eval yandex-base + RAG
#                               ├─► eyfr  eval yandex-ft   + RAG
#                               └─► eqfr  eval qwen-ft     + RAG   (waits for the Qwen training job "tq" if it runs)
#
# Before the first run download the encoder (login node, internet):
#   python scripts/prefetch.py --skip_data --models --encoders intfloat/multilingual-e5-large
set -eo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
submit() { sbatch --parsable "$@" | cut -d';' -f1; }
INDEX=data/rag_index

if [ -f "$INDEX/info.json" ]; then
  echo "Index $INDEX exists - skipping ri (delete the folder to rebuild)"; DEP=""
else
  RI=$(submit slurm/rag_index.sbatch); DEP="--dependency=afterok:$RI"; echo "ri    $RI   build RAG index"
fi

EYBR=$(submit $DEP --job-name=eybr slurm/eval.sbatch yandex-base-rag --base_model yandex/YandexGPT-5-Lite-8B-instruct --rag $INDEX)
EYFR=$(submit $DEP --job-name=eyfr slurm/eval.sbatch yandex-ft-rag --adapter outputs/yandexgpt5-recipes/final_adapter --rag $INDEX)
echo "eybr  $EYBR   yandex-base + RAG"
echo "eyfr  $EYFR   yandex-ft + RAG"

QA=outputs/qwen2.5-7b-clean-recipes/final_adapter
TQ=$(squeue -u "$USER" -h -n tq -o %i | head -1)
if [ -n "$TQ" ]; then
  QDEP="--dependency=afterok:$TQ${RI:+:$RI}"
elif [ -d "$QA" ]; then
  QDEP="$DEP"
fi
if [ -n "$TQ" ] || [ -d "$QA" ]; then
  EQFR=$(submit $QDEP --job-name=eqfr slurm/eval.sbatch qwen-ft-rag --adapter $QA --rag $INDEX)
  echo "eqfr  $EQFR   qwen-ft + RAG ${TQ:+(after Qwen training $TQ)}"
else
  echo "Qwen adapter not found and no running 'tq' job - skipping qwen-ft-rag"
fi
echo "Watch: squeue -u \$USER   Then: python scripts/compare.py yandex-base yandex-base-rag yandex-ft yandex-ft-rag qwen-ft qwen-ft-rag"
