#!/bin/bash
# Re-evaluate all 8 systems on the current eval/prompts.jsonl as 8 independent 1-GPU jobs (4 GPUs -> 2 waves).
# --resume keeps answers already generated for unchanged prompts (decoding is greedy), so after extending the
# eval set only the new prompts are generated. Metrics are recomputed for all answers.
#
#   bash slurm/run_eval.sh               # from the project root, on the login node
#   then:  python scripts/report.py
# If a job hits the time limit, just run this script again: it continues from the saved answers.
set -eo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
submit() { sbatch --parsable "$@" | cut -d';' -f1; }

YANDEX=yandex/YandexGPT-5-Lite-8B-instruct
QWEN=Qwen/Qwen2.5-7B-Instruct
YA=outputs/yandexgpt5-recipes/final_adapter
QA=outputs/qwen2.5-7b-clean-recipes/final_adapter
RAG=data/rag_index
for p in "$YA" "$QA" "$RAG/info.json"; do
  [ -e "$p" ] || { echo "missing $p - run slurm/run_all.sh / slurm/run_rag.sh first"; exit 1; }
done

ev() {  # job-name system args...
  local job=$1 name=$2; shift 2
  echo "$job  $(submit --job-name="$job" slurm/eval.sbatch "$name" --resume "$@")  $name"
}
ev eyb  yandex-base      --base_model "$YANDEX"
ev eyf  yandex-ft        --adapter "$YA"
ev eqb  qwen-base        --base_model "$QWEN"
ev eqf  qwen-ft          --adapter "$QA"
ev eybr yandex-base-rag  --base_model "$YANDEX" --rag "$RAG"
ev eyfr yandex-ft-rag    --adapter "$YA" --rag "$RAG"
ev eqbr qwen-base-rag    --base_model "$QWEN" --rag "$RAG"
ev eqfr qwen-ft-rag      --adapter "$QA" --rag "$RAG"
echo "Watch: squeue -u \$USER     When all are done: python scripts/report.py"
