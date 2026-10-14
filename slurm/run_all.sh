#!/bin/bash
# Overnight pipeline on 4 GPUs as a chain of 1-GPU Slurm jobs (short job names):
#
#   login node: build_dataset (CPU, ~1-2 min)
#   d0 d1 d2 d3   LLM judge, 4 shards in parallel         4 GPUs, ~20-40 min
#      │  the last shard merges and writes data/clean/train.parquet
#      ├─► ty   train YandexGPT ──► eyf  eval yandex-ft
#      ├─► tq   train Qwen-7B   ──► eqf  eval qwen-ft     4 GPUs while training
#      ├─► eyb  eval yandex-base
#      └─► eqb  eval qwen-base
#
# Each job starts only if the ones it depends on succeeded (afterok). If something fails, the
# dependent jobs stay pending (DependencyNeverSatisfied) - remove them: scancel -u $USER --state=PENDING
#
# Usage (from the project root, on the login node, conda env active):  bash slurm/run_all.sh
set -eo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

YANDEX=yandex/YandexGPT-5-Lite-8B-instruct
QWEN=Qwen/Qwen2.5-7B-Instruct
YCFG=configs/yandexgpt5-lite-8b_v100-32gb.yaml
QCFG=configs/qwen2.5-7b_clean_v100-32gb.yaml
SHARDS=${SHARDS:-4}

submit() { sbatch --parsable "$@" | cut -d';' -f1; }

# Stage 1a: rule-based cleaning is CPU-only and quick - done right here, before the GPU jobs.
if [ ! -f data/clean/candidates.parquet ]; then
  ( source slurm/env.sh && python scripts/build_dataset.py --mlflow )
fi

# Stage 1b: LLM judge, one shard per GPU.
D=()
for ((k = 0; k < SHARDS; k++)); do
  D+=("$(submit --job-name=d$k slurm/data.sbatch -- --shard $k --num_shards $SHARDS)")
done
AFTER_DATA="afterok:$(IFS=:; echo "${D[*]}")"

# Stage 2: both trainings + both base-model evaluations at the same time.
TY=$(submit --dependency=$AFTER_DATA --job-name=ty slurm/train.sbatch "$YCFG")
TQ=$(submit --dependency=$AFTER_DATA --job-name=tq slurm/train.sbatch "$QCFG")
EYB=$(submit --dependency=$AFTER_DATA --job-name=eyb slurm/eval.sbatch yandex-base --base_model "$YANDEX")
EQB=$(submit --dependency=$AFTER_DATA --job-name=eqb slurm/eval.sbatch qwen-base --base_model "$QWEN")

# Stage 3: fine-tuned models.
EYF=$(submit --dependency=afterok:$TY --job-name=eyf slurm/eval.sbatch yandex-ft --adapter outputs/yandexgpt5-recipes/final_adapter)
EQF=$(submit --dependency=afterok:$TQ --job-name=eqf slurm/eval.sbatch qwen-ft --adapter outputs/qwen2.5-7b-clean-recipes/final_adapter)

cat <<EOF
Submitted:
  d0..d$((SHARDS - 1))  ${D[*]}   LLM judge shards
  ty   $TY   train YandexGPT        (after d*)
  tq   $TQ   train Qwen-7B          (after d*)
  eyb  $EYB   eval yandex-base       (after d*)
  eqb  $EQB   eval qwen-base         (after d*)
  eyf  $EYF   eval yandex-ft         (after ty)
  eqf  $EQF   eval qwen-ft           (after tq)
Watch:  squeue -u \$USER      logs: logs/<name>-<id>.out
EOF
