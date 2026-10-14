#!/bin/bash
#SBATCH --job-name=recipe-llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_long
#SBATCH --output=logs/%x-%j.out
##SBATCH --mem=64G            # uncomment if the cluster's default RAM per job is small (7B needs ~32G host RAM)
#
# Submit from the project root (where train.py is):
#   sbatch slurm/train.sbatch                                            # YandexGPT-5-Lite-8B (after slurm/data.sbatch)
#   sbatch slurm/train.sbatch configs/qwen2.5-7b_clean_v100-32gb.yaml   # Qwen2.5-7B, same recipe
#   sbatch slurm/train.sbatch configs/yandexgpt5-lite-8b_v100-32gb.yaml max_train_samples=2000 run_name=quick
#   sbatch slurm/train.sbatch configs/yandexgpt5-lite-8b_v100-32gb.yaml \
#          resume_from_checkpoint=outputs/yandexgpt5-recipes/checkpoint-500   # continue after time limit
set -eo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"

CONFIG=${1:-configs/yandexgpt5-lite-8b_v100-32gb.yaml}
shift || true

# Configs that set data_files themselves (the YandexGPT one uses data/clean/train.parquet) keep it;
# otherwise train on the raw povarenok dump downloaded by scripts/prefetch.py.
DATA_ARGS=()
CFG_DATA=$(sed -n 's/^data_files:[[:space:]]*//p' "$CONFIG" | head -1)
if [ -n "$CFG_DATA" ]; then
  if ! ls $CFG_DATA >/dev/null 2>&1; then
    echo "No dataset at $CFG_DATA - run 'sbatch slurm/data.sbatch' first." >&2; exit 1
  fi
else
  if ! ls $DATA_GLOB >/dev/null 2>&1; then
    echo "No dataset at $DATA_GLOB - run 'python scripts/prefetch.py' on the login node first." >&2; exit 1
  fi
  DATA_ARGS=("data_files=$DATA_GLOB")
fi

NGPU=${NGPU:-${SLURM_GPUS_ON_NODE:-$(python -c "import torch; print(torch.cuda.device_count())")}}
NGPU=${NGPU##*:}   # "v100:2" -> "2" on clusters that report the GPU type
CPUS=${SLURM_CPUS_PER_TASK:-8}
PROCS=$(( NGPU > 1 ? NGPU : 1 ))
export OMP_NUM_THREADS=$(( CPUS / PROCS > 1 ? CPUS / PROCS / 2 : 1 ))
PREP_WORKERS=$(( CPUS > 8 ? 8 : CPUS ))
DL_WORKERS=$(( CPUS / PROCS / 4 > 1 ? CPUS / PROCS / 4 : 1 ))

# Cluster-specific settings first, then the user's key=value overrides (later ones win).
ARGS=(--config "$CONFIG"
      "${DATA_ARGS[@]}"
      "preprocessing_num_workers=$PREP_WORKERS"
      "dataloader_num_workers=$DL_WORKERS"
      "run_name=${SLURM_JOB_NAME:-recipe-llm}-${SLURM_JOB_ID:-local}"
      "$@")

echo "Config: $CONFIG | GPUs: $NGPU | OMP_NUM_THREADS=$OMP_NUM_THREADS | overrides: $*"
if [ "$NGPU" -gt 1 ]; then
  # export NCCL_P2P_DISABLE=1   # uncomment if a 2x PCIe V100 job hangs at NCCL init
  # Static rendezvous on 127.0.0.1 instead of --standalone: on some nodes the name
  # "localhost" does not resolve and torchrun --standalone times out.
  MASTER_PORT=${MASTER_PORT:-$(python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")}
  echo "torchrun: 127.0.0.1:$MASTER_PORT, $NGPU processes"
  srun torchrun --nnodes=1 --node_rank=0 --nproc_per_node="$NGPU" \
       --rdzv_backend=static --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
       train.py "${ARGS[@]}"
else
  srun python train.py "${ARGS[@]}"
fi
