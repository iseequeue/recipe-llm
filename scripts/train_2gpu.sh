#!/usr/bin/env bash
# Data-parallel (DDP) training on 2 GPUs. Usage: scripts/train_2gpu.sh [config] [key=value ...]
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:-configs/qwen2.5-1.5b_v100.yaml}; shift || true
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TOKENIZERS_PARALLELISM=false
# PCIe V100 boxes without P2P sometimes hang in NCCL init; uncomment if so (SXM2/NVLink is fine):
# export NCCL_P2P_DISABLE=1
MASTER_PORT=${MASTER_PORT:-$(python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")}
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2 --rdzv_backend=static \
  --master_addr=127.0.0.1 --master_port="$MASTER_PORT" train.py --config "$CONFIG" "$@"
