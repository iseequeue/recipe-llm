#!/usr/bin/env bash
# Usage: scripts/train_1gpu.sh [config] [key=value ...]
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:-configs/yandexgpt5-lite-8b_v100-32gb.yaml}; shift || true
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false
python train.py --config "$CONFIG" "$@"
