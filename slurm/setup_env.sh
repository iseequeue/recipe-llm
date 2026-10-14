#!/bin/bash
# Run ONCE on the login node (it needs internet; compute nodes usually don't have it):
#   bash slurm/setup_env.sh
# Creates the conda env, installs PyTorch (cu126 build, has V100 kernels) + deps,
# downloads the model(s) and the dataset into the project folder.
#
# Other models (e.g. for the 3B / 7B configs):
#   MODELS="yandex/YandexGPT-5-Lite-8B-instruct Qwen/Qwen2.5-3B-Instruct" bash slurm/setup_env.sh
# Small home quota? Put the cache elsewhere and export the same HF_HOME before sbatch:
#   export HF_HOME=/scratch/$USER/hf_cache
set -eo pipefail
cd "$(dirname "$0")/.."

CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-llm}
MODELS=${MODELS:-yandex/YandexGPT-5-Lite-8B-instruct}
export HF_HOME=${HF_HOME:-$PWD/hf_cache}

source "$CONDA_SH"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  # conda-forge only: avoids the Anaconda ToS prompt on the default channel
  conda create -y -n "$CONDA_ENV" -c conda-forge --override-channels python=3.12
fi
conda activate "$CONDA_ENV"

pip install --upgrade pip
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

python scripts/prefetch.py --models $MODELS --data_dir data/povarenok
mkdir -p logs outputs

cat <<EOF

Setup done. Env: $CONDA_ENV, HF_HOME: $HF_HOME
1) Check the GPU node sees the card and torch has sm_70 kernels:
   srun -p gpu_long --gres=gpu:1 --time=00:10:00 bash -c 'source $CONDA_SH; conda activate $CONDA_ENV; python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"'
2) Short test run (~10-15 min):
   sbatch slurm/train.sbatch configs/qwen2.5-1.5b_v100.yaml max_train_samples=2000 run_name=quick
3) Full run:
   sbatch slurm/train.sbatch            # or: sbatch --gres=gpu:2 slurm/train.sbatch
EOF
