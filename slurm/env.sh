# Sourced by the sbatch scripts. Any variable here can be overridden by exporting it
# before `sbatch` (sbatch passes the submitting shell's environment to the job).

PROJECT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$PROJECT_DIR"

CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-llm}
source "$CONDA_SH"
conda activate "$CONDA_ENV"

# Model weights and the dataset were downloaded by slurm/setup_env.sh on the login node.
# Compute nodes usually have no internet, so never try to reach the Hub from a job.
export HF_HOME=${HF_HOME:-$PROJECT_DIR/hf_cache}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
DATA_GLOB=${DATA_GLOB:-$PROJECT_DIR/data/povarenok/data/*.parquet}

# Several jobs write to one sqlite MLflow store at once: wait up to 60 s for the lock instead of failing.
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:///$PROJECT_DIR/mlflow.db?timeout=60}
export MLFLOW_ENABLE_TELEMETRY=false
export TOKENIZERS_PARALLELISM=false
export MLFLOW_DISABLE_AGENT_HINT=1
export PYTHONUNBUFFERED=1
# Slurm >= 22.05: srun does not inherit --cpus-per-task from sbatch unless told to.
if [ -n "$SLURM_CPUS_PER_TASK" ]; then export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK; fi

[ -n "$SLURM_JOB_ID" ] || return 0   # sourced on the login node: no job info to print
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-none} | CPUs: ${SLURM_CPUS_PER_TASK:-?}"
echo "Python: $(command -v python) | HF_HOME: $HF_HOME"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
fi
