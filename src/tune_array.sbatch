#!/bin/bash
#SBATCH --job-name=tune_cr
#SBATCH --output=/home/%u/logs/%x.%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --time=3:00:00
#SBATCH --array=0-6%7

set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Fast local scratch for caches (/tmp is node-local NVMe, meant for temporary job storage) :contentReference[oaicite:2]{index=2}
SCRATCH="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$SCRATCH"

export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" /home/$USER/logs

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv   # or: conda activate /l/users/<user.name>/conda/envs/myenv

LOCAL_UPDATES=50
SAMPLING_RATE=0.02
SIGMA=5.0
MAX_GRAD_NORM=2.0
DP=True

RESULTS=False
TUNE=True

HYPERPARAMETER="[0.16]"
LOG_PREFIX="0"

RATIOS=(0.02 0.04 0.06 0.08 0.10 0.13 0.15 0.17 0.21)
CR="${RATIOS[$SLURM_ARRAY_TASK_ID]}"

echo "Running client_ratio=$CR on node=$SLURMD_NODENAME, visible_gpus=$CUDA_VISIBLE_DEVICES"

PYTHONUNBUFFERED=1 python main.py \
  run_mode.tune_hyperparameter=$TUNE \
  run_mode.compile_tuning_results=$RESULTS \
  tuning.hyperparameter_grid=$HYPERPARAMETER \
  server.dp=$DP \
  server.local_updates=$LOCAL_UPDATES \
  server.sampling_rate=$SAMPLING_RATE \
  server.sigma=$SIGMA \
  server.max_grad_norm=$MAX_GRAD_NORM \
  server.client_ratio=$CR \
  > "logs/${LOG_PREFIX}_client_ratio_${CR}.log" 2>&1
