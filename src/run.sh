#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Pick a writable location
export XDG_CACHE_HOME="$HOME/.cache"
export HF_HOME="$HOME/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$XDG_CACHE_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE"
mkdir -p logs
trap 'echo; echo "Stopping all background jobs..."; kill 0' SIGINT SIGTERM

LOCAL_UPDATES=5000
SAMPLING_RATE=0.2
SIGMA=5.0
MAX_GRAD_NORM=2.0
DP=True

RESULTS=False
TUNE=True

GPU=0
HYPERPARAMETER="[0.16]"
LOG_PREFIX="0"

CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.02 

# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.02 > logs/${LOG_PREFIX}_client_ratio_0.02 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.04 > logs/${LOG_PREFIX}_client_ratio_0.04 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.06 > logs/${LOG_PREFIX}_client_ratio_0.06 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.08 > logs/${LOG_PREFIX}_client_ratio_0.08 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.1 > logs/${LOG_PREFIX}_client_ratio_0.1 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.13 > logs/${LOG_PREFIX}_client_ratio_0.13 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.15 > logs/${LOG_PREFIX}_client_ratio_0.15 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.17 > logs/${LOG_PREFIX}_client_ratio_0.17 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.21 > logs/${LOG_PREFIX}_client_ratio_0.21 2>&1 &

wait
echo "All jobs finished"

# # HYPERPARAMETER="[0.02]"
# # LOG_PREFIX="0.02"
# # CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.13 > logs/${LOG_PREFIX}_client_ratio_0.13 2>&1 &

# # HYPERPARAMETER="[0.08]"
# # LOG_PREFIX="0.08"
# # CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.15 > logs/${LOG_PREFIX}_client_ratio_0.15 2>&1 &
# # CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.21 > logs/${LOG_PREFIX}_client_ratio_0.21 2>&1 &

# GPU=6
# HYPERPARAMETER="[0.16]"
# LOG_PREFIX="0.16"
# # CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.06 > logs/${LOG_PREFIX}_client_ratio_0.06 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.17 > logs/${LOG_PREFIX}_client_ratio_0.17 2>&1 &


# HYPERPARAMETER="[0.32]"
# LOG_PREFIX="0.32"
# # CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.15 > logs/${LOG_PREFIX}_client_ratio_0.15 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.21 > logs/${LOG_PREFIX}_client_ratio_0.21 2>&1 &

# HYPERPARAMETER="[0.64]"
# LOG_PREFIX="0.64"
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.15 > logs/${LOG_PREFIX}_client_ratio_0.15 2>&1 &
# # CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.21 > logs/${LOG_PREFIX}_client_ratio_0.21 2>&1 &

# wait

# DP=False

# GPU=7
# HYPERPARAMETER="[0.01]"
# LOG_PREFIX="0.01_npv"
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.02 > logs/${LOG_PREFIX}_client_ratio_0.02 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.04 > logs/${LOG_PREFIX}_client_ratio_0.04 2>&1 &
# # CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.06 > logs/${LOG_PREFIX}_client_ratio_0.06 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.08 > logs/${LOG_PREFIX}_client_ratio_0.08 2>&1 &
# # CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.1 > logs/${LOG_PREFIX}_client_ratio_0.1 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.13 > logs/${LOG_PREFIX}_client_ratio_0.13 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.15 > logs/${LOG_PREFIX}_client_ratio_0.15 2>&1 &
# # CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.17 > logs/${LOG_PREFIX}_client_ratio_0.17 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.21 > logs/${LOG_PREFIX}_client_ratio_0.21 2>&1 &

# wait
# GPU=2
# HYPERPARAMETER="[0.02]"
# LOG_PREFIX="0.02_npv"
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.02 > logs/${LOG_PREFIX}_client_ratio_0.02 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.04 > logs/${LOG_PREFIX}_client_ratio_0.04 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.06 > logs/${LOG_PREFIX}_client_ratio_0.06 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.08 > logs/${LOG_PREFIX}_client_ratio_0.08 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.1 > logs/${LOG_PREFIX}_client_ratio_0.1 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.13 > logs/${LOG_PREFIX}_client_ratio_0.13 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.15 > logs/${LOG_PREFIX}_client_ratio_0.15 2>&1 &
# # CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.17 > logs/${LOG_PREFIX}_client_ratio_0.17 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.21 > logs/${LOG_PREFIX}_client_ratio_0.21 2>&1 &

# GPU=2
# HYPERPARAMETER="[0.04]"
# LOG_PREFIX="0.04_npv"
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.02 > logs/${LOG_PREFIX}_client_ratio_0.02 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.04 > logs/${LOG_PREFIX}_client_ratio_0.04 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.06 > logs/${LOG_PREFIX}_client_ratio_0.06 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.08 > logs/${LOG_PREFIX}_client_ratio_0.08 2>&1 &
# # CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.1 > logs/${LOG_PREFIX}_client_ratio_0.1 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.13 > logs/${LOG_PREFIX}_client_ratio_0.13 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.15 > logs/${LOG_PREFIX}_client_ratio_0.15 2>&1 &
# # CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.17 > logs/${LOG_PREFIX}_client_ratio_0.17 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.21 > logs/${LOG_PREFIX}_client_ratio_0.21 2>&1 &

# GPU=2
# HYPERPARAMETER="[0.08]"
# LOG_PREFIX="0.08_npv"
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.02 > logs/${LOG_PREFIX}_client_ratio_0.02 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.04 > logs/${LOG_PREFIX}_client_ratio_0.04 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.06 > logs/${LOG_PREFIX}_client_ratio_0.06 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.08 > logs/${LOG_PREFIX}_client_ratio_0.08 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.1 > logs/${LOG_PREFIX}_client_ratio_0.1 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.13 > logs/${LOG_PREFIX}_client_ratio_0.13 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.15 > logs/${LOG_PREFIX}_client_ratio_0.15 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.17 > logs/${LOG_PREFIX}_client_ratio_0.17 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.21 > logs/${LOG_PREFIX}_client_ratio_0.21 2>&1 &

# GPU=2
# HYPERPARAMETER="[0.16]"
# LOG_PREFIX="0.16_npv"
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.02 > logs/${LOG_PREFIX}_client_ratio_0.02 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.04 > logs/${LOG_PREFIX}_client_ratio_0.04 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.06 > logs/${LOG_PREFIX}_client_ratio_0.06 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.08 > logs/${LOG_PREFIX}_client_ratio_0.08 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.1 > logs/${LOG_PREFIX}_client_ratio_0.1 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.13 > logs/${LOG_PREFIX}_client_ratio_0.13 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.15 > logs/${LOG_PREFIX}_client_ratio_0.15 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.17 > logs/${LOG_PREFIX}_client_ratio_0.17 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.21 > logs/${LOG_PREFIX}_client_ratio_0.21 2>&1 &

# GPU=2
# HYPERPARAMETER="[0.32]"
# LOG_PREFIX="0.32_npv"
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.02 > logs/${LOG_PREFIX}_client_ratio_0.02 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.04 > logs/${LOG_PREFIX}_client_ratio_0.04 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.06 > logs/${LOG_PREFIX}_client_ratio_0.06 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.08 > logs/${LOG_PREFIX}_client_ratio_0.08 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.1 > logs/${LOG_PREFIX}_client_ratio_0.1 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.13 > logs/${LOG_PREFIX}_client_ratio_0.13 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.15 > logs/${LOG_PREFIX}_client_ratio_0.15 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.17 > logs/${LOG_PREFIX}_client_ratio_0.17 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.21 > logs/${LOG_PREFIX}_client_ratio_0.21 2>&1 &

# GPU=2
# HYPERPARAMETER="[0.64]"
# LOG_PREFIX="0.64_npv"
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.02 > logs/${LOG_PREFIX}_client_ratio_0.02 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.04 > logs/${LOG_PREFIX}_client_ratio_0.04 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.06 > logs/${LOG_PREFIX}_client_ratio_0.06 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.08 > logs/${LOG_PREFIX}_client_ratio_0.08 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.1 > logs/${LOG_PREFIX}_client_ratio_0.1 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.13 > logs/${LOG_PREFIX}_client_ratio_0.13 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.15 > logs/${LOG_PREFIX}_client_ratio_0.15 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.17 > logs/${LOG_PREFIX}_client_ratio_0.17 2>&1 &
# CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py run_mode.tune_hyperparameter=$TUNE run_mode.compile_tuning_results=$RESULTS tuning.hyperparameter_grid=$HYPERPARAMETER server.dp=$DP server.local_updates=$LOCAL_UPDATES server.sampling_rate=$SAMPLING_RATE server.sigma=$SIGMA server.max_grad_norm=$MAX_GRAD_NORM server.client_ratio=0.21 > logs/${LOG_PREFIX}_client_ratio_0.21 2>&1 &




# Synthetic Dataset without DP | IMPORTANT CHECK DP IS FALSE IN CONFIG
# CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.02 > logs/screen1_client_ratio_0.02 2>&1 &
# CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.04 > logs/screen1_client_ratio_0.04 2>&1 &
# CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.06 > logs/screen1_client_ratio_0.06 2>&1 &
# CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.08 > logs/screen1_client_ratio_0.08 2>&1 &
# CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.1 > logs/screen1_client_ratio_0.1 2>&1 &
# CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.13 > logs/screen2_client_ratio_0.13 2>&1 &
# CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.15 > logs/screen2_client_ratio_0.15 2>&1 &
# CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.17 > logs/screen2_client_ratio_0.17 2>&1 &
# CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.21 > logs/screen2_client_ratio_0.21 2>&1 &


# Synthetic Dataset with DP | IMPORTANT CHECK DP IS TRUE IN CONFIG
# CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python main.py server.sigma=1000.0 server.client_ratio=0.02 > logs/screen5_client_ratio_0.02 2>&1 &
# CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python main.py server.sigma=1000.0 server.client_ratio=0.04 > logs/screen5_client_ratio_0.04 2>&1 &
# CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python main.py server.sigma=1000.0 server.client_ratio=0.06 > logs/screen5_client_ratio_0.06 2>&1 &
# CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python main.py server.sigma=1000.0 server.client_ratio=0.08 > logs/screen5_client_ratio_0.08 2>&1 &
# CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python main.py server.sigma=1000.0 server.client_ratio=0.1 > logs/screen5_client_ratio_0.1 2>&1 &
# CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python main.py server.sigma=1000.0 server.client_ratio=0.13 > logs/screen5_client_ratio_0.13 2>&1 &
# CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python main.py server.sigma=1000.0 server.client_ratio=0.15 > logs/screen5_client_ratio_0.15 2>&1 &
# CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python main.py server.sigma=1000.0 server.client_ratio=0.17 > logs/screen5_client_ratio_0.17 2>&1 &
# CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python main.py server.sigma=1000.0 server.client_ratio=0.21 > logs/screen5_client_ratio_0.21 2>&1 &



# For compiling results after hyperparameter tuning
# CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.sampling_rate=0.2 server.local_updates=5 server.sigma=5.0 server.max_grad_norm=2.0 server.client_ratio=0.02 server.dp=True run_mode.tune_hyperparameter=False run_mode.compile_tuning_results=True
# CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.sampling_rate=0.2 server.local_updates=5 server.sigma=5.0 server.max_grad_norm=2.0 server.client_ratio=0.04 server.dp=True run_mode.tune_hyperparameter=False run_mode.compile_tuning_results=True
# CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.sampling_rate=0.2 server.local_updates=5 server.sigma=5.0 server.max_grad_norm=2.0 server.client_ratio=0.06 server.dp=True run_mode.tune_hyperparameter=False run_mode.compile_tuning_results=True
# CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.sampling_rate=0.2 server.local_updates=5 server.sigma=5.0 server.max_grad_norm=2.0 server.client_ratio=0.08 server.dp=True run_mode.tune_hyperparameter=False run_mode.compile_tuning_results=True
# CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.sampling_rate=0.2 server.local_updates=5 server.sigma=5.0 server.max_grad_norm=2.0 server.client_ratio=0.1 server.dp=True run_mode.tune_hyperparameter=False run_mode.compile_tuning_results=True
# CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.sampling_rate=0.2 server.local_updates=5 server.sigma=5.0 server.max_grad_norm=2.0 server.client_ratio=0.13 server.dp=True run_mode.tune_hyperparameter=False run_mode.compile_tuning_results=True
# CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.sampling_rate=0.2 server.local_updates=5 server.sigma=5.0 server.max_grad_norm=2.0 server.client_ratio=0.15 server.dp=True run_mode.tune_hyperparameter=False run_mode.compile_tuning_results=True
# CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.sampling_rate=0.2 server.local_updates=5 server.sigma=5.0 server.max_grad_norm=2.0 server.client_ratio=0.21 server.dp=True run_mode.tune_hyperparameter=False run_mode.compile_tuning_results=True
# CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.sampling_rate=0.2 server.local_updates=5 server.sigma=5.0 server.max_grad_norm=2.0 server.client_ratio=0.17 server.dp=True run_mode.tune_hyperparameter=False run_mode.compile_tuning_results=True
