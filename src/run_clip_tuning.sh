#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Pick a writable location
export XDG_CACHE_HOME="${HOME}/.cache"
export HF_HOME="${HOME}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

mkdir -p "$XDG_CACHE_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" logs

# -----------------------------
# Config
# -----------------------------
LOCAL_UPDATES=50
SAMPLING_RATE=0.2
SIGMA=20.0
DP=True

RESULTS=False
TUNE=True
TUNING_TYPE="cross_validation"
GLOBAL_STEP_CONFIG=Heuristic
GLOBAL_STEP_SIZE=null
LOCAL_STEP_SIZE=5.12
RESUME=False
GPU=7
ROUNDS=150

HYPERPARAMETER="[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5]"
LOG_PREFIX="0"

# Client ratios to run
CLIENT_RATIOS=(
  0.02
  0.04
  0.06
  0.08
  0.10
  0.13
  0.15
  0.17
  0.21
)
# CLIENT_RATIOS=(
#   0.13
# )

# Common args for all runs (safer than one giant string)
COMMON_ARGS=(
  "server.constant_global_step=${GLOBAL_STEP_CONFIG}"
  "run_settings.rounds=${ROUNDS}"
  "server.global_step=${GLOBAL_STEP_SIZE}"
  "server.local_step=${LOCAL_STEP_SIZE}"
  "run_settings.resume_from_checkpoint=${RESUME}"
  "run_mode.tune_hyperparameter=${TUNE}"
  "run_mode.compile_tuning_results=${RESULTS}"
  "tuning.hyperparameter_grid=${HYPERPARAMETER}"
  "server.dp=${DP}"
  "server.local_updates=${LOCAL_UPDATES}"
  "server.sampling_rate=${SAMPLING_RATE}"
  "server.sigma=${SIGMA}"
  "tuning.type=${TUNING_TYPE}"
)

# Track background jobs so we can cleanly kill them on Ctrl+C
PIDS=()

cleanup() {
  echo
  echo "Stopping all background jobs..."
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
}
trap cleanup SIGINT SIGTERM

echo "Launching ${#CLIENT_RATIOS[@]} jobs on GPU ${GPU}..."

for ratio in "${CLIENT_RATIOS[@]}"; do
  log_file="logs/${LOG_PREFIX}_client_ratio_${ratio}.log"
  echo "Starting client_ratio=${ratio} -> ${log_file}"

  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
    python main.py \
      "${COMMON_ARGS[@]}" \
      "server.client_ratio=${ratio}" \
      >"$log_file" 2>&1 &

  PIDS+=("$!")
done

# Wait for all jobs to finish
wait
echo "All jobs finished"