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

# ─── Fixed settings ──────────────────────────────────────────────────────────
LOCAL_UPDATES=50
SAMPLING_RATE=0.2
MAX_GRAD_NORM=2.0
DP=True

RESULTS=False
TUNE=True
GLOBAL_STEP_SIZE=Adaptive
TUNING_TYPE=cross_validation
TUNING_PARAMETER=step_size

# ─── Defaults (used unless overridden below) ─────────────────────────────────
DEFAULT_GPU=4
# DEFAULT_HYPERPARAMETER="[0.00125,0.0025,0.005,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28]"
DEFAULT_HYPERPARAMETER="[0.005,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28]"
DEFAULT_RESUME=False

# ─── Sweep definitions ────────────────────────────────────────────────────────
# Each entry in SIGMA_LIST / CLIENT_RATIO_LIST defines one axis of the sweep.
#
# Override priority (highest to lowest): client_ratio > sigma > default
#
# Per-sigma overrides (uncomment / add as needed):
declare -A SIGMA_GPU=(
    # ["5.0"]=3
    # ["10.0"]=4
)
declare -A SIGMA_HYPERPARAMETER=(
    # ["50.0"]="[0.0025]"
)
declare -A SIGMA_RESUME=(
    # ["5.0"]=True
    # ["10.0"]=True
    # ["30.0"]=True
    # ["40.0"]=True
    # ["50.0"]=True
)

# Per-client_ratio overrides (uncomment / add as needed):
declare -A CLIENT_RATIO_GPU=(
    # ["0.02"]=5
    # ["0.1"]=6
)
declare -A CLIENT_RATIO_HYPERPARAMETER=(
    # ["0.02"]="[0.005,0.01,0.02]"
)
declare -A CLIENT_RATIO_RESUME=(
    # ["0.21"]=True
)

# Sweep values
SIGMA_LIST=(5.0 10.0 30.0 40.0 50.0)
CLIENT_RATIO_LIST=(0.21)

# ─── Launch jobs ─────────────────────────────────────────────────────────────
for SIGMA in "${SIGMA_LIST[@]}"; do
    for CLIENT_RATIO in "${CLIENT_RATIO_LIST[@]}"; do

        # Resolve GPU: client_ratio override > sigma override > default
        if [[ -n "${CLIENT_RATIO_GPU[$CLIENT_RATIO]+_}" ]]; then
            GPU="${CLIENT_RATIO_GPU[$CLIENT_RATIO]}"
        elif [[ -n "${SIGMA_GPU[$SIGMA]+_}" ]]; then
            GPU="${SIGMA_GPU[$SIGMA]}"
        else
            GPU="$DEFAULT_GPU"
        fi

        # Resolve HYPERPARAMETER: client_ratio override > sigma override > default
        if [[ -n "${CLIENT_RATIO_HYPERPARAMETER[$CLIENT_RATIO]+_}" ]]; then
            HYPERPARAMETER="${CLIENT_RATIO_HYPERPARAMETER[$CLIENT_RATIO]}"
        elif [[ -n "${SIGMA_HYPERPARAMETER[$SIGMA]+_}" ]]; then
            HYPERPARAMETER="${SIGMA_HYPERPARAMETER[$SIGMA]}"
        else
            HYPERPARAMETER="$DEFAULT_HYPERPARAMETER"
        fi

        # Resolve RESUME: client_ratio override > sigma override > default
        if [[ -n "${CLIENT_RATIO_RESUME[$CLIENT_RATIO]+_}" ]]; then
            RESUME="${CLIENT_RATIO_RESUME[$CLIENT_RATIO]}"
        elif [[ -n "${SIGMA_RESUME[$SIGMA]+_}" ]]; then
            RESUME="${SIGMA_RESUME[$SIGMA]}"
        else
            RESUME="$DEFAULT_RESUME"
        fi

        LOG="logs/sigma_${SIGMA}_client_ratio_${CLIENT_RATIO}"

        echo "Launching: sigma=${SIGMA}, client_ratio=${CLIENT_RATIO}, gpu=${GPU}, resume=${RESUME}"

        CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py \
            server.constant_global_step=$GLOBAL_STEP_SIZE \
            run_settings.resume_from_checkpoint=$RESUME \
            run_mode.tune_hyperparameter=$TUNE \
            run_mode.compile_tuning_results=$RESULTS \
            tuning.hyperparameter_grid=$HYPERPARAMETER \
            tuning.type=$TUNING_TYPE \
            tuning.parameter_to_tune=$TUNING_PARAMETER \
            server.dp=$DP \
            server.local_updates=$LOCAL_UPDATES \
            server.sampling_rate=$SAMPLING_RATE \
            server.sigma=$SIGMA \
            server.max_grad_norm=$MAX_GRAD_NORM \
            server.client_ratio=$CLIENT_RATIO \
            > "$LOG" 2>&1 &

    done
done

wait
echo "All jobs finished"
