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

# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
# Set SWEEP_PARAM to the parameter you want to vary:
#   sigma | rounds | local_updates | sampling_rate | client_ratio
# Then fill in the corresponding *_LIST and leave the fixed values below.
# ═══════════════════════════════════════════════════════════════════════════════

SWEEP_PARAM=sampling_rate

SIGMA_LIST=(20.0)
LOCAL_UPDATES_LIST=(8)
SAMPLING_RATE_LIST=(0.15)
CLIENT_RATIO_LIST=(0.02)

# ─── Fixed values (used for all parameters NOT being swept) ───────────────────
FIXED_SIGMA=20.0
FIXED_ROUNDS=150
FIXED_LOCAL_UPDATES=50
FIXED_SAMPLING_RATE=0.2
FIXED_CLIENT_RATIO=0.21

# ─── Other fixed settings ─────────────────────────────────────────────────────
MAX_GRAD_NORM=2.0
DP=True
RESULTS=False
TUNE=True
GLOBAL_STEP_SIZE=Adaptive
GLOBAL_STEP=1.0
LOCAL_STEP=1.28
TUNING_TYPE=cross_validation

# ─── Defaults for overridable settings ────────────────────────────────────────
DEFAULT_GPU=5
DEFAULT_RESUME=False
PARAMETER_TO_TUNE="step_size"
GPU_LIST=(5) # GPUs to cycle through for parallel jobs

if [ "$PARAMETER_TO_TUNE" == "step_size" ]; then
    # DEFAULT_HYPERPARAMETER="[0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24]"
    DEFAULT_HYPERPARAMETER="[0.16]"
elif [ "$PARAMETER_TO_TUNE" == "clipping" ]; then
    DEFAULT_HYPERPARAMETER="[0.5,1.0,1.5,2.0,2.5,3.0,3.5]"
fi

# ─── Per-value overrides for GPU, HYPERPARAMETER, and RESUME ──────────────────
# Keys are values of the swept parameter.
# Override priority (highest wins): value-specific override > default
#
# Examples (uncomment / add as needed):
declare -A OVERRIDE_GPU=(
    # ["5.0"]=3
    # ["0.1"]=6
)
declare -A OVERRIDE_HYPERPARAMETER=(
    # ["12"]="[0.04,0.08,0.16,0.32,0.64,1.28]"
    # ["32"]="[0.04,0.08,0.16,0.32,0.64,1.28]"
    # ["0.1"]="[3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5]"
)
declare -A OVERRIDE_RESUME=(
    # ["5.0"]=True
    # ["0.21"]=True
)

# ═══════════════════════════════════════════════════════════════════════════════
# LAUNCH JOBS  (no edits needed below this line)
# ═══════════════════════════════════════════════════════════════════════════════

# Select which list to iterate over
case "$SWEEP_PARAM" in
    sigma)         SWEEP_LIST=("${SIGMA_LIST[@]}") ;;
    rounds)        SWEEP_LIST=("${ROUNDS_LIST[@]}") ;;
    local_updates) SWEEP_LIST=("${LOCAL_UPDATES_LIST[@]}") ;;
    sampling_rate) SWEEP_LIST=("${SAMPLING_RATE_LIST[@]}") ;;
    client_ratio)  SWEEP_LIST=("${CLIENT_RATIO_LIST[@]}") ;;
    *) echo "Unknown SWEEP_PARAM: $SWEEP_PARAM"; exit 1 ;;
esac

GPU_COUNT=${#GPU_LIST[@]}
JOB_INDEX=0

for VAL in "${SWEEP_LIST[@]}"; do

    # Apply fixed values, then override the swept parameter
    SIGMA="$FIXED_SIGMA"
    ROUNDS="$FIXED_ROUNDS"
    LOCAL_UPDATES="$FIXED_LOCAL_UPDATES"
    SAMPLING_RATE="$FIXED_SAMPLING_RATE"
    CLIENT_RATIO="$FIXED_CLIENT_RATIO"

    case "$SWEEP_PARAM" in
        sigma)         SIGMA="$VAL" ;;
        rounds)        ROUNDS="$VAL" ;;
        local_updates) LOCAL_UPDATES="$VAL" ;;
        sampling_rate) SAMPLING_RATE="$VAL" ;;
        client_ratio)  CLIENT_RATIO="$VAL" ;;
    esac

    # Resolve per-value overrides
    RESUME="${OVERRIDE_RESUME[$VAL]:-$DEFAULT_RESUME}"
    BASE_HYPERPARAMETER_GRID="${OVERRIDE_HYPERPARAMETER[$VAL]:-$DEFAULT_HYPERPARAMETER}"

    # Clean up the grid string and prepare for iteration
    CLEAN_GRID=$(echo "$BASE_HYPERPARAMETER_GRID" | tr -d '[] ')
    
    IFS=',' read -r -a HYPERPARAMETER_VALUES <<< "$CLEAN_GRID"

    for HP_VAL in "${HYPERPARAMETER_VALUES[@]}"; do
        # Cycle through available GPUs
        GPU_INDEX=$((JOB_INDEX % GPU_COUNT))
        GPU=${GPU_LIST[$GPU_INDEX]}
        JOB_INDEX=$((JOB_INDEX + 1))

        LOG="logs/${SWEEP_PARAM}_${VAL}_${PARAMETER_TO_TUNE}_${HP_VAL}_gpu${GPU}_2.log"

        echo "Launching: ${SWEEP_PARAM}=${VAL}, ${PARAMETER_TO_TUNE}=${HP_VAL} | sigma=${SIGMA} rounds=${ROUNDS} local_updates=${LOCAL_UPDATES} sampling_rate=${SAMPLING_RATE} client_ratio=${CLIENT_RATIO} | gpu=${GPU} resume=${RESUME}"

        CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py \
            server.constant_global_step=$GLOBAL_STEP_SIZE \
            run_mode.tune_hyperparameter=$TUNE \
            run_mode.compile_tuning_results=$RESULTS \
            tuning.hyperparameter_grid="[$HP_VAL]" \
            tuning.type=$TUNING_TYPE \
            tuning.parameter_to_tune=$PARAMETER_TO_TUNE \
            server.dp=$DP \
            server.local_updates=$LOCAL_UPDATES \
            server.sampling_rate=$SAMPLING_RATE \
            server.sigma=$SIGMA \
            server.max_grad_norm=$MAX_GRAD_NORM \
            server.global_step=$GLOBAL_STEP \
            server.local_step=$LOCAL_STEP \
            server.client_ratio=$CLIENT_RATIO \
            run_settings.rounds=$ROUNDS \
            > "$LOG" 2>&1 &
    done
    wait
done

wait
echo "All jobs finished"
