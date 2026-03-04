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

SWEEP_PARAM=sigma

SIGMA_LIST=(25.0 60.0 110.0)
ROUNDS_LIST=(50 100 150 200)
LOCAL_UPDATES_LIST=(10 25 50 100)
SAMPLING_RATE_LIST=(0.05 0.1 0.2 0.4)
CLIENT_RATIO_LIST=(0.02 0.04 0.06 0.08 0.1 0.13 0.15 0.17 0.21)

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
TUNING_TYPE=cross_validation
TUNING_PARAMETER=step_size

# ─── Defaults for overridable settings ────────────────────────────────────────
DEFAULT_GPU=7
DEFAULT_HYPERPARAMETER="[0.00125,0.0025,0.005,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28]"
DEFAULT_RESUME=False

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
    # ["50.0"]="[0.0025,0.005]"
    # ["0.02"]="[0.005,0.01,0.02]"
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
    GPU="${OVERRIDE_GPU[$VAL]:-$DEFAULT_GPU}"
    HYPERPARAMETER="${OVERRIDE_HYPERPARAMETER[$VAL]:-$DEFAULT_HYPERPARAMETER}"
    RESUME="${OVERRIDE_RESUME[$VAL]:-$DEFAULT_RESUME}"

    LOG="logs/${SWEEP_PARAM}_${VAL}"

    echo "Launching: ${SWEEP_PARAM}=${VAL} | sigma=${SIGMA} rounds=${ROUNDS} local_updates=${LOCAL_UPDATES} sampling_rate=${SAMPLING_RATE} client_ratio=${CLIENT_RATIO} | gpu=${GPU} resume=${RESUME}"

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
        run_settings.rounds=$ROUNDS \
        > "$LOG" 2>&1 &

done

wait
echo "All jobs finished"
