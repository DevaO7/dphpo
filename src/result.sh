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

LOCAL_UPDATES=50
SAMPLING_RATE=0.2
SIGMA=20.0
MAX_GRAD_NORM=2.0

DP=True
ROUNDS=150
SIMILARITY=null
CLIENT_RATIO=0.21
RESULTS=True
TUNE=False
GPU=5
LOG_PREFIX="0"
TUNING_METHOD=cross_validation
MIN_RESOURCE=10
ELIMINATION_RATE=2
ALPHA=1.0
BETA=1.0

# HYPERPARAMETER="[0.0025,0.005,0.01,0.02,0.04,0.08]"
# HYPERPARAMETER="[0.00125,0.0025,0.005,0.01,0.02]"
# HYPERPARAMETER="[1.28]"
# HYPERPARAMETER="[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5]"
# HYPERPARAMETER="[2.0,2.5,3.0]"
GLOBAL_STEP_SIZE=Fixed
LOCAL_STEP_SIZE=0.01
PARAMETER_TO_TUNE="clipping"
if [ "$PARAMETER_TO_TUNE" == "step_size" ]; then
    # HYPERPARAMETER="[0.00125,0.0025,0.005,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28]"
    # HYPERPARAMETER="[0.08,0.16,0.32,0.64,1.28]"
    HYPERPARAMETER="[0.005,0.01,0.02,0.04,0.08,0.16]"
elif [ "$PARAMETER_TO_TUNE" == "clipping" ]; then
    HYPERPARAMETER="[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5]"
    # HYPERPARAMETER="[0.5,1.0,1.5,2.0,2.5,3.0,3.5]"
    
fi
TRANSFER_MODE=local_updates
if [ "$TRANSFER_MODE" == "local_updates" ]; then
    TRANSFER_PARAMETERS="[2,6,12,22,32,50]"
elif [ "$TRANSFER_MODE" == "client_ratio" ]; then
    TRANSFER_PARAMETERS="[0.02,0.04,0.06,0.08,0.1,0.21]"
elif [ "$TRANSFER_MODE" == "sampling_rate" ]; then
    TRANSFER_PARAMETERS="[0.035,0.0675,0.1,0.14,0.16,0.2]"
elif [ "$TRANSFER_MODE" == "sigma" ]; then
    TRANSFER_PARAMETERS="[110.0,60.0,40.0,30.0,25.0,20.0]"
elif [ "$TRANSFER_MODE" == "rounds" ]; then
    TRANSFER_PARAMETERS="[5,17,38,65,95,150]"
fi


CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python main.py \
                        server.constant_global_step=$GLOBAL_STEP_SIZE \
                        run_mode.tune_hyperparameter=$TUNE \
                        run_mode.compile_tuning_results=$RESULTS \
                        tuning.hyperparameter_grid=$HYPERPARAMETER \
                        server.dp=$DP \
                        server.local_updates=$LOCAL_UPDATES \
                        server.sampling_rate=$SAMPLING_RATE \
                        server.sigma=$SIGMA \
                        server.max_grad_norm=$MAX_GRAD_NORM \
                        server.client_ratio=$CLIENT_RATIO \
                        tuning.parameter_to_tune=$PARAMETER_TO_TUNE \
                        tuning.type=$TUNING_METHOD \
                        tuning.min_resource=$MIN_RESOURCE \
                        tuning.elimination_rate=$ELIMINATION_RATE \
                        results.transfer_parameters=$TRANSFER_PARAMETERS \
                        results.transfer_mode=$TRANSFER_MODE \
                        dataset.alpha=$ALPHA \
                        dataset.beta=$BETA \
                        run_settings.rounds=$ROUNDS \
                        dataset.similarity=$SIMILARITY \
                        server.local_step=$LOCAL_STEP_SIZE \

wait
echo "All jobs finished"
