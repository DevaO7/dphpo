#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p logs
trap 'echo; echo "Stopping all background jobs..."; kill 0' SIGINT SIGTERM

CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.02 > logs/client_ratio_0.02 2>&1 &
CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.04 > logs/client_ratio_0.04 2>&1 &
CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.06 > logs/client_ratio_0.06 2>&1 &
CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.08 > logs/client_ratio_0.08 2>&1 &
CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.1 > logs/client_ratio_0.1 2>&1 &
CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.13 > logs/client_ratio_0.13 2>&1 &
CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.15 > logs/client_ratio_0.15 2>&1 &
CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.17 > logs/client_ratio_0.17 2>&1 &
CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 python main.py server.client_ratio=0.21 > logs/client_ratio_0.21 2>&1 &

wait
echo "All jobs finished"


# CUDA_VISIBLE_DEVICES=7 python main.py 