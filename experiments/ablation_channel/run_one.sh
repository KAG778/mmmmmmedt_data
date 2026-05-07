#!/bin/bash
# Launch a single channel ablation variant
# Usage: bash run_one.sh <variant> <gpu_id>
# Example: bash run_one.sh A1_no_task 1

set -e
cd "$(dirname "$0")"/../..

VARIANT=${1:?"Usage: bash run_one.sh <variant> <gpu_id>"}
GPU=${2:?"Usage: bash run_one.sh <variant> <gpu_id>"}

LOG_DIR="experiments/ablation_channel/logs"
DATA_DIR="data/v3"
EXP_DIR="experiments/ablation_channel"

mkdir -p "$LOG_DIR"

echo "[$VARIANT] Starting on GPU $GPU"

# Stage 1
echo "[$VARIANT] Stage 1..."
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU nohup python -u ${EXP_DIR}/train_stage1.py \
    --variant $VARIANT --epochs 100 --save_interval_epochs 10 --log_interval_steps 100 \
    > "${LOG_DIR}/${VARIANT}_stage1.log" 2>&1 &
S1_PID=$!
echo "[$VARIANT] Stage 1 PID=$S1_PID"

wait $S1_PID
echo "[$VARIANT] Stage 1 done"

# Find best checkpoint
CKPT_DIR="${EXP_DIR}/checkpoints/${VARIANT}/stage1"
BEST_STAGE1=$(ls -d ${CKPT_DIR}/epoch_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
echo "[$VARIANT] Using checkpoint: $BEST_STAGE1"

# Stage 2
echo "[$VARIANT] Stage 2..."
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU nohup python -u ${EXP_DIR}/train_stage2.py \
    --variant $VARIANT \
    --policy_ckpt "${BEST_STAGE1}/policy.pt" \
    --world_model_ckpt "${BEST_STAGE1}/world_model.pt" \
    --epochs 50 --selfplay_iterations 1000 --save_interval 10 \
    > "${LOG_DIR}/${VARIANT}_stage2.log" 2>&1 &
S2_PID=$!
echo "[$VARIANT] Stage 2 PID=$S2_PID"

wait $S2_PID
echo "[$VARIANT] Stage 2 done"

# Rollout
echo "[$VARIANT] Rollout..."
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU python -u ${EXP_DIR}/stratified_rollout.py \
    --variant $VARIANT \
    --checkpoint "${EXP_DIR}/checkpoints/${VARIANT}/stage2/best_checkpoint.pt" \
    --data "${DATA_DIR}/test_Phys45_v3.pickle" \
    --output "${EXP_DIR}/results/${VARIANT}.json" \
    2>&1 | tee "${LOG_DIR}/${VARIANT}_rollout.log"

echo "[$VARIANT] COMPLETE!"
