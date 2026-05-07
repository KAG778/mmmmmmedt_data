#!/bin/bash
# Launch all 6 channel ablation experiments
# Usage: bash run_all.sh <gpu_id>
# Each variant: Stage1 (100 epochs) -> Stage2 (50 epochs) -> Rollout (sigma=2)

set -e
cd "$(dirname "$0")"/../..
echo "Working directory: $(pwd)"

GPU=${1:?"Usage: bash run_all.sh <gpu_id>"}
VARIANTS="A1_no_task A2_no_hindsight A3_no_foresight A4_only_task A5_only_hindsight A6_only_foresight"
LOG_DIR="experiments/ablation_channel/logs"
DATA_DIR="data/v3"
EXP_DIR="experiments/ablation_channel"

mkdir -p "$LOG_DIR"

for VARIANT in $VARIANTS; do
    echo "============================================"
    echo "Starting $VARIANT on GPU $GPU"
    echo "============================================"

    # Stage 1
    echo "[$VARIANT] Starting Stage 1..."
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU python -u ${EXP_DIR}/train_stage1.py \
        --variant $VARIANT \
        --epochs 100 \
        --save_interval_epochs 10 \
        --log_interval_steps 100 \
        2>&1 | tee "${LOG_DIR}/${VARIANT}_stage1.log"

    CKPT_DIR="${EXP_DIR}/checkpoints/${VARIANT}/stage1"
    # Find latest epoch checkpoint
    BEST_STAGE1=$(ls -d ${CKPT_DIR}/epoch_* 2>/dev/null | sort -t_ -k2 -n | tail -1)

    if [ -z "$BEST_STAGE1" ]; then
        echo "[$VARIANT] ERROR: No Stage 1 checkpoint found!"
        continue
    fi

    echo "[$VARIANT] Using Stage 1 checkpoint: $BEST_STAGE1"

    # Stage 2
    echo "[$VARIANT] Starting Stage 2..."
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU python -u ${EXP_DIR}/train_stage2.py \
        --variant $VARIANT \
        --policy_ckpt "${BEST_STAGE1}/policy.pt" \
        --world_model_ckpt "${BEST_STAGE1}/world_model.pt" \
        --epochs 50 \
        --selfplay_iterations 1000 \
        --save_interval 10 \
        2>&1 | tee "${LOG_DIR}/${VARIANT}_stage2.log"

    # Rollout
    echo "[$VARIANT] Starting Rollout evaluation..."
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU python -u ${EXP_DIR}/stratified_rollout.py \
        --variant $VARIANT \
        --checkpoint "${EXP_DIR}/checkpoints/${VARIANT}/stage2/best_checkpoint.pt" \
        --data "${DATA_DIR}/test_Phys45_v3.pickle" \
        --output "${EXP_DIR}/results/${VARIANT}.json" \
        2>&1 | tee "${LOG_DIR}/${VARIANT}_rollout.log"

    echo "[$VARIANT] COMPLETE! Results at ${EXP_DIR}/results/${VARIANT}.json"
done

echo "============================================"
echo "All 6 channel ablation experiments complete!"
echo "============================================"
