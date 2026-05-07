#!/bin/bash
# Launch all 4 fusion ablation experiments on 3 GPUs
# Fusion variants only change Policy, WM reuses existing approach (no semantic)
set -e
cd "$(dirname "$0")"
BASE=$(pwd)

VARIANTS="concat residual gated cross_attn"
GPUS=(1 2 3)
DATA="/home/wangmeiyi/AuctionNet/medical/last_exp/data/v3"
TEST_DATA="$DATA/test_Phys45_v3.pickle"

echo "============================================="
echo "  Fusion Ablation — Full Pipeline"
echo "  Variants: $VARIANTS"
echo "============================================="

# ---- Stage 1: 100 epochs ----
echo ""
echo "=== Stage 1: Cold-start training ==="
for i in $(seq 0 3); do
    v=$(echo $VARIANTS | cut -d' ' -f$((i+1)))
    GPU=${GPUS[$((i % 3))]}
    CKPT="$BASE/checkpoints_fusion/$v/stage1"
    LOG="$BASE/logs_fusion/${v}_stage1.log"
    mkdir -p "$CKPT"

    echo "  [$v] GPU=$GPU, epochs=100"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU python -u "$BASE/train_stage1_fusion.py" \
        --fusion $v \
        --epochs 100 \
        --logdir "$CKPT" \
        > "$LOG" 2>&1 &
    PID_S1[$i]=$!
    echo "  [$v] Stage 1 PID=${PID_S1[$i]}, log=$LOG"
done

echo "Waiting for all Stage 1 jobs to finish..."
for pid in ${PID_S1[@]}; do
    wait $pid || echo "Warning: PID $pid exited with non-zero status"
done
echo "=== Stage 1 complete ==="

# ---- Stage 2: 50 epochs ----
echo ""
echo "=== Stage 2: Counterfactual Self-Play ==="
for i in $(seq 0 3); do
    v=$(echo $VARIANTS | cut -d' ' -f$((i+1)))
    GPU=${GPUS[$((i % 3))]}
    CKPT_S1="$BASE/checkpoints_fusion/$v/stage1/epoch_100"
    CKPT_S2="$BASE/checkpoints_fusion/$v/stage2"
    LOG="$BASE/logs_fusion/${v}_stage2.log"
    mkdir -p "$CKPT_S2"

    if [ ! -f "$CKPT_S1/policy.pt" ]; then
        echo "  [$v] Stage 1 checkpoint not found at $CKPT_S1 — skipping Stage 2"
        continue
    fi

    echo "  [$v] GPU=$GPU, epochs=50"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU python -u "$BASE/train_stage2_fusion.py" \
        --fusion $v \
        --policy_ckpt "$CKPT_S1/policy.pt" \
        --world_model_ckpt "$CKPT_S1/world_model.pt" \
        --logdir "$CKPT_S2" \
        --epochs 50 \
        > "$LOG" 2>&1 &
    PID_S2[$i]=$!
    echo "  [$v] Stage 2 PID=${PID_S2[$i]}, log=$LOG"
done

echo "Waiting for all Stage 2 jobs to finish..."
for pid in ${PID_S2[@]}; do
    wait $pid 2>/dev/null || true
done
echo "=== Stage 2 complete ==="

# ---- Rollout Evaluation ----
echo ""
echo "=== Rollout Evaluation ==="
for i in $(seq 0 3); do
    v=$(echo $VARIANTS | cut -d' ' -f$((i+1)))
    GPU=${GPUS[$((i % 3))]}
    CKPT_S2="$BASE/checkpoints_fusion/$v/stage2/best_checkpoint.pt"
    RESULT="$BASE/results_fusion/$v"
    LOG="$BASE/logs_fusion/${v}_rollout.log"
    mkdir -p "$RESULT"

    if [ ! -f "$CKPT_S2" ]; then
        echo "  [$v] Stage 2 best checkpoint not found at $CKPT_S2 — skipping rollout"
        continue
    fi

    echo "  [$v] GPU=$GPU"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU python -u "$BASE/stratified_rollout_fusion.py" \
        --fusion $v \
        --checkpoint "$CKPT_S2" \
        --data "$TEST_DATA" \
        --output "$RESULT/rollout.json" \
        > "$LOG" 2>&1 &
    PID_RO[$i]=$!
    echo "  [$v] Rollout PID=${PID_RO[$i]}, log=$LOG"
done

echo "Waiting for all Rollout jobs to finish..."
for pid in ${PID_RO[@]}; do
    wait $pid 2>/dev/null || true
done
echo "=== Rollout complete ==="

echo ""
echo "============================================="
echo "  All Fusion Ablation experiments finished!"
echo "============================================="
echo "Results:"
for v in $VARIANTS; do
    RESULT="$BASE/results_fusion/$v/rollout.json"
    if [ -f "$RESULT" ]; then
        echo "  $v: $RESULT"
        python -c "import json; d=json.load(open('$RESULT')); print(f'    Overall: {d[\"overall\"][\"mean\"]:.4f} +/- {d[\"overall\"][\"std\"]:.4f}')" 2>/dev/null || true
    else
        echo "  $v: NOT FOUND"
    fi
done
