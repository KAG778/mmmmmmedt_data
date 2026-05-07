#!/bin/bash
# Auto-run pipeline: Stage1 -> Stage2 -> Eval (No-Semantic WM variant)
# Adapted for eICU dataset
set -e

BASEDIR="/home/wangmeiyi/AuctionNet/medical/last_exp/main_model_eicu/scheme3_cspdt_v2"
cd "$BASEDIR"

export CUDA_VISIBLE_DEVICES=0

LOGDIR="$BASEDIR/logs"
mkdir -p "$LOGDIR"

echo "========== Stage 1 (No-Sem WM) =========="
python train_stage1_no_sem_epoch.py \
    --datadir /home/wangmeiyi/AuctionNet/medical/last_exp/data/eicu_v3 \
    --logdir "$BASEDIR/checkpoints_no_sem_wm/stage1" \
    2>&1 | tee "$LOGDIR/stage1_no_sem.log"

echo "========== Stage 2 (No-Sem WM) =========="
# Find latest stage1 checkpoint
STAGE1_CKPT=$(ls -d "$BASEDIR"/checkpoints_no_sem_wm/stage1/epoch_* 2>/dev/null | sed 's/.*epoch_//' | sort -n | tail -1)
if [ -n "$STAGE1_CKPT" ]; then
    STAGE1_CKPT="$BASEDIR/checkpoints_no_sem_wm/stage1/epoch_$STAGE1_CKPT"
    echo "Using Stage1 checkpoint: $STAGE1_CKPT"
    python train_stage2_no_sem_epoch.py \
        --datadir /home/wangmeiyi/AuctionNet/medical/last_exp/data/eicu_v3 \
        --policy_ckpt "$STAGE1_CKPT/policy.pt" \
        --world_model_ckpt "$STAGE1_CKPT/world_model.pt" \
        --logdir "$BASEDIR/checkpoints_no_sem_wm/stage2" \
        2>&1 | tee "$LOGDIR/stage2_no_sem.log"
else
    echo "ERROR: No Stage1 checkpoint found"
    exit 1
fi

echo "========== Eval Stage1 only =========="
python stratified_rollout_no_sem.py \
    --checkpoint "$STAGE1_CKPT" \
    --data /home/wangmeiyi/AuctionNet/medical/last_exp/data/eicu_v3/test_Phys45_v3.pickle \
    --output "$BASEDIR/results/no_sem/eval_stage1.json" 2>&1 | tee "$LOGDIR/eval_stage1_no_sem.log"

echo "========== Eval Stage2 best =========="
STAGE2_CKPT="$BASEDIR/checkpoints_no_sem_wm/stage2/best_checkpoint.pt"
if [ -f "$STAGE2_CKPT" ]; then
    python stratified_rollout_no_sem.py \
        --checkpoint "$STAGE2_CKPT" \
        --data /home/wangmeiyi/AuctionNet/medical/last_exp/data/eicu_v3/test_Phys45_v3.pickle \
        --output "$BASEDIR/results/no_sem/eval_stage2_best.json" 2>&1 | tee "$LOGDIR/eval_stage2_no_sem.log"
else
    echo "WARN: best_checkpoint.pt not found, trying latest epoch"
    STAGE2_CKPT_DIR=$(ls -d "$BASEDIR"/checkpoints_no_sem_wm/stage2/epoch_* 2>/dev/null | sort | tail -1)
    if [ -n "$STAGE2_CKPT_DIR" ]; then
        python stratified_rollout_no_sem.py \
            --checkpoint "$STAGE2_CKPT_DIR" \
            --data /home/wangmeiyi/AuctionNet/medical/last_exp/data/eicu_v3/test_Phys45_v3.pickle \
            --output "$BASEDIR/results/no_sem/eval_stage2_best.json" 2>&1 | tee "$LOGDIR/eval_stage2_no_sem.log"
    fi
fi

echo "========== ALL DONE =========="
