#!/bin/bash
# Auto-run pipeline: Stage1 → Stage2 → Eval (No-Semantic WM variant)
set -e

BASEDIR="/home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2"
cd "$BASEDIR"

export CUDA_VISIBLE_DEVICES=1

LOGDIR="$BASEDIR/logs"
mkdir -p "$LOGDIR"

echo "========== Stage 1 (No-Sem WM) =========="
python train_stage1_no_sem.py 2>&1 | tee "$LOGDIR/stage1_no_sem_gpu1.log"

echo "========== Stage 2 (No-Sem WM) =========="
python train_stage2_no_sem.py 2>&1 | tee "$LOGDIR/stage2_no_sem.log"

echo "========== Eval Stage1 only =========="
python stratified_rollout_no_sem.py \
    --checkpoint ./checkpoints/stage1_no_sem/step_200000 \
    --data "$BASEDIR/../../data/v3/test_Phys45_v3.pickle" \
    --output "$BASEDIR/results/no_sem/eval_stage1.json" 2>&1 | tee "$LOGDIR/eval_stage1_no_sem.log"

echo "========== Eval Stage2 best =========="
python stratified_rollout_no_sem.py \
    --checkpoint ./checkpoints/stage2_no_sem/best_checkpoint.pt \
    --data "$BASEDIR/../../data/v3/test_Phys45_v3.pickle" \
    --output "$BASEDIR/results/no_sem/eval_stage2_best.json" 2>&1 | tee "$LOGDIR/eval_stage2_no_sem.log"

echo "========== ALL DONE =========="
