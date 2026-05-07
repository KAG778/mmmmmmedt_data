#!/bin/bash
# Smoke Test: MIMIC-IV Full Pipeline (3 representative models)
# Tests: Main model + BC (simplest baseline) + CQL (complex baseline)
# Uses: 2 epochs, 5 trajectories for eval, seed=0, GPU 1
set -e
cd "$(dirname "$0")"

GPU=1
SEED=0
EPOCHS=2
MAX_TRAJ=5

MAIN_DIR=main_model_IV/scheme3_cspdt_v2
BASELINE_DIR=baseline_Iv
DATA_TEST="data/mimic_iv_v3/test_Phys45_v3.pickle"
SMOKE_DIR="logs/IV_smoke"
mkdir -p "$SMOKE_DIR"

# Clean previous smoke test outputs
rm -rf "$MAIN_DIR/checkpoints_IV_smoke"
rm -rf "$BASELINE_DIR/results_smoke"

ts() { date +%H:%M:%S; }
FAIL=0

run_cmd() {
    local name=$1; shift
    echo ""
    echo "[$(ts)] ========== $name =========="
    echo "[$(ts)] CMD: $@"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(ts)] FAILED: $name (rc=$rc)"
        FAIL=1
        return 1
    fi
    echo "[$(ts)] PASSED: $name"
    return 0
}

echo "[$(ts)] =========================================="
echo "[$(ts)] MIMIC-IV Smoke Test (3 models, seed=$SEED)"
echo "[$(ts)] Epochs=$EPOCHS, Max_traj=$MAX_TRAJ, GPU=$GPU"
echo "[$(ts)] =========================================="

# ============================================================
# 1. Main Model: stage1 -> stage2 -> eval
# ============================================================
S1DIR="$MAIN_DIR/checkpoints_IV_smoke/seed${SEED}_stage1"
S2DIR="$MAIN_DIR/checkpoints_IV_smoke/seed${SEED}_stage2"

run_cmd "MAIN Stage1" \
    $MAIN_DIR/train_stage1_no_sem_epoch.py \
    --seed $SEED --epochs $EPOCHS --logdir "$S1DIR" \
    || { echo "MAIN Stage1 failed, skipping rest"; exit 1; }

run_cmd "MAIN Stage2" \
    $MAIN_DIR/train_stage2_no_sem_sigma2.py \
    --seed $SEED --epochs $EPOCHS --selfplay_iterations 10 \
    --logdir "$S2DIR" \
    --policy_ckpt "$S1DIR/epoch_${EPOCHS}/policy.pt" \
    --world_model_ckpt "$S1DIR/epoch_${EPOCHS}/world_model.pt" \
    || { echo "MAIN Stage2 failed, skipping eval"; exit 1; }

run_cmd "MAIN Eval" \
    $MAIN_DIR/stratified_rollout_no_sem_wm.py \
    --seed $SEED --max_traj $MAX_TRAJ \
    --checkpoint "$S2DIR/best_checkpoint.pt" \
    --data "$DATA_TEST" \
    --output "$MAIN_DIR/results/smoke_seed${SEED}_eval.json" \
    || true  # eval failure is non-fatal for smoke test

# ============================================================
# 2. BC Baseline: stage1 -> stage2 -> eval
# ============================================================
BC_S1="$BASELINE_DIR/results_smoke/bc/seed${SEED}_stage1"
BC_S2="$BASELINE_DIR/results_smoke/bc/seed${SEED}_stage2"

run_cmd "BC Stage1" \
    $BASELINE_DIR/train/train_stage1.py \
    --config "$BASELINE_DIR/configs/bc_sigma2.yaml" \
    --seed $SEED --epochs $EPOCHS --output_dir "$BC_S1" \
    || true

run_cmd "BC Stage2" \
    $BASELINE_DIR/train/train_stage2.py \
    --config "$BASELINE_DIR/configs/bc_sigma2.yaml" \
    --seed $SEED --epochs $EPOCHS \
    --checkpoint "$BC_S1/best_checkpoint.pt" --output_dir "$BC_S2" \
    || true

run_cmd "BC Eval" \
    $BASELINE_DIR/evaluate/stratified_rollout_v3v7.py \
    --config "$BASELINE_DIR/configs/bc_sigma2.yaml" \
    --seed $SEED --max_traj $MAX_TRAJ \
    --checkpoint "$BC_S2/best_checkpoint.pt" \
    --data "$DATA_TEST" \
    --output "$BASELINE_DIR/results_smoke/bc/seed${SEED}_eval.json" \
    || true

# ============================================================
# 3. CQL Baseline: stage1 -> stage2 -> eval
# ============================================================
CQL_S1="$BASELINE_DIR/results_smoke/cql/seed${SEED}_stage1"
CQL_S2="$BASELINE_DIR/results_smoke/cql/seed${SEED}_stage2"

run_cmd "CQL Stage1" \
    $BASELINE_DIR/train/train_stage1.py \
    --config "$BASELINE_DIR/configs/cql_sigma2.yaml" \
    --seed $SEED --epochs $EPOCHS --output_dir "$CQL_S1" \
    || true

run_cmd "CQL Stage2" \
    $BASELINE_DIR/train/train_stage2.py \
    --config "$BASELINE_DIR/configs/cql_sigma2.yaml" \
    --seed $SEED --epochs $EPOCHS \
    --checkpoint "$CQL_S1/best_checkpoint.pt" --output_dir "$CQL_S2" \
    || true

run_cmd "CQL Eval" \
    $BASELINE_DIR/evaluate/stratified_rollout_v3v7.py \
    --config "$BASELINE_DIR/configs/cql_sigma2.yaml" \
    --seed $SEED --max_traj $MAX_TRAJ \
    --checkpoint "$CQL_S2/best_checkpoint.pt" \
    --data "$DATA_TEST" \
    --output "$BASELINE_DIR/results_smoke/cql/seed${SEED}_eval.json" \
    || true

# ============================================================
# Summary
# ============================================================
echo ""
echo "[$(ts)] =========================================="
if [ $FAIL -eq 0 ]; then
    echo "[$(ts)] SMOKE TEST: ALL PASSED"
else
    echo "[$(ts)] SMOKE TEST: SOME FAILURES - check output above"
fi
echo "[$(ts)] =========================================="
