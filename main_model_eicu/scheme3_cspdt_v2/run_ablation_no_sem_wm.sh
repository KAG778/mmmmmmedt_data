#!/bin/bash
# Fully automated ablation study pipeline: Stage1 -> Stage2 -> Evaluation
# Policy WITH semantic, World Model WITHOUT semantic
# Adapted for eICU dataset

set -e

BASEDIR="/home/wangmeiyi/AuctionNet/medical/last_exp/main_model_eicu/scheme3_cspdt_v2"
DATA_DIR="/home/wangmeiyi/AuctionNet/medical/last_exp/data/eicu_v3"
LOG_DIR="$BASEDIR/logs/ablation_no_sem_wm"
RESULT_DIR="$BASEDIR/results/ablation_no_sem_wm"
CKPT_BASE="$BASEDIR/checkpoints_no_sem_wm"

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$CKPT_BASE"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "Ablation Study: No Semantic WM (eICU)"
echo "Policy: WITH semantic"
echo "World Model: WITHOUT semantic"
echo "=========================================="
echo ""

# Function to find latest checkpoint
find_latest_checkpoint() {
    local ckpt_dir=$1
    local latest_epoch=$(ls -d "$ckpt_dir"/epoch_* 2>/dev/null | sed 's/.*epoch_//' | sort -n | tail -1)
    if [ -n "$latest_epoch" ]; then
        echo "$ckpt_dir/epoch_$latest_epoch"
    else
        echo ""
    fi
}

# Function to wait for checkpoint
wait_for_checkpoint() {
    local ckpt_path=$1
    local timeout=7200
    local elapsed=0

    echo "Waiting for checkpoint: $ckpt_path"
    while [ ! -f "$ckpt_path/policy.pt" ] || [ ! -f "$ckpt_path/world_model.pt" ]; do
        if [ $elapsed -ge $timeout ]; then
            echo -e "${RED}ERROR: Timeout waiting for checkpoint${NC}"
            return 1
        fi
        sleep 30
        elapsed=$((elapsed + 30))
    done
    echo -e "${GREEN}Checkpoint ready: $ckpt_path${NC}"
    return 0
}

# ============================================================
# Phase 1: Stage 1 Training
# ============================================================
echo ""
echo "=========================================="
echo "Phase 1: Stage 1 Training (No Sem WM)"
echo "=========================================="
echo ""

echo -e "${BLUE}Starting CSP-DT Stage1 (No Sem WM)...${NC}"
python "$BASEDIR/train_stage1_no_sem_epoch.py" \
    --datadir "$DATA_DIR" \
    --epochs 100 \
    --save_interval_epochs 10 \
    --log_interval_steps 100 \
    --logdir "$CKPT_BASE/stage1" \
    > "$LOG_DIR/stage1.log" 2>&1

STAGE1_EXIT=$?
if [ $STAGE1_EXIT -eq 0 ]; then
    echo -e "${GREEN}[OK] Stage1 completed successfully${NC}"
else
    echo -e "${RED}[FAIL] Stage1 failed with exit code $STAGE1_EXIT${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo "Phase 1 Complete"
echo "=========================================="

# ============================================================
# Phase 2: Stage 2 Training
# ============================================================
echo ""
echo "=========================================="
echo "Phase 2: Stage 2 Training (No Sem WM)"
echo "=========================================="
echo ""

# Find and validate Stage1 checkpoint
STAGE1_CKPT=$(find_latest_checkpoint "$CKPT_BASE/stage1")
if [ -z "$STAGE1_CKPT" ]; then
    echo -e "${RED}ERROR: No Stage1 checkpoint found${NC}"
    exit 1
fi

echo -e "${BLUE}Using Stage1 checkpoint: $STAGE1_CKPT${NC}"
wait_for_checkpoint "$STAGE1_CKPT" || exit 1

# Check if train_stage2_no_sem_epoch.py exists
if [ ! -f "$BASEDIR/train_stage2_no_sem_epoch.py" ]; then
    echo -e "${RED}ERROR: train_stage2_no_sem_epoch.py not found${NC}"
    exit 1
fi

echo -e "${BLUE}Starting CSP-DT Stage2 (No Sem WM)...${NC}"
python "$BASEDIR/train_stage2_no_sem_epoch.py" \
    --datadir "$DATA_DIR" \
    --policy_ckpt "$STAGE1_CKPT/policy.pt" \
    --world_model_ckpt "$STAGE1_CKPT/world_model.pt" \
    --epochs 50 \
    --selfplay_iterations 1000 \
    --logdir "$CKPT_BASE/stage2" \
    > "$LOG_DIR/stage2.log" 2>&1

STAGE2_EXIT=$?
if [ $STAGE2_EXIT -eq 0 ]; then
    echo -e "${GREEN}[OK] Stage2 completed successfully${NC}"
else
    echo -e "${RED}[FAIL] Stage2 failed with exit code $STAGE2_EXIT${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo "Phase 2 Complete"
echo "=========================================="

# ============================================================
# Phase 3: Evaluation
# ============================================================
echo ""
echo "=========================================="
echo "Phase 3: Evaluation (No Sem WM)"
echo "=========================================="
echo ""

# Find Stage2 checkpoint (try best first, then latest)
if [ -f "$CKPT_BASE/stage2/best_checkpoint.pt" ]; then
    STAGE2_CKPT="$CKPT_BASE/stage2/best_checkpoint.pt"
else
    STAGE2_CKPT=$(find_latest_checkpoint "$CKPT_BASE/stage2")
fi

if [ -z "$STAGE2_CKPT" ]; then
    echo -e "${RED}ERROR: No Stage2 checkpoint found${NC}"
    exit 1
fi

echo -e "${BLUE}Using Stage2 checkpoint: $STAGE2_CKPT${NC}"

echo -e "${BLUE}Evaluating CSP-DT (No Sem WM)...${NC}"
python "$BASEDIR/stratified_rollout_no_sem_wm.py" \
    --checkpoint "$STAGE2_CKPT" \
    --data "$DATA_DIR/test_Phys45_v3.pickle" \
    --output "$RESULT_DIR/cspdt_no_sem_wm_results.json" \
    > "$LOG_DIR/evaluation.log" 2>&1

EVAL_EXIT=$?
if [ $EVAL_EXIT -eq 0 ]; then
    echo -e "${GREEN}[OK] Evaluation completed successfully${NC}"
else
    echo -e "${RED}[FAIL] Evaluation failed with exit code $EVAL_EXIT${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo "Ablation Study Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Stage1 checkpoint: $STAGE1_CKPT"
echo "  Stage2 checkpoint: $STAGE2_CKPT"
echo "  Logs: $LOG_DIR"
echo "  Results: $RESULT_DIR/cspdt_no_sem_wm_results.json"
echo ""
echo "Configuration:"
echo "  Dataset: eICU ($DATA_DIR)"
echo "  Policy: WITH semantic embeddings"
echo "  World Model: WITHOUT semantic embeddings"
echo ""

# Display results if available
if [ -f "$RESULT_DIR/cspdt_no_sem_wm_results.json" ]; then
    echo "Evaluation Results:"
    cat "$RESULT_DIR/cspdt_no_sem_wm_results.json"
fi
