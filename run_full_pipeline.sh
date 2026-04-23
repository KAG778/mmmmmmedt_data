#!/bin/bash
# Fully automated training pipeline: Stage1 → Stage2 → Evaluation
# Ensures correct checkpoint usage and automatic progression

set -e  # Exit on error

LOG_DIR="/tmp/parallel_training_logs"
RESULT_DIR="/tmp/evaluation_results"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "Fully Automated Training Pipeline"
echo "Stage1 → Stage2 → Evaluation"
echo "=========================================="
echo ""

# Function to find latest epoch checkpoint
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

# Backup old step-based checkpoints
CSPDT_CKPT_DIR="/home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage1"
if ls "$CSPDT_CKPT_DIR"/step_* 1> /dev/null 2>&1; then
    echo -e "${YELLOW}Backing up old step-based checkpoints...${NC}"
    mkdir -p "$CSPDT_CKPT_DIR/old_step_based_backup"
    mv "$CSPDT_CKPT_DIR"/step_* "$CSPDT_CKPT_DIR/old_step_based_backup/" 2>/dev/null || true
    echo "Backup complete"
fi

# ============================================================
# Phase 1: Stage 1 Training
# ============================================================
echo ""
echo "=========================================="
echo "Phase 1: Stage 1 Training"
echo "=========================================="
echo ""

declare -A stage1_pids
declare -A stage1_status

# Batch 1: CSP-DT, BC, IQL, CQL
echo -e "${BLUE}Starting batch 1...${NC}"
CUDA_VISIBLE_DEVICES=0 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/train_stage1.py \
    --epochs 100 --save_interval_epochs 10 --log_interval_steps 100 \
    --logdir "$CSPDT_CKPT_DIR" \
    > "$LOG_DIR/cspdt_stage1.log" 2>&1 &
stage1_pids["cspdt"]=$!

CUDA_VISIBLE_DEVICES=1 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/bc.yaml \
    > "$LOG_DIR/bc_stage1.log" 2>&1 &
stage1_pids["bc"]=$!

CUDA_VISIBLE_DEVICES=2 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/iql.yaml \
    > "$LOG_DIR/iql_stage1.log" 2>&1 &
stage1_pids["iql"]=$!

CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/cql.yaml \
    > "$LOG_DIR/cql_stage1.log" 2>&1 &
stage1_pids["cql"]=$!

for model in cspdt bc iql cql; do
    wait ${stage1_pids[$model]}
    stage1_status[$model]=$?
    if [ ${stage1_status[$model]} -eq 0 ]; then
        echo -e "${GREEN}[✓] $model Stage1 done${NC}"
    else
        echo -e "${RED}[✗] $model Stage1 failed (exit ${stage1_status[$model]})${NC}"
    fi
done

# Batch 2: DT, BCQ, DQN
echo -e "${BLUE}Starting batch 2...${NC}"
CUDA_VISIBLE_DEVICES=1 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/dt.yaml \
    > "$LOG_DIR/dt_stage1.log" 2>&1 &
stage1_pids["dt"]=$!

CUDA_VISIBLE_DEVICES=2 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/bcq.yaml \
    > "$LOG_DIR/bcq_stage1.log" 2>&1 &
stage1_pids["bcq"]=$!

CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/dqn.yaml \
    > "$LOG_DIR/dqn_stage1.log" 2>&1 &
stage1_pids["dqn"]=$!

for model in dt bcq dqn; do
    wait ${stage1_pids[$model]}
    stage1_status[$model]=$?
    if [ ${stage1_status[$model]} -eq 0 ]; then
        echo -e "${GREEN}[✓] $model Stage1 done${NC}"
    else
        echo -e "${RED}[✗] $model Stage1 failed (exit ${stage1_status[$model]})${NC}"
    fi
done

# Batch 3: TD3BC
echo -e "${BLUE}Starting batch 3...${NC}"
CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/td3bc.yaml \
    > "$LOG_DIR/td3bc_stage1.log" 2>&1 &
stage1_pids["td3bc"]=$!

wait ${stage1_pids["td3bc"]}
stage1_status["td3bc"]=$?
if [ ${stage1_status["td3bc"]} -eq 0 ]; then
    echo -e "${GREEN}[✓] td3bc Stage1 done${NC}"
else
    echo -e "${RED}[✗] td3bc Stage1 failed (exit ${stage1_status["td3bc"]})${NC}"
fi

echo ""
echo "=========================================="
echo "Phase 1 Complete"
echo "=========================================="

# ============================================================
# Phase 2: Stage 2 Training (with checkpoint validation)
# ============================================================
echo ""
echo "=========================================="
echo "Phase 2: Stage 2 Training"
echo "=========================================="
echo ""

# Find and validate CSP-DT checkpoint
CSPDT_LATEST_CKPT=$(find_latest_checkpoint "$CSPDT_CKPT_DIR")
if [ -z "$CSPDT_LATEST_CKPT" ]; then
    echo -e "${RED}ERROR: No epoch checkpoint found for CSP-DT${NC}"
    exit 1
fi

echo -e "${BLUE}Using CSP-DT checkpoint: $CSPDT_LATEST_CKPT${NC}"
wait_for_checkpoint "$CSPDT_LATEST_CKPT" || exit 1

declare -A stage2_pids
declare -A stage2_status

# Batch 1: CSP-DT, BC, IQL, CQL
echo -e "${BLUE}Starting Stage2 batch 1...${NC}"
CUDA_VISIBLE_DEVICES=0 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/train_stage2.py \
    --policy_ckpt "$CSPDT_LATEST_CKPT/policy.pt" \
    --world_model_ckpt "$CSPDT_LATEST_CKPT/world_model.pt" \
    --epochs 50 --selfplay_iterations 1000 \
    --logdir /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage2 \
    > "$LOG_DIR/cspdt_stage2.log" 2>&1 &
stage2_pids["cspdt"]=$!

CUDA_VISIBLE_DEVICES=1 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/bc.yaml \
    > "$LOG_DIR/bc_stage2.log" 2>&1 &
stage2_pids["bc"]=$!

CUDA_VISIBLE_DEVICES=2 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/iql.yaml \
    > "$LOG_DIR/iql_stage2.log" 2>&1 &
stage2_pids["iql"]=$!

CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/cql.yaml \
    > "$LOG_DIR/cql_stage2.log" 2>&1 &
stage2_pids["cql"]=$!

for model in cspdt bc iql cql; do
    wait ${stage2_pids[$model]}
    stage2_status[$model]=$?
    if [ ${stage2_status[$model]} -eq 0 ]; then
        echo -e "${GREEN}[✓] $model Stage2 done${NC}"
    else
        echo -e "${RED}[✗] $model Stage2 failed (exit ${stage2_status[$model]})${NC}"
    fi
done

# Batch 2: DT, BCQ, DQN
echo -e "${BLUE}Starting Stage2 batch 2...${NC}"
CUDA_VISIBLE_DEVICES=1 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/dt.yaml \
    > "$LOG_DIR/dt_stage2.log" 2>&1 &
stage2_pids["dt"]=$!

CUDA_VISIBLE_DEVICES=2 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/bcq.yaml \
    > "$LOG_DIR/bcq_stage2.log" 2>&1 &
stage2_pids["bcq"]=$!

CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/dqn.yaml \
    > "$LOG_DIR/dqn_stage2.log" 2>&1 &
stage2_pids["dqn"]=$!

for model in dt bcq dqn; do
    wait ${stage2_pids[$model]}
    stage2_status[$model]=$?
    if [ ${stage2_status[$model]} -eq 0 ]; then
        echo -e "${GREEN}[✓] $model Stage2 done${NC}"
    else
        echo -e "${RED}[✗] $model Stage2 failed (exit ${stage2_status[$model]})${NC}"
    fi
done

# Batch 3: TD3BC
echo -e "${BLUE}Starting Stage2 batch 3...${NC}"
CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/td3bc.yaml \
    > "$LOG_DIR/td3bc_stage2.log" 2>&1 &
stage2_pids["td3bc"]=$!

wait ${stage2_pids["td3bc"]}
stage2_status["td3bc"]=$?
if [ ${stage2_status["td3bc"]} -eq 0 ]; then
    echo -e "${GREEN}[✓] td3bc Stage2 done${NC}"
else
    echo -e "${RED}[✗] td3bc Stage2 failed (exit ${stage2_status["td3bc"]})${NC}"
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
echo "Phase 3: Evaluation"
echo "=========================================="
echo ""

# Evaluate CSP-DT
echo -e "${BLUE}Evaluating CSP-DT...${NC}"
CSPDT_STAGE2_CKPT=$(find_latest_checkpoint "/home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage2")
if [ -n "$CSPDT_STAGE2_CKPT" ]; then
    python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/stratified_rollout_v3v7.py \
        --checkpoint "$CSPDT_STAGE2_CKPT" \
        --data /home/wangmeiyi/AuctionNet/medical/重构之后的代码/data/v3/test_Phys45_v3.pickle \
        --output "$RESULT_DIR/cspdt_results.json" \
        > "$LOG_DIR/cspdt_eval.log" 2>&1
    echo -e "${GREEN}[✓] CSP-DT evaluation done${NC}"
else
    echo -e "${RED}[✗] CSP-DT Stage2 checkpoint not found${NC}"
fi

# Evaluate baselines (if evaluation scripts exist)
for model in bc dt iql bcq cql dqn td3bc; do
    echo -e "${BLUE}Evaluating $model...${NC}"
    # Add baseline evaluation commands here when available
    echo -e "${YELLOW}[⊙] $model evaluation script not configured${NC}"
done

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  CSP-DT checkpoint: $CSPDT_LATEST_CKPT"
echo "  Logs: $LOG_DIR"
echo "  Results: $RESULT_DIR"
echo ""
echo "Stage1 Status:"
for model in cspdt bc dt iql bcq cql dqn td3bc; do
    if [ ${stage1_status[$model]} -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} $model"
    else
        echo -e "  ${RED}✗${NC} $model (exit ${stage1_status[$model]})"
    fi
done
echo ""
echo "Stage2 Status:"
for model in cspdt bc dt iql bcq cql dqn td3bc; do
    if [ ${stage2_status[$model]} -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} $model"
    else
        echo -e "  ${RED}✗${NC} $model (exit ${stage2_status[$model]})"
    fi
done
echo ""
