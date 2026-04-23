#!/bin/bash
# Parallel training with checkpoint validation
# Ensures Stage2 uses the correct Stage1 checkpoints

LOG_DIR="/tmp/parallel_training_logs"
mkdir -p "$LOG_DIR"

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

# Function to wait for checkpoint to exist
wait_for_checkpoint() {
    local ckpt_path=$1
    local timeout=7200  # 2 hours
    local elapsed=0

    echo "Waiting for checkpoint: $ckpt_path"
    while [ ! -f "$ckpt_path/policy.pt" ] || [ ! -f "$ckpt_path/world_model.pt" ]; do
        if [ $elapsed -ge $timeout ]; then
            echo "ERROR: Timeout waiting for checkpoint"
            return 1
        fi
        sleep 30
        elapsed=$((elapsed + 30))
    done
    echo "Checkpoint ready: $ckpt_path"
    return 0
}

echo "=========================================="
echo "Parallel Training with Checkpoint Safety"
echo "=========================================="
echo ""

# Clean old step-based checkpoints to avoid confusion
CSPDT_CKPT_DIR="/home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage1"
echo "Checking for old step-based checkpoints..."
if ls "$CSPDT_CKPT_DIR"/step_* 1> /dev/null 2>&1; then
    echo "WARNING: Found old step-based checkpoints. Moving to backup..."
    mkdir -p "$CSPDT_CKPT_DIR/old_step_based_backup"
    mv "$CSPDT_CKPT_DIR"/step_* "$CSPDT_CKPT_DIR/old_step_based_backup/" 2>/dev/null
    echo "Old checkpoints backed up to: $CSPDT_CKPT_DIR/old_step_based_backup/"
fi

# ============================================================
# Phase 1: Stage 1 Training
# ============================================================
echo ""
echo "=========================================="
echo "Phase 1: Stage 1 Training"
echo "=========================================="
echo ""

# Start CSP-DT Stage1
echo "[GPU 0] Starting CSP-DT Stage1"
CUDA_VISIBLE_DEVICES=0 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/train_stage1.py \
    --epochs 100 --save_interval_epochs 10 --log_interval_steps 100 \
    --logdir "$CSPDT_CKPT_DIR" \
    > "$LOG_DIR/cspdt_stage1.log" 2>&1 &
CSPDT_S1_PID=$!

# Start baseline Stage1 (BC, IQL, CQL in parallel)
echo "[GPU 1] Starting BC Stage1"
CUDA_VISIBLE_DEVICES=1 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/bc.yaml \
    > "$LOG_DIR/bc_stage1.log" 2>&1 &
BC_S1_PID=$!

echo "[GPU 2] Starting IQL Stage1"
CUDA_VISIBLE_DEVICES=2 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/iql.yaml \
    > "$LOG_DIR/iql_stage1.log" 2>&1 &
IQL_S1_PID=$!

echo "[GPU 3] Starting CQL Stage1"
CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/cql.yaml \
    > "$LOG_DIR/cql_stage1.log" 2>&1 &
CQL_S1_PID=$!

echo ""
echo "Waiting for first batch to complete..."
wait $CSPDT_S1_PID && echo "[✓] CSP-DT Stage1 done" || echo "[✗] CSP-DT Stage1 failed"
wait $BC_S1_PID && echo "[✓] BC Stage1 done" || echo "[✗] BC Stage1 failed"
wait $IQL_S1_PID && echo "[✓] IQL Stage1 done" || echo "[✗] IQL Stage1 failed"
wait $CQL_S1_PID && echo "[✓] CQL Stage1 done" || echo "[✗] CQL Stage1 failed"

# Second batch
echo ""
echo "Starting second batch..."
CUDA_VISIBLE_DEVICES=1 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/dt.yaml \
    > "$LOG_DIR/dt_stage1.log" 2>&1 &
DT_S1_PID=$!

CUDA_VISIBLE_DEVICES=2 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/bcq.yaml \
    > "$LOG_DIR/bcq_stage1.log" 2>&1 &
BCQ_S1_PID=$!

CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/dqn.yaml \
    > "$LOG_DIR/dqn_stage1.log" 2>&1 &
DQN_S1_PID=$!

wait $DT_S1_PID && echo "[✓] DT Stage1 done" || echo "[✗] DT Stage1 failed"
wait $BCQ_S1_PID && echo "[✓] BCQ Stage1 done" || echo "[✗] BCQ Stage1 failed"
wait $DQN_S1_PID && echo "[✓] DQN Stage1 done" || echo "[✗] DQN Stage1 failed"

# Third batch
echo ""
echo "Starting third batch..."
CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/td3bc.yaml \
    > "$LOG_DIR/td3bc_stage1.log" 2>&1 &
TD3BC_S1_PID=$!

wait $TD3BC_S1_PID && echo "[✓] TD3BC Stage1 done" || echo "[✗] TD3BC Stage1 failed"

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
    echo "ERROR: No epoch-based checkpoint found for CSP-DT"
    exit 1
fi

echo "Using CSP-DT checkpoint: $CSPDT_LATEST_CKPT"
wait_for_checkpoint "$CSPDT_LATEST_CKPT" || exit 1

# Start Stage2 with validated checkpoint
echo "[GPU 0] Starting CSP-DT Stage2"
CUDA_VISIBLE_DEVICES=0 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/train_stage2.py \
    --policy_ckpt "$CSPDT_LATEST_CKPT/policy.pt" \
    --world_model_ckpt "$CSPDT_LATEST_CKPT/world_model.pt" \
    --epochs 50 --selfplay_iterations 1000 \
    --logdir /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage2 \
    > "$LOG_DIR/cspdt_stage2.log" 2>&1 &
CSPDT_S2_PID=$!

# Start baseline Stage2
echo "[GPU 1] Starting BC Stage2"
CUDA_VISIBLE_DEVICES=1 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/bc.yaml \
    > "$LOG_DIR/bc_stage2.log" 2>&1 &
BC_S2_PID=$!

echo "[GPU 2] Starting IQL Stage2"
CUDA_VISIBLE_DEVICES=2 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/iql.yaml \
    > "$LOG_DIR/iql_stage2.log" 2>&1 &
IQL_S2_PID=$!

echo "[GPU 3] Starting CQL Stage2"
CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/cql.yaml \
    > "$LOG_DIR/cql_stage2.log" 2>&1 &
CQL_S2_PID=$!

wait $CSPDT_S2_PID && echo "[✓] CSP-DT Stage2 done" || echo "[✗] CSP-DT Stage2 failed"
wait $BC_S2_PID && echo "[✓] BC Stage2 done" || echo "[✗] BC Stage2 failed"
wait $IQL_S2_PID && echo "[✓] IQL Stage2 done" || echo "[✗] IQL Stage2 failed"
wait $CQL_S2_PID && echo "[✓] CQL Stage2 done" || echo "[✗] CQL Stage2 failed"

# Continue with remaining models...
echo ""
echo "Starting remaining Stage2 models..."

CUDA_VISIBLE_DEVICES=1 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/dt.yaml \
    > "$LOG_DIR/dt_stage2.log" 2>&1 &
DT_S2_PID=$!

CUDA_VISIBLE_DEVICES=2 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/bcq.yaml \
    > "$LOG_DIR/bcq_stage2.log" 2>&1 &
BCQ_S2_PID=$!

CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/dqn.yaml \
    > "$LOG_DIR/dqn_stage2.log" 2>&1 &
DQN_S2_PID=$!

wait $DT_S2_PID && echo "[✓] DT Stage2 done" || echo "[✗] DT Stage2 failed"
wait $BCQ_S2_PID && echo "[✓] BCQ Stage2 done" || echo "[✗] BCQ Stage2 failed"
wait $DQN_S2_PID && echo "[✓] DQN Stage2 done" || echo "[✗] DQN Stage2 failed"

CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/td3bc.yaml \
    > "$LOG_DIR/td3bc_stage2.log" 2>&1 &
TD3BC_S2_PID=$!

wait $TD3BC_S2_PID && echo "[✓] TD3BC Stage2 done" || echo "[✗] TD3BC Stage2 failed"

echo ""
echo "=========================================="
echo "All Training Complete!"
echo "=========================================="
echo ""
echo "CSP-DT used checkpoint: $CSPDT_LATEST_CKPT"
echo "Logs: $LOG_DIR"
echo ""
