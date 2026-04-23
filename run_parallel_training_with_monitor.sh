#!/bin/bash
# Parallel training with real-time progress monitoring
# Stage1 and Stage2 are SEQUENTIAL (Stage2 depends on Stage1 checkpoints)

LOG_DIR="/tmp/parallel_training_logs"
mkdir -p "$LOG_DIR"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Parallel Training with Progress Monitor"
echo "=========================================="
echo ""

# Function to check if a process is still running
is_running() {
    kill -0 "$1" 2>/dev/null
}

# Function to get last N lines from log
get_progress() {
    local log_file=$1
    if [ -f "$log_file" ]; then
        tail -3 "$log_file" | grep -E "(Epoch|step|Loss|loss)" | tail -1
    else
        echo "Log not found"
    fi
}

# Function to monitor training progress
monitor_progress() {
    local stage=$1
    shift
    local -n pids=$1
    shift
    local -n models=$1

    echo ""
    echo "=========================================="
    echo "Monitoring $stage Progress"
    echo "=========================================="
    echo ""

    while true; do
        all_done=true
        clear
        echo "=========================================="
        echo "$stage Training Progress"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        echo ""

        for model in "${models[@]}"; do
            local pid=${pids[$model]}
            local log_file="$LOG_DIR/${model}_${stage}.log"

            if [ -n "$pid" ] && is_running "$pid"; then
                all_done=false
                echo -e "${YELLOW}[RUNNING]${NC} $model (PID: $pid)"
                echo "  $(get_progress "$log_file")"
            elif [ -n "$pid" ]; then
                wait "$pid" 2>/dev/null
                exit_code=$?
                if [ $exit_code -eq 0 ]; then
                    echo -e "${GREEN}[✓ DONE]${NC} $model"
                else
                    echo -e "${RED}[✗ FAILED]${NC} $model (exit code: $exit_code)"
                fi
            fi
            echo ""
        done

        if $all_done; then
            echo "=========================================="
            echo "$stage Complete!"
            echo "=========================================="
            break
        fi

        sleep 10
    done
}

# ============================================================
# Phase 1: Stage 1 Training
# ============================================================
echo ""
echo "=========================================="
echo "Phase 1: Stage 1 Training"
echo "=========================================="
echo ""

declare -A stage1_pids
stage1_models=("cspdt" "bc" "iql" "cql")

# Start first batch
echo "[GPU 0] Starting CSP-DT Stage1"
CUDA_VISIBLE_DEVICES=0 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/train_stage1.py \
    --epochs 100 --save_interval_epochs 10 --log_interval_steps 100 \
    --logdir /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage1 \
    > "$LOG_DIR/cspdt_stage1.log" 2>&1 &
stage1_pids["cspdt"]=$!

echo "[GPU 1] Starting BC Stage1"
CUDA_VISIBLE_DEVICES=1 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/bc.yaml \
    > "$LOG_DIR/bc_stage1.log" 2>&1 &
stage1_pids["bc"]=$!

echo "[GPU 2] Starting IQL Stage1"
CUDA_VISIBLE_DEVICES=2 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/iql.yaml \
    > "$LOG_DIR/iql_stage1.log" 2>&1 &
stage1_pids["iql"]=$!

echo "[GPU 3] Starting CQL Stage1"
CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/cql.yaml \
    > "$LOG_DIR/cql_stage1.log" 2>&1 &
stage1_pids["cql"]=$!

sleep 5  # Give processes time to start

# Monitor first batch
monitor_progress "stage1" stage1_pids stage1_models

# Start second batch
echo ""
echo "Starting second batch of Stage1..."
stage1_models_batch2=("dt" "bcq" "dqn")

echo "[GPU 1] Starting DT Stage1"
CUDA_VISIBLE_DEVICES=1 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/dt.yaml \
    > "$LOG_DIR/dt_stage1.log" 2>&1 &
stage1_pids["dt"]=$!

echo "[GPU 2] Starting BCQ Stage1"
CUDA_VISIBLE_DEVICES=2 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/bcq.yaml \
    > "$LOG_DIR/bcq_stage1.log" 2>&1 &
stage1_pids["bcq"]=$!

echo "[GPU 3] Starting DQN Stage1"
CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/dqn.yaml \
    > "$LOG_DIR/dqn_stage1.log" 2>&1 &
stage1_pids["dqn"]=$!

sleep 5
monitor_progress "stage1_batch2" stage1_pids stage1_models_batch2

# Start third batch
echo ""
echo "Starting third batch of Stage1..."
stage1_models_batch3=("td3bc")

echo "[GPU 3] Starting TD3BC Stage1"
CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/td3bc.yaml \
    > "$LOG_DIR/td3bc_stage1.log" 2>&1 &
stage1_pids["td3bc"]=$!

sleep 5
monitor_progress "stage1_batch3" stage1_pids stage1_models_batch3

echo ""
echo "=========================================="
echo "Phase 1 Complete - All Stage1 Done"
echo "=========================================="
echo ""

# ============================================================
# Phase 2: Stage 2 Training
# ============================================================
echo ""
echo "=========================================="
echo "Phase 2: Stage 2 Training"
echo "=========================================="
echo ""

declare -A stage2_pids
stage2_models=("cspdt" "bc" "iql" "cql")

# Start first batch
echo "[GPU 0] Starting CSP-DT Stage2"
CUDA_VISIBLE_DEVICES=0 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/train_stage2.py \
    --policy_ckpt /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage1/epoch_100/policy.pt \
    --world_model_ckpt /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage1/epoch_100/world_model.pt \
    --epochs 50 --selfplay_iterations 1000 \
    --logdir /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage2 \
    > "$LOG_DIR/cspdt_stage2.log" 2>&1 &
stage2_pids["cspdt"]=$!

echo "[GPU 1] Starting BC Stage2"
CUDA_VISIBLE_DEVICES=1 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/bc.yaml \
    > "$LOG_DIR/bc_stage2.log" 2>&1 &
stage2_pids["bc"]=$!

echo "[GPU 2] Starting IQL Stage2"
CUDA_VISIBLE_DEVICES=2 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/iql.yaml \
    > "$LOG_DIR/iql_stage2.log" 2>&1 &
stage2_pids["iql"]=$!

echo "[GPU 3] Starting CQL Stage2"
CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/cql.yaml \
    > "$LOG_DIR/cql_stage2.log" 2>&1 &
stage2_pids["cql"]=$!

sleep 5
monitor_progress "stage2" stage2_pids stage2_models

# Continue with remaining models...
stage2_models_batch2=("dt" "bcq" "dqn")

echo "[GPU 1] Starting DT Stage2"
CUDA_VISIBLE_DEVICES=1 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/dt.yaml \
    > "$LOG_DIR/dt_stage2.log" 2>&1 &
stage2_pids["dt"]=$!

echo "[GPU 2] Starting BCQ Stage2"
CUDA_VISIBLE_DEVICES=2 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/bcq.yaml \
    > "$LOG_DIR/bcq_stage2.log" 2>&1 &
stage2_pids["bcq"]=$!

echo "[GPU 3] Starting DQN Stage2"
CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/dqn.yaml \
    > "$LOG_DIR/dqn_stage2.log" 2>&1 &
stage2_pids["dqn"]=$!

sleep 5
monitor_progress "stage2_batch2" stage2_pids stage2_models_batch2

stage2_models_batch3=("td3bc")

echo "[GPU 3] Starting TD3BC Stage2"
CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/td3bc.yaml \
    > "$LOG_DIR/td3bc_stage2.log" 2>&1 &
stage2_pids["td3bc"]=$!

sleep 5
monitor_progress "stage2_batch3" stage2_pids stage2_models_batch3

echo ""
echo "=========================================="
echo "All Training Complete!"
echo "=========================================="
echo ""
echo "Logs: $LOG_DIR"
echo "Models: CSP-DT, BC, DT, IQL, BCQ, CQL, DQN, TD3BC"
echo ""
