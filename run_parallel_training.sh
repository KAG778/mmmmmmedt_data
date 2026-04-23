#!/bin/bash
"""
Parallel training script for CSP-DT and all baseline models.
Distributes training across 4 available GPUs.
"""

# GPU allocation:
# GPU 0: CSP-DT (Scheme3)
# GPU 1: BC + DT
# GPU 2: IQL + BCQ
# GPU 3: CQL + DQN + TD3BC

LOG_DIR="/tmp/parallel_training_logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Starting Parallel Training Pipeline"
echo "=========================================="
echo "GPU 0: CSP-DT (Scheme3)"
echo "GPU 1: BC, DT"
echo "GPU 2: IQL, BCQ"
echo "GPU 3: CQL, DQN, TD3BC"
echo "=========================================="
echo ""

# Function to run training on specific GPU
run_on_gpu() {
    local gpu_id=$1
    local model=$2
    local stage=$3
    local log_file="$LOG_DIR/${model}_${stage}.log"

    echo "[GPU $gpu_id] Starting $model $stage"
    echo "  Log: $log_file"

    if [ "$model" == "cspdt" ]; then
        # CSP-DT training
        if [ "$stage" == "stage1" ]; then
            CUDA_VISIBLE_DEVICES=$gpu_id python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/train_stage1.py \
                --epochs 100 \
                --save_interval_epochs 10 \
                --log_interval_steps 100 \
                --logdir /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage1 \
                > "$log_file" 2>&1 &
        else
            CUDA_VISIBLE_DEVICES=$gpu_id python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/train_stage2.py \
                --policy_ckpt /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage1/epoch_100/policy.pt \
                --world_model_ckpt /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage1/epoch_100/world_model.pt \
                --epochs 50 \
                --selfplay_iterations 1000 \
                --logdir /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage2 \
                > "$log_file" 2>&1 &
        fi
    else
        # Baseline training
        if [ "$stage" == "stage1" ]; then
            CUDA_VISIBLE_DEVICES=$gpu_id python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
                --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/${model}.yaml \
                > "$log_file" 2>&1 &
        else
            CUDA_VISIBLE_DEVICES=$gpu_id python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
                --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/${model}.yaml \
                > "$log_file" 2>&1 &
        fi
    fi

    echo $!  # Return PID
}

# ============================================================
# Phase 1: Stage 1 Training (all models in parallel)
# ============================================================
echo ""
echo "=========================================="
echo "Phase 1: Stage 1 Training"
echo "=========================================="
echo ""

declare -A stage1_pids

# GPU 0: CSP-DT
stage1_pids["cspdt"]=$(run_on_gpu 0 "cspdt" "stage1")

# GPU 1: BC, DT (sequential on same GPU)
stage1_pids["bc"]=$(run_on_gpu 1 "bc" "stage1")
wait ${stage1_pids["bc"]}
stage1_pids["dt"]=$(run_on_gpu 1 "dt" "stage1")

# GPU 2: IQL, BCQ (sequential on same GPU)
stage1_pids["iql"]=$(run_on_gpu 2 "iql" "stage1")
wait ${stage1_pids["iql"]}
stage1_pids["bcq"]=$(run_on_gpu 2 "bcq" "stage1")

# GPU 3: CQL, DQN, TD3BC (sequential on same GPU)
stage1_pids["cql"]=$(run_on_gpu 3 "cql" "stage1")
wait ${stage1_pids["cql"]}
stage1_pids["dqn"]=$(run_on_gpu 3 "dqn" "stage1")
wait ${stage1_pids["dqn"]}
stage1_pids["td3bc"]=$(run_on_gpu 3 "td3bc" "stage1")

echo ""
echo "All Stage 1 training jobs started. Waiting for completion..."
echo ""

# Wait for all Stage 1 jobs to complete
for model in cspdt bc dt iql bcq cql dqn td3bc; do
    if [ -n "${stage1_pids[$model]}" ]; then
        wait ${stage1_pids[$model]}
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "[✓] $model Stage 1 completed successfully"
        else
            echo "[✗] $model Stage 1 failed with exit code $exit_code"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Phase 1 Complete"
echo "=========================================="
echo ""

# ============================================================
# Phase 2: Stage 2 Training (all models in parallel)
# ============================================================
echo ""
echo "=========================================="
echo "Phase 2: Stage 2 Training"
echo "=========================================="
echo ""

declare -A stage2_pids

# GPU 0: CSP-DT
stage2_pids["cspdt"]=$(run_on_gpu 0 "cspdt" "stage2")

# GPU 1: BC, DT (sequential on same GPU)
stage2_pids["bc"]=$(run_on_gpu 1 "bc" "stage2")
wait ${stage2_pids["bc"]}
stage2_pids["dt"]=$(run_on_gpu 1 "dt" "stage2")

# GPU 2: IQL, BCQ (sequential on same GPU)
stage2_pids["iql"]=$(run_on_gpu 2 "iql" "stage2")
wait ${stage2_pids["iql"]}
stage2_pids["bcq"]=$(run_on_gpu 2 "bcq" "stage2")

# GPU 3: CQL, DQN, TD3BC (sequential on same GPU)
stage2_pids["cql"]=$(run_on_gpu 3 "cql" "stage2")
wait ${stage2_pids["cql"]}
stage2_pids["dqn"]=$(run_on_gpu 3 "dqn" "stage2")
wait ${stage2_pids["dqn"]}
stage2_pids["td3bc"]=$(run_on_gpu 3 "td3bc" "stage2")

echo ""
echo "All Stage 2 training jobs started. Waiting for completion..."
echo ""

# Wait for all Stage 2 jobs to complete
for model in cspdt bc dt iql bcq cql dqn td3bc; do
    if [ -n "${stage2_pids[$model]}" ]; then
        wait ${stage2_pids[$model]}
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "[✓] $model Stage 2 completed successfully"
        else
            echo "[✗] $model Stage 2 failed with exit code $exit_code"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Phase 2 Complete"
echo "=========================================="
echo ""

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "Training Pipeline Complete"
echo "=========================================="
echo ""
echo "Logs are available in: $LOG_DIR"
echo ""
echo "Models trained:"
echo "  - CSP-DT (Scheme3)"
echo "  - BC, DT, IQL, BCQ, CQL, DQN, TD3BC"
echo ""
echo "Next steps:"
echo "  1. Check logs for any errors"
echo "  2. Run evaluation on all models"
echo "  3. Compare results"
echo ""
