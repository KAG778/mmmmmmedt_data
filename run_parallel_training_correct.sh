#!/bin/bash
"""
Correct parallel training script for CSP-DT and all baseline models.
Stage1 and Stage2 are SEQUENTIAL (Stage2 depends on Stage1 checkpoints).
Only models are parallelized across GPUs.
"""

LOG_DIR="/tmp/parallel_training_logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Parallel Training Pipeline (Corrected)"
echo "=========================================="
echo "Phase 1: All models Stage1 (parallel)"
echo "Phase 2: All models Stage2 (parallel, after Stage1)"
echo ""
echo "GPU allocation:"
echo "  GPU 0: CSP-DT"
echo "  GPU 1: BC, DT"
echo "  GPU 2: IQL, BCQ"
echo "  GPU 3: CQL, DQN, TD3BC"
echo "=========================================="
echo ""

# ============================================================
# Phase 1: Stage 1 Training (all models in parallel)
# ============================================================
echo ""
echo "=========================================="
echo "Phase 1: Stage 1 Training"
echo "=========================================="
echo ""

# GPU 0: CSP-DT Stage1
echo "[GPU 0] Starting CSP-DT Stage1"
CUDA_VISIBLE_DEVICES=0 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/train_stage1.py \
    --epochs 100 \
    --save_interval_epochs 10 \
    --log_interval_steps 100 \
    --logdir /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage1 \
    > "$LOG_DIR/cspdt_stage1.log" 2>&1 &
CSPDT_S1_PID=$!

# GPU 1: BC Stage1
echo "[GPU 1] Starting BC Stage1"
CUDA_VISIBLE_DEVICES=1 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/bc.yaml \
    > "$LOG_DIR/bc_stage1.log" 2>&1 &
BC_S1_PID=$!

# GPU 2: IQL Stage1
echo "[GPU 2] Starting IQL Stage1"
CUDA_VISIBLE_DEVICES=2 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/iql.yaml \
    > "$LOG_DIR/iql_stage1.log" 2>&1 &
IQL_S1_PID=$!

# GPU 3: CQL Stage1
echo "[GPU 3] Starting CQL Stage1"
CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/cql.yaml \
    > "$LOG_DIR/cql_stage1.log" 2>&1 &
CQL_S1_PID=$!

echo ""
echo "Waiting for first batch of Stage1 to complete..."
wait $CSPDT_S1_PID && echo "[✓] CSP-DT Stage1 done" || echo "[✗] CSP-DT Stage1 failed"
wait $BC_S1_PID && echo "[✓] BC Stage1 done" || echo "[✗] BC Stage1 failed"
wait $IQL_S1_PID && echo "[✓] IQL Stage1 done" || echo "[✗] IQL Stage1 failed"
wait $CQL_S1_PID && echo "[✓] CQL Stage1 done" || echo "[✗] CQL Stage1 failed"

# Second batch on same GPUs
echo ""
echo "Starting second batch of Stage1..."

# GPU 1: DT Stage1
echo "[GPU 1] Starting DT Stage1"
CUDA_VISIBLE_DEVICES=1 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/dt.yaml \
    > "$LOG_DIR/dt_stage1.log" 2>&1 &
DT_S1_PID=$!

# GPU 2: BCQ Stage1
echo "[GPU 2] Starting BCQ Stage1"
CUDA_VISIBLE_DEVICES=2 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/bcq.yaml \
    > "$LOG_DIR/bcq_stage1.log" 2>&1 &
BCQ_S1_PID=$!

# GPU 3: DQN Stage1
echo "[GPU 3] Starting DQN Stage1"
CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/dqn.yaml \
    > "$LOG_DIR/dqn_stage1.log" 2>&1 &
DQN_S1_PID=$!

wait $DT_S1_PID && echo "[✓] DT Stage1 done" || echo "[✗] DT Stage1 failed"
wait $BCQ_S1_PID && echo "[✓] BCQ Stage1 done" || echo "[✗] BCQ Stage1 failed"
wait $DQN_S1_PID && echo "[✓] DQN Stage1 done" || echo "[✗] DQN Stage1 failed"

# Third batch
echo ""
echo "Starting third batch of Stage1..."

# GPU 3: TD3BC Stage1
echo "[GPU 3] Starting TD3BC Stage1"
CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage1.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/td3bc.yaml \
    > "$LOG_DIR/td3bc_stage1.log" 2>&1 &
TD3BC_S1_PID=$!

wait $TD3BC_S1_PID && echo "[✓] TD3BC Stage1 done" || echo "[✗] TD3BC Stage1 failed"

echo ""
echo "=========================================="
echo "Phase 1 Complete - All Stage1 Done"
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

# GPU 0: CSP-DT Stage2
echo "[GPU 0] Starting CSP-DT Stage2"
CUDA_VISIBLE_DEVICES=0 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/train_stage2.py \
    --policy_ckpt /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage1/epoch_100/policy.pt \
    --world_model_ckpt /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage1/epoch_100/world_model.pt \
    --epochs 50 \
    --selfplay_iterations 1000 \
    --logdir /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/checkpoints/stage2 \
    > "$LOG_DIR/cspdt_stage2.log" 2>&1 &
CSPDT_S2_PID=$!

# GPU 1: BC Stage2
echo "[GPU 1] Starting BC Stage2"
CUDA_VISIBLE_DEVICES=1 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/bc.yaml \
    > "$LOG_DIR/bc_stage2.log" 2>&1 &
BC_S2_PID=$!

# GPU 2: IQL Stage2
echo "[GPU 2] Starting IQL Stage2"
CUDA_VISIBLE_DEVICES=2 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/iql.yaml \
    > "$LOG_DIR/iql_stage2.log" 2>&1 &
IQL_S2_PID=$!

# GPU 3: CQL Stage2
echo "[GPU 3] Starting CQL Stage2"
CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/cql.yaml \
    > "$LOG_DIR/cql_stage2.log" 2>&1 &
CQL_S2_PID=$!

wait $CSPDT_S2_PID && echo "[✓] CSP-DT Stage2 done" || echo "[✗] CSP-DT Stage2 failed"
wait $BC_S2_PID && echo "[✓] BC Stage2 done" || echo "[✗] BC Stage2 failed"
wait $IQL_S2_PID && echo "[✓] IQL Stage2 done" || echo "[✗] IQL Stage2 failed"
wait $CQL_S2_PID && echo "[✓] CQL Stage2 done" || echo "[✗] CQL Stage2 failed"

# Second batch Stage2
echo ""
echo "Starting second batch of Stage2..."

# GPU 1: DT Stage2
echo "[GPU 1] Starting DT Stage2"
CUDA_VISIBLE_DEVICES=1 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/dt.yaml \
    > "$LOG_DIR/dt_stage2.log" 2>&1 &
DT_S2_PID=$!

# GPU 2: BCQ Stage2
echo "[GPU 2] Starting BCQ Stage2"
CUDA_VISIBLE_DEVICES=2 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/bcq.yaml \
    > "$LOG_DIR/bcq_stage2.log" 2>&1 &
BCQ_S2_PID=$!

# GPU 3: DQN Stage2
echo "[GPU 3] Starting DQN Stage2"
CUDA_VISIBLE_DEVICES=3 nohup python /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/train/train_stage2.py \
    --config /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline/configs/dqn.yaml \
    > "$LOG_DIR/dqn_stage2.log" 2>&1 &
DQN_S2_PID=$!

wait $DT_S2_PID && echo "[✓] DT Stage2 done" || echo "[✗] DT Stage2 failed"
wait $BCQ_S2_PID && echo "[✓] BCQ Stage2 done" || echo "[✗] BCQ Stage2 failed"
wait $DQN_S2_PID && echo "[✓] DQN Stage2 done" || echo "[✗] DQN Stage2 failed"

# Third batch Stage2
echo ""
echo "Starting third batch of Stage2..."

# GPU 3: TD3BC Stage2
echo "[GPU 3] Starting TD3BC Stage2"
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
echo "Logs: $LOG_DIR"
echo ""
echo "Models trained:"
echo "  - CSP-DT (Scheme3)"
echo "  - BC, DT, IQL, BCQ, CQL, DQN, TD3BC"
echo ""
