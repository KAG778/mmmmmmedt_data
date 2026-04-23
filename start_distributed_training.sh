#!/bin/bash
# 分布式训练启动脚本 - 将所有训练分配到不同的 GPU 卡

set -e

LOG_DIR="/tmp/parallel_training_logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "分布式训练启动"
echo "将训练分配到 4 张 GPU 卡"
echo "=========================================="
echo ""

# GPU 分配方案：
# GPU 0: CSP-DT Stage1 (主实验)
# GPU 1: BC, IQL, DT (3个baseline)
# GPU 2: CQL, BCQ, DQN (3个baseline)
# GPU 3: TD3BC (1个baseline)

echo "=== GPU 0: CSP-DT 主实验 Stage1 ==="
cd /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2
CUDA_VISIBLE_DEVICES=0 nohup python train_stage1.py \
    --epochs 100 \
    --save_interval_epochs 10 \
    --log_interval_steps 100 \
    --logdir checkpoints/stage1 \
    > logs/stage1.log 2>&1 &
echo "CSP-DT Stage1 started on GPU 0 (PID: $!)"
echo ""

sleep 2

echo "=== GPU 1: BC, IQL, DT ==="
cd /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline

CUDA_VISIBLE_DEVICES=1 nohup python train/train_stage1.py --config configs/bc.yaml \
    > "$LOG_DIR/bc_stage1.log" 2>&1 &
echo "BC started on GPU 1 (PID: $!)"

sleep 2

CUDA_VISIBLE_DEVICES=1 nohup python train/train_stage1.py --config configs/iql.yaml \
    > "$LOG_DIR/iql_stage1.log" 2>&1 &
echo "IQL started on GPU 1 (PID: $!)"

sleep 2

CUDA_VISIBLE_DEVICES=1 nohup python train/train_stage1.py --config configs/dt.yaml \
    > "$LOG_DIR/dt_stage1.log" 2>&1 &
echo "DT started on GPU 1 (PID: $!)"
echo ""

sleep 2

echo "=== GPU 2: CQL, BCQ, DQN ==="
CUDA_VISIBLE_DEVICES=2 nohup python train/train_stage1.py --config configs/cql.yaml \
    > "$LOG_DIR/cql_stage1.log" 2>&1 &
echo "CQL started on GPU 2 (PID: $!)"

sleep 2

CUDA_VISIBLE_DEVICES=2 nohup python train/train_stage1.py --config configs/bcq.yaml \
    > "$LOG_DIR/bcq_stage1.log" 2>&1 &
echo "BCQ started on GPU 2 (PID: $!)"

sleep 2

CUDA_VISIBLE_DEVICES=2 nohup python train/train_stage1.py --config configs/dqn.yaml \
    > "$LOG_DIR/dqn_stage1.log" 2>&1 &
echo "DQN started on GPU 2 (PID: $!)"
echo ""

sleep 2

echo "=== GPU 3: TD3BC ==="
CUDA_VISIBLE_DEVICES=3 nohup python train/train_stage1.py --config configs/td3bc.yaml \
    > "$LOG_DIR/td3bc_stage1.log" 2>&1 &
echo "TD3BC started on GPU 3 (PID: $!)"
echo ""

sleep 5

echo "=========================================="
echo "所有训练已启动"
echo "=========================================="
echo ""
echo "GPU 分配："
echo "  GPU 0: CSP-DT Stage1"
echo "  GPU 1: BC, IQL, DT"
echo "  GPU 2: CQL, BCQ, DQN"
echo "  GPU 3: TD3BC"
echo ""
echo "查看 GPU 使用情况："
echo "  nvidia-smi"
echo ""
echo "查看训练日志："
echo "  CSP-DT: tail -f /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/logs/stage1.log"
echo "  Baseline: tail -f $LOG_DIR/<baseline>_stage1.log"
echo ""
