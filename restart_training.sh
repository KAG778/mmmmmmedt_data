#!/bin/bash
# 重新启动正确的训练流程

set -e
LOG_DIR="/tmp/parallel_training_logs"
mkdir -p "$LOG_DIR"

echo "=== 启动 Baseline Stage1 训练 ==="

# BC
cd /home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline
nohup python train/train_stage1.py --config configs/bc.yaml > "$LOG_DIR/bc_stage1.log" 2>&1 &
echo "BC Stage1 started (PID: $!)"

# IQL
nohup python train/train_stage1.py --config configs/iql.yaml > "$LOG_DIR/iql_stage1.log" 2>&1 &
echo "IQL Stage1 started (PID: $!)"

# DT
nohup python train/train_stage1.py --config configs/dt.yaml > "$LOG_DIR/dt_stage1.log" 2>&1 &
echo "DT Stage1 started (PID: $!)"

# CQL
nohup python train/train_stage1.py --config configs/cql.yaml > "$LOG_DIR/cql_stage1.log" 2>&1 &
echo "CQL Stage1 started (PID: $!)"

echo ""
echo "=== 所有 Baseline Stage1 训练已启动 ==="
echo "日志目录: $LOG_DIR"
