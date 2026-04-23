#!/bin/bash
# 重新训练 CSP-DT 主实验（Stage1 + Stage2）
# 使用改进的训练流程：Stage1 保存 best_checkpoint，Stage2 使用 best_checkpoint

set -e

CSPDT_DIR="/home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2"
LOG_DIR="$CSPDT_DIR/logs"
CKPT_DIR="$CSPDT_DIR/checkpoints"

echo "=========================================="
echo "CSP-DT 主实验重新训练"
echo "Stage1 → Stage2 (使用 best_checkpoint)"
echo "=========================================="
echo ""

# 备份旧的 checkpoints
if [ -d "$CKPT_DIR/stage1" ]; then
    BACKUP_DIR="$CKPT_DIR/backup_$(date +%Y%m%d_%H%M%S)"
    echo "备份旧的 checkpoints 到: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    mv "$CKPT_DIR/stage1" "$BACKUP_DIR/" 2>/dev/null || true
    mv "$CKPT_DIR/stage2" "$BACKUP_DIR/" 2>/dev/null || true
fi

# 清理旧的日志
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR/stage1.log" "$LOG_DIR/stage2_main.log"

echo ""
echo "=== Phase 1: Stage1 训练 (带 validation 和 best checkpoint) ==="
cd "$CSPDT_DIR"

nohup python train_stage1.py \
    --epochs 100 \
    --save_interval_epochs 10 \
    --log_interval_steps 100 \
    --logdir checkpoints/stage1 \
    > logs/stage1.log 2>&1 &

STAGE1_PID=$!
echo "Stage1 训练已启动 (PID: $STAGE1_PID)"
echo "日志: $LOG_DIR/stage1.log"
echo ""
echo "等待 Stage1 训练完成..."
echo "提示: 可以使用 'tail -f $LOG_DIR/stage1.log' 查看训练进度"
echo ""

# 等待 Stage1 完成
wait $STAGE1_PID
STAGE1_EXIT_CODE=$?

if [ $STAGE1_EXIT_CODE -ne 0 ]; then
    echo "❌ Stage1 训练失败 (exit code: $STAGE1_EXIT_CODE)"
    echo "请检查日志: $LOG_DIR/stage1.log"
    exit 1
fi

echo "✓ Stage1 训练完成"
echo ""

# 检查 best_checkpoint 是否存在
if [ ! -f "$CKPT_DIR/stage1/best_checkpoint/policy.pt" ]; then
    echo "❌ 错误: Stage1 best_checkpoint 不存在"
    echo "请检查 Stage1 训练日志"
    exit 1
fi

# 显示 best_checkpoint 信息
if [ -f "$CKPT_DIR/stage1/best_checkpoint/metadata.json" ]; then
    echo "Best checkpoint 信息:"
    cat "$CKPT_DIR/stage1/best_checkpoint/metadata.json"
    echo ""
fi

echo ""
echo "=== Phase 2: Stage2 训练 (使用 best_checkpoint) ==="

nohup python train_stage2.py \
    --policy_ckpt checkpoints/stage1/best_checkpoint/policy.pt \
    --world_model_ckpt checkpoints/stage1/best_checkpoint/world_model.pt \
    --epochs 50 \
    --selfplay_iterations 1000 \
    --logdir checkpoints/stage2 \
    > logs/stage2_main.log 2>&1 &

STAGE2_PID=$!
echo "Stage2 训练已启动 (PID: $STAGE2_PID)"
echo "日志: $LOG_DIR/stage2_main.log"
echo ""
echo "提示: 可以使用 'tail -f $LOG_DIR/stage2_main.log' 查看训练进度"
echo ""
echo "=========================================="
echo "训练流程已启动"
echo "Stage1 PID: $STAGE1_PID (已完成)"
echo "Stage2 PID: $STAGE2_PID (运行中)"
echo "=========================================="
