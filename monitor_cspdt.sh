#!/bin/bash
# 监控 CSP-DT 训练进度

CSPDT_DIR="/home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2"

echo "=========================================="
echo "CSP-DT 训练进度监控"
echo "=========================================="
echo ""

# 检查 Stage1 训练进程
STAGE1_PID=$(ps aux | grep "train_stage1.py.*scheme3_cspdt_v2" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$STAGE1_PID" ]; then
    CPU=$(ps aux | grep "^[^ ]* *${STAGE1_PID}" | awk '{print $3}')
    MEM=$(ps aux | grep "^[^ ]* *${STAGE1_PID}" | awk '{print $4}')
    TIME=$(ps aux | grep "^[^ ]* *${STAGE1_PID}" | awk '{print $10}')
    echo "✅ Stage1 训练正在运行"
    echo "   PID: $STAGE1_PID"
    echo "   CPU: ${CPU}%"
    echo "   MEM: ${MEM}%"
    echo "   运行时间: $TIME"
    echo ""

    # 显示最新的训练日志
    echo "=== Stage1 最新日志 ==="
    tail -10 "$CSPDT_DIR/logs/stage1.log" 2>/dev/null | grep -E "(Epoch|Validation|Best)" || echo "等待日志输出..."
else
    echo "❌ Stage1 训练未运行"

    # 检查 Stage2 是否在运行
    STAGE2_PID=$(ps aux | grep "train_stage2.py.*scheme3_cspdt_v2" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$STAGE2_PID" ]; then
        echo "✅ Stage2 训练正在运行"
        echo "   PID: $STAGE2_PID"
        echo ""
        echo "=== Stage2 最新日志 ==="
        tail -10 "$CSPDT_DIR/logs/stage2_main.log" 2>/dev/null | grep -E "(Epoch|Iteration|Advantage)" || echo "等待日志输出..."
    fi
fi

echo ""
echo "=========================================="
echo "查看完整日志:"
echo "  Stage1: tail -f $CSPDT_DIR/logs/stage1.log"
echo "  Stage2: tail -f $CSPDT_DIR/logs/stage2_main.log"
echo "=========================================="
