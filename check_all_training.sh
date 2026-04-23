#!/bin/bash
# 最终训练状态总结

echo "=========================================="
echo "所有训练进程最终状态"
echo "=========================================="
echo ""

echo "=== CSP-DT 主实验 (改进后) ==="
CSPDT_PID=$(ps aux | grep "train_stage1.py.*scheme3_cspdt_v2" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$CSPDT_PID" ]; then
    echo "✅ Stage1 正在运行 (PID: $CSPDT_PID)"
    tail -3 /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/logs/stage1.log 2>/dev/null | grep -E "(Epoch|Validation|Best)"
else
    echo "❌ Stage1 未运行"
fi
echo ""

echo "=== Baseline 训练状态 ==="
for config in bc iql dt cql bcq dqn td3bc; do
    PID=$(ps aux | grep "train_stage1.py.*${config}.yaml" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$PID" ]; then
        echo "✅ $config: PID=$PID"
    else
        echo "❌ $config: 未运行"
    fi
done
echo ""

echo "=== GPU 使用情况 ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | \
    awk -F', ' '{printf "GPU %s: %s (%s%%)\n", $1, $2, $3}'
echo ""

echo "=========================================="
echo "总计: $(ps aux | grep 'train_stage1.py' | grep -v grep | wc -l) 个训练进程正在运行"
echo "=========================================="
