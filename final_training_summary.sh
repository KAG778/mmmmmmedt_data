#!/bin/bash
# 完整的训练状态总结

echo "=========================================="
echo "所有训练进程最终状态"
echo "=========================================="
echo ""

echo "=== CSP-DT 主实验 (改进后的训练流程) ==="
CSPDT_STAGE1=$(ps aux | grep "train_stage1.py.*scheme3_cspdt_v2" | grep -v grep | wc -l)
if [ $CSPDT_STAGE1 -gt 0 ]; then
    echo "✅ Stage1 训练正在运行"
    tail -3 /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/logs/stage1.log 2>/dev/null | grep -E "(Epoch|Validation|Best)"
else
    echo "❌ Stage1 训练未运行（可能已完成或出错）"
fi
echo ""

echo "=== Baseline 训练 (GPU 0) ==="
for config in bc iql dt cql; do
    PID=$(ps aux | grep "train_stage1.py.*${config}.yaml" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$PID" ]; then
        CPU=$(ps aux | grep "^[^ ]* *${PID}" | awk '{print $3}')
        TIME=$(ps aux | grep "^[^ ]* *${PID}" | awk '{print $10}')
        echo "✅ $config: PID=$PID, CPU=${CPU}%, TIME=$TIME"
    else
        echo "❌ $config: 未运行"
    fi
done
echo ""

echo "=== Baseline 训练 (GPU 1-3) ==="
for config in bcq dqn td3bc; do
    PID=$(ps aux | grep "train_stage1.py.*${config}.yaml" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$PID" ]; then
        CPU=$(ps aux | grep "^[^ ]* *${PID}" | awk '{print $3}')
        TIME=$(ps aux | grep "^[^ ]* *${PID}" | awk '{print $10}')
        echo "✅ $config: PID=$PID, CPU=${CPU}%, TIME=$TIME"
    else
        echo "❌ $config: 未运行"
    fi
done
echo ""

echo "=========================================="
echo "GPU 使用情况"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | \
    awk -F', ' '{printf "GPU %s: %s / %s (%s%%)\n", $1, $3, $4, $5}'
echo ""

echo "=========================================="
echo "改进总结"
echo "=========================================="
echo "✅ CSP-DT Stage1: 添加了 validation loss 跟踪和 best checkpoint"
echo "✅ CSP-DT Stage2: 默认使用 best_checkpoint"
echo "✅ 所有 Baseline: 修复了维度错误（DT、CQL、BCQ、DQN、TD3BC）"
echo "✅ 配置验证: action_dim=25 (正确)"
echo ""
echo "=========================================="
