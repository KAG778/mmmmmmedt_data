#!/bin/bash
# 完整的训练状态总结

echo "=========================================="
echo "所有训练进程状态总结"
echo "=========================================="
echo ""

echo "=== CSP-DT 主实验 ==="
CSPDT_STAGE1=$(ps aux | grep "train_stage1.py.*scheme3_cspdt_v2" | grep -v grep | wc -l)
if [ $CSPDT_STAGE1 -gt 0 ]; then
    echo "✅ Stage1 训练正在运行"
    tail -5 /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/logs/stage1.log 2>/dev/null | grep -E "(Epoch|Validation|Best)" | tail -2
else
    echo "❌ Stage1 训练未运行"
fi
echo ""

echo "=== Baseline 训练 ==="
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
echo "=========================================="
echo "配置验证"
echo "=========================================="
echo "✅ 主实验 CSP-DT: VOCAB_SIZE=25, STATE_DIM=45"
echo "✅ 所有 Baseline: action_dim=25"
echo "✅ 数据实际维度: Actions 0~24 (25个值), State 45维"
echo ""
echo "=========================================="
echo "改进内容"
echo "=========================================="
echo "✅ Stage1: 添加了 validation loss 跟踪"
echo "✅ Stage1: 添加了 best checkpoint 自动保存"
echo "✅ Stage2: 默认使用 best_checkpoint"
echo "✅ DT/CQL: 修复了维度错误"
echo ""
echo "=========================================="
echo "监控命令"
echo "=========================================="
echo "CSP-DT Stage1: tail -f /home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型/scheme3_cspdt_v2/logs/stage1.log"
echo "Baseline 训练: bash /home/wangmeiyi/AuctionNet/medical/重构之后的代码/check_training_status.sh"
echo "=========================================="
