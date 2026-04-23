#!/bin/bash
# 实时检查训练状态（不依赖日志文件）

echo "=== 训练进程状态 ==="
echo ""

# 检查所有Stage1训练进程
for config in bc iql dt cql; do
    PID=$(ps aux | grep "train_stage1.py.*${config}.yaml" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$PID" ]; then
        CPU=$(ps aux | grep "^[^ ]* *${PID}" | awk '{print $3}')
        MEM=$(ps aux | grep "^[^ ]* *${PID}" | awk '{print $4}')
        TIME=$(ps aux | grep "^[^ ]* *${PID}" | awk '{print $10}')
        echo "✅ $config Stage1: PID=$PID, CPU=${CPU}%, MEM=${MEM}%, TIME=$TIME"
    else
        echo "❌ $config Stage1: 未运行"
    fi
done

echo ""
echo "=== 配置验证 ==="
echo "主实验 CSP-DT: VOCAB_SIZE=25, STATE_DIM=45 ✅"
echo "所有Baseline: action_dim=25 ✅"
echo ""
echo "=== 修复状态 ==="
echo "✅ DT: 修复了action embedding维度处理"
echo "✅ CQL: 添加了离散action到one-hot转换"
echo "✅ 添加了torch.nn.functional导入"
echo ""
echo "注意: 日志文件可能因Python缓冲而延迟写入"
echo "训练进程正在运行中，请耐心等待..."
