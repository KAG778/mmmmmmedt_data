#!/bin/bash
# 实时监控所有训练进程的状态

BASELINE_DIR="/home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline"
LOG_DIR="/tmp/parallel_training_logs"

while true; do
  clear
  echo "=========================================="
  echo "训练进程实时监控 - $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=========================================="
  echo ""

  echo "=== Stage2 训练（今天新训练的模型）==="
  for model in bc dt iql cql; do
    log="$LOG_DIR/${model}_stage2.log"
    ckpt="$BASELINE_DIR/results/${model}/stage2/best_checkpoint.pt"

    # 检查进程状态
    if ps aux | grep -v grep | grep "train_stage2.py" | grep -q "$model"; then
      pid=$(ps aux | grep -v grep | grep "train_stage2.py" | grep "$model" | awk '{print $2}')
      status="🔄 运行中 (PID: $pid)"
    else
      status="⏹️  已停止"
    fi

    # 检查最新进度
    if [ -f "$log" ]; then
      last_epoch=$(tail -20 "$log" 2>/dev/null | grep -E "Epoch [0-9]+/[0-9]+" | tail -1)
      if [ -n "$last_epoch" ]; then
        progress="$last_epoch"
      else
        error=$(tail -10 "$log" 2>/dev/null | grep -E "Error|Traceback" | head -1)
        if [ -n "$error" ]; then
          progress="❌ 错误: ${error:0:50}..."
        else
          progress="启动中..."
        fi
      fi
    else
      progress="无日志"
    fi

    # 检查 checkpoint
    if [ -f "$ckpt" ]; then
      ckpt_status="✅"
    else
      ckpt_status="❌"
    fi

    echo "  $model: $status"
    echo "    进度: $progress"
    echo "    Checkpoint: $ckpt_status"
    echo ""
  done

  echo "=== Stage1 训练（重新训练的模型）==="
  for model in bcq dqn td3bc; do
    log="$LOG_DIR/${model}_stage1.log"
    ckpt="$BASELINE_DIR/results/${model}/stage1/best_checkpoint.pt"

    # 检查进程状态
    if ps aux | grep -v grep | grep "train_stage1.py" | grep -q "$model"; then
      pid=$(ps aux | grep -v grep | grep "train_stage1.py" | grep "$model" | awk '{print $2}')
      status="🔄 运行中 (PID: $pid)"
    else
      status="⏹️  已停止"
    fi

    # 检查最新进度
    if [ -f "$log" ]; then
      last_epoch=$(tail -20 "$log" 2>/dev/null | grep -E "Epoch [0-9]+/[0-9]+" | tail -1)
      if [ -n "$last_epoch" ]; then
        progress="$last_epoch"
      else
        error=$(tail -10 "$log" 2>/dev/null | grep -E "Error|Traceback" | head -1)
        if [ -n "$error" ]; then
          progress="❌ 错误: ${error:0:50}..."
        else
          progress="启动中..."
        fi
      fi
    else
      progress="无日志"
    fi

    # 检查 checkpoint
    if [ -f "$ckpt" ]; then
      ckpt_info=$(stat -c '%y' "$ckpt" | cut -d'.' -f1)
      ckpt_status="✅ ($ckpt_info)"
    else
      ckpt_status="❌"
    fi

    echo "  $model: $status"
    echo "    进度: $progress"
    echo "    Checkpoint: $ckpt_status"
    echo ""
  done

  echo "=== GPU 使用情况 ==="
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | \
    awk -F', ' '{printf "  GPU %s: %s / %s (%s)\n", $1, $2, $3, $4}'

  echo ""
  echo "=========================================="
  echo "按 Ctrl+C 停止监控"
  echo "=========================================="

  sleep 10
done
