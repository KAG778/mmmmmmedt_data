#!/bin/bash
# Train baseline_low models: stage1 -> stage2, seed=0
# Usage: bash train/auto_pipeline_low.sh

set -e
cd "$(dirname "$0")/.."

MODELS=(bc dqn cql iql bcq td3bc dt)
GPUS=(0 1 2 3)
SEED=0
LOGDIR=logs_low
mkdir -p $LOGDIR

stage1() {
    local model=$1 gpu=$2
    echo "[$(date +%H:%M:%S)] STAGE1 ${model}_low on GPU $gpu"
    env CUDA_VISIBLE_DEVICES=$gpu python -u train/train_stage1.py \
        --config configs_low/${model}_sigma2_low.yaml \
        --seed $SEED \
        > $LOGDIR/${model}_low_stage1.log 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] STAGE1 ${model}_low FAILED (exit=$rc)"
        return 1
    fi
    echo "[$(date +%H:%M:%S)] STAGE1 ${model}_low DONE"
}

stage2() {
    local model=$1 gpu=$2
    echo "[$(date +%H:%M:%S)] STAGE2 ${model}_low on GPU $gpu"
    env CUDA_VISIBLE_DEVICES=$gpu python -u train/train_stage2.py \
        --config configs_low/${model}_sigma2_low.yaml \
        --checkpoint results/${model}_low/stage1/best_checkpoint.pt \
        --semantic \
        --seed $SEED \
        > $LOGDIR/${model}_low_stage2.log 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] STAGE2 ${model}_low FAILED (exit=$rc)"
        return 1
    fi
    echo "[$(date +%H:%M:%S)] STAGE2 ${model}_low DONE"
}

run_model() {
    local model=$1 gpu=$2
    stage1 $model $gpu && stage2 $model $gpu
    echo "[$(date +%H:%M:%S)] PIPELINE ${model}_low COMPLETE"
}

echo "============================================"
echo "Baseline LOW Pipeline: ${MODELS[*]}"
echo "GPUs: ${GPUS[*]}  Seed: $SEED"
echo "============================================"

# Launch models round-robin on GPUs
PIDS=()
for i in "${!MODELS[@]}"; do
    gpu_idx=$(( i % ${#GPUS[@]} ))
    gpu=${GPUS[$gpu_idx]}
    run_model ${MODELS[$i]} $gpu &
    PIDS+=($!)
    echo "[$(date +%H:%M:%S)] Launched ${MODELS[$i]}_low on GPU $gpu (PID $!)"
    sleep 10
done

# Wait for all
FAIL=0
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]} || FAIL=1
done

echo "============================================"
if [ $FAIL -eq 0 ]; then
    echo "ALL DONE - no failures"
else
    echo "SOME FAILURES - check logs in $LOGDIR"
fi
echo "============================================"
