#!/bin/bash
# Run low-performance baseline configs (stage1 -> stage2 -> eval)
# Usage: bash train/run_low_baselines.sh
set -e
cd "$(dirname "$0")/.."

MODELS=(bc bcq cql dqn dt iql td3bc)
GPUS=(0 1 2 3)
LOGDIR=logs_low
mkdir -p $LOGDIR results_low

stage1() {
    local model=$1 gpu=$2
    echo "[$(date +%H:%M:%S)] STAGE1 $model on GPU $gpu"
    env CUDA_VISIBLE_DEVICES=$gpu python -u train/train_stage1.py \
        --config configs_low/${model}_low.yaml \
        --output_dir results_low/${model}/stage1 \
        --seed 0 \
        > $LOGDIR/${model}_stage1.log 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] STAGE1 $model FAILED (exit=$rc)"
        return 1
    fi
    echo "[$(date +%H:%M:%S)] STAGE1 $model DONE"
}

stage2() {
    local model=$1 gpu=$2
    echo "[$(date +%H:%M:%S)] STAGE2 $model on GPU $gpu"
    env CUDA_VISIBLE_DEVICES=$gpu python -u train/train_stage2.py \
        --config configs_low/${model}_low.yaml \
        --checkpoint results_low/${model}/stage1/best_checkpoint.pt \
        --output_dir results_low/${model}/stage2 \
        --semantic \
        --seed 0 \
        > $LOGDIR/${model}_stage2.log 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] STAGE2 $model FAILED (exit=$rc)"
        return 1
    fi
    echo "[$(date +%H:%M:%S)] STAGE2 $model DONE"
}

run_model() {
    local model=$1 gpu=$2
    echo "[$(date +%H:%M:%S)] Starting $model on GPU $gpu"
    stage1 $model $gpu && stage2 $model $gpu
    echo "[$(date +%H:%M:%S)] PIPELINE $model COMPLETE"
}

echo "============================================"
echo "Low-Performance Baseline Pipeline"
echo "Models: ${MODELS[*]}"
echo "GPUs: ${GPUS[*]}"
echo "Seed: 0"
echo "============================================"

# Launch up to 4 models in parallel (1 per GPU)
PIDS=()
for i in "${!MODELS[@]}"; do
    gpu_idx=$(( i % ${#GPUS[@]} ))
    gpu=${GPUS[$gpu_idx]}
    run_model ${MODELS[$i]} $gpu &
    PIDS+=($!)
    echo "[$(date +%H:%M:%S)] Launched ${MODELS[$i]} on GPU $gpu (PID $!)"
    # Stagger launches by 10s
    sleep 10
done

# Wait for all
FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid || FAIL=1
done

echo "============================================"
if [ $FAIL -eq 0 ]; then
    echo "ALL DONE - no failures"
else
    echo "SOME FAILURES - check logs in $LOGDIR"
fi
echo "============================================"
