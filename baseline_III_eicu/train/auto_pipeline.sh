#!/bin/bash
# Auto pipeline: stage1 -> stage2 -> eval, 1 model per GPU
# Usage: bash train/auto_pipeline.sh

set -e
cd "$(dirname "$0")/.."

MODELS=(bc iql bcq cql dqn td3bc)
GPUS=(0 1 2 3)
LOGDIR=/tmp/training_logs_persistent
mkdir -p $LOGDIR

stage1() {
    local model=$1 gpu=$2
    echo "[$(date +%H:%M:%S)] STAGE1 $model on GPU $gpu"
    env CUDA_VISIBLE_DEVICES=$gpu python -u train/train_stage1.py \
        --config configs/${model}.yaml \
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
        --config configs/${model}.yaml \
        --checkpoint results/${model}/stage1/best_checkpoint.pt \
        --semantic \
        > $LOGDIR/${model}_stage2.log 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] STAGE2 $model FAILED (exit=$rc)"
        return 1
    fi
    echo "[$(date +%H:%M:%S)] STAGE2 $model DONE"
}

evaluate() {
    local model=$1 gpu=$2
    local eval_script="evaluate/stratified_rollout_v3v7.py"
    if [ ! -f "$eval_script" ]; then
        echo "[$(date +%H:%M:%S)] EVAL $model SKIP (no eval script)"
        return 0
    fi
    local data_path="/home/wangmeiyi/AuctionNet/medical/last_exp/data/eicu_v3/test_Phys45_v3.pickle"
    local output_path="results/${model}/stage2_eval.json"
    echo "[$(date +%H:%M:%S)] EVAL $model on GPU $gpu"
    env CUDA_VISIBLE_DEVICES=$gpu python -u $eval_script \
        --config configs/${model}.yaml \
        --checkpoint results/${model}/stage2/best_checkpoint.pt \
        --data "$data_path" \
        --output "$output_path" \
        > $LOGDIR/${model}_eval.log 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] EVAL $model FAILED (exit=$rc)"
        return 1
    fi
    echo "[$(date +%H:%M:%S)] EVAL $model DONE"
}

run_model() {
    local model=$1 gpu=$2
    stage1 $model $gpu && stage2 $model $gpu && evaluate $model $gpu
    echo "[$(date +%H:%M:%S)] PIPELINE $model COMPLETE"
}

echo "============================================"
echo "Auto Pipeline: ${MODELS[*]}"
echo "GPUs: ${GPUS[*]}"
echo "============================================"

# Launch up to 4 models in parallel (1 per GPU)
PIDS=()
for i in "${!MODELS[@]}"; do
    gpu_idx=$(( i % ${#GPUS[@]} ))
    gpu=${GPUS[$gpu_idx]}
    run_model ${MODELS[$i]} $gpu &
    PIDS+=($!)
    echo "[$(date +%H:%M:%S)] Launched ${MODELS[$i]} on GPU $gpu (PID $!)"
    # Stagger launches by 10s to avoid memory spike
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
