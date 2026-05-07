#!/bin/bash
# Parallel pipeline for baseline_III + main_model_III
# Each experiment runs its own stage1 -> stage2 -> eval independently
# 2 experiments per GPU (except main model gets its own GPU slot)
#
# GPU 0 (18.7GB free): bcq, cql
# GPU 1 (36.8GB free): main, dqn
# GPU 2 (35.1GB free): bc, dt
# GPU 3 (20.8GB free): iql, td3bc

set -e
cd "$(dirname "$0")"

LOGDIR=logs
mkdir -p $LOGDIR

BASELINE_DIR=baseline_III
MAIN_DIR=main_model_III/scheme3_cspdt_v2
DATA_TEST="data/v3/test_Phys45_v3.pickle"

ts() { date +%H:%M:%S; }

# Single baseline: stage1 -> stage2 -> eval
run_baseline() {
    local model=$1 gpu=$2
    local cfg="$BASELINE_DIR/configs/${model}_sigma2.yaml"
    local res="$BASELINE_DIR/results/${model}_sigma2"

    echo "[$(ts)] [$model] Starting on GPU $gpu"

    # Stage 1
    echo "[$(ts)] [$model] Stage 1 starting"
    CUDA_VISIBLE_DEVICES=$gpu python -u $BASELINE_DIR/train/train_stage1.py \
        --config "$cfg" \
        > "$LOGDIR/${model}_sigma2_stage1.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [$model] Stage 1 FAILED"; return 1; fi
    echo "[$(ts)] [$model] Stage 1 done"

    # Stage 2
    echo "[$(ts)] [$model] Stage 2 starting"
    CUDA_VISIBLE_DEVICES=$gpu python -u $BASELINE_DIR/train/train_stage2.py \
        --config "$cfg" \
        --checkpoint "$res/stage1/best_checkpoint.pt" \
        > "$LOGDIR/${model}_sigma2_stage2.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [$model] Stage 2 FAILED"; return 1; fi
    echo "[$(ts)] [$model] Stage 2 done"

    # Eval
    echo "[$(ts)] [$model] Eval starting"
    CUDA_VISIBLE_DEVICES=$gpu python -u $BASELINE_DIR/evaluate/stratified_rollout_v3v7.py \
        --config "$cfg" \
        --checkpoint "$res/stage2/best_checkpoint.pt" \
        --data "$DATA_TEST" \
        --output "$res/stage2_eval.json" \
        > "$LOGDIR/${model}_sigma2_eval.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [$model] Eval FAILED"; return 1; fi
    echo "[$(ts)] [$model] Eval done"
    echo "[$(ts)] [$model] COMPLETE"
}

# Main model: stage1 -> stage2 -> eval
run_main() {
    local gpu=$1

    echo "[$(ts)] [MAIN] Starting on GPU $gpu"

    echo "[$(ts)] [MAIN] Stage 1 starting"
    CUDA_VISIBLE_DEVICES=$gpu python -u $MAIN_DIR/train_stage1_no_sem_epoch.py \
        > "$LOGDIR/main_stage1.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [MAIN] Stage 1 FAILED"; return 1; fi
    echo "[$(ts)] [MAIN] Stage 1 done"

    echo "[$(ts)] [MAIN] Stage 2 starting"
    CUDA_VISIBLE_DEVICES=$gpu python -u $MAIN_DIR/train_stage2_no_sem_sigma2.py \
        > "$LOGDIR/main_stage2.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [MAIN] Stage 2 FAILED"; return 1; fi
    echo "[$(ts)] [MAIN] Stage 2 done"

    echo "[$(ts)] [MAIN] Eval starting"
    CUDA_VISIBLE_DEVICES=$gpu python -u $MAIN_DIR/stratified_rollout_no_sem.py \
        > "$LOGDIR/main_eval.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [MAIN] Eval FAILED"; return 1; fi
    echo "[$(ts)] [MAIN] Eval done"
    echo "[$(ts)] [MAIN] COMPLETE"
}

echo "[$(ts)] ================================"
echo "[$(ts)] III Parallel Pipeline"
echo "[$(ts)] GPU 0: bcq, cql"
echo "[$(ts)] GPU 1: main, dqn"
echo "[$(ts)] GPU 2: bc, dt"
echo "[$(ts)] GPU 3: iql, td3bc"
echo "[$(ts)] ================================"

# Launch all 8 experiments in parallel
PIDS=()

# GPU 0
run_baseline bcq 0 &
PIDS+=($!)
sleep 3

run_baseline cql 0 &
PIDS+=($!)
sleep 3

# GPU 1
run_main 1 &
PIDS+=($!)
sleep 3

run_baseline dqn 1 &
PIDS+=($!)
sleep 3

# GPU 2
run_baseline bc 2 &
PIDS+=($!)
sleep 3

run_baseline dt 2 &
PIDS+=($!)
sleep 3

# GPU 3
run_baseline iql 3 &
PIDS+=($!)
sleep 3

run_baseline td3bc 3 &
PIDS+=($!)
sleep 3

echo "[$(ts)] All 8 experiments launched. PIDs: ${PIDS[*]}"
echo "[$(ts)] Waiting for all to complete..."

FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid || FAIL=1
done

echo "[$(ts)] ================================"
if [ $FAIL -eq 0 ]; then
    echo "[$(ts)] ALL DONE"
else
    echo "[$(ts)] SOME FAILURES - check logs/"
fi
echo "[$(ts)] ================================"
