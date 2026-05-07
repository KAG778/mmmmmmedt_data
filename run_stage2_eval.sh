#!/bin/bash
# Stage 2 + Eval only (Stage 1 all completed)
# Checkpoints are at baseline_III/results/{model}/stage1/best_checkpoint.pt
# (NOT results/{model}_sigma2/)

set -e
cd "$(dirname "$0")"

LOGDIR=logs
BASELINE_DIR=baseline_III
MAIN_DIR=main_model_III/scheme3_cspdt_v2
DATA_TEST="data/v3/test_Phys45_v3.pickle"

ts() { date +%H:%M:%S; }

run_baseline_s2eval() {
    local model=$1 gpu=$2
    local cfg="$BASELINE_DIR/configs/${model}_sigma2.yaml"
    local ckpt="$BASELINE_DIR/results/${model}/stage1/best_checkpoint.pt"

    echo "[$(ts)] [$model] Stage 2 on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -u $BASELINE_DIR/train/train_stage2.py \
        --config "$cfg" \
        --checkpoint "$ckpt" \
        > "$LOGDIR/${model}_sigma2_stage2.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [$model] Stage 2 FAILED"; return 1; fi
    echo "[$(ts)] [$model] Stage 2 done"

    local ckpt2="$BASELINE_DIR/results/${model}/stage2_sigma2/best_checkpoint.pt"
    echo "[$(ts)] [$model] Eval on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -u $BASELINE_DIR/evaluate/stratified_rollout_v3v7.py \
        --config "$cfg" \
        --checkpoint "$ckpt2" \
        --data "$DATA_TEST" \
        --output "$BASELINE_DIR/results/${model}/stage2_sigma2_eval.json" \
        > "$LOGDIR/${model}_sigma2_eval.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [$model] Eval FAILED"; return 1; fi
    echo "[$(ts)] [$model] Eval done -> COMPLETE"
}

run_main_s2eval() {
    local gpu=$1

    echo "[$(ts)] [MAIN] Stage 2 on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -u $MAIN_DIR/train_stage2_no_sem_sigma2.py \
        --logdir $MAIN_DIR/checkpoints_no_sem_wm/stage2_sigma2 \
        --policy_ckpt $MAIN_DIR/checkpoints_no_sem_wm/stage1/epoch_100/policy.pt \
        --world_model_ckpt $MAIN_DIR/checkpoints_no_sem_wm/stage1/epoch_100/world_model.pt \
        > "$LOGDIR/main_stage2.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [MAIN] Stage 2 FAILED"; return 1; fi
    echo "[$(ts)] [MAIN] Stage 2 done"

    echo "[$(ts)] [MAIN] Eval on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -u $MAIN_DIR/stratified_rollout_no_sem.py \
        --checkpoint $MAIN_DIR/checkpoints_no_sem_wm/stage2_sigma2/best_checkpoint.pt \
        --data "$DATA_TEST" \
        --output $MAIN_DIR/results/stage2_sigma2_eval.json \
        > "$LOGDIR/main_eval.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [MAIN] Eval FAILED"; return 1; fi
    echo "[$(ts)] [MAIN] Eval done -> COMPLETE"
}

echo "[$(ts)] ================================"
echo "[$(ts)] III Stage2 + Eval (parallel)"
echo "[$(ts)] GPU 0: bcq, cql"
echo "[$(ts)] GPU 1: main, dqn"
echo "[$(ts)] GPU 2: bc, dt"
echo "[$(ts)] GPU 3: iql, td3bc"
echo "[$(ts)] ================================"

PIDS=()

# GPU 0
run_baseline_s2eval bcq 0 &
PIDS+=($!)
sleep 3
run_baseline_s2eval cql 0 &
PIDS+=($!)
sleep 3

# GPU 1
run_main_s2eval 1 &
PIDS+=($!)
sleep 3
run_baseline_s2eval dqn 1 &
PIDS+=($!)
sleep 3

# GPU 2
run_baseline_s2eval bc 2 &
PIDS+=($!)
sleep 3
run_baseline_s2eval dt 2 &
PIDS+=($!)
sleep 3

# GPU 3
run_baseline_s2eval iql 3 &
PIDS+=($!)
sleep 3
run_baseline_s2eval td3bc 3 &
PIDS+=($!)
sleep 3

echo "[$(ts)] All launched. PIDs: ${PIDS[*]}"

FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid || FAIL=1
done

echo "[$(ts)] ================================"
if [ $FAIL -eq 0 ]; then echo "[$(ts)] ALL DONE"; else echo "[$(ts)] SOME FAILURES"; fi
echo "[$(ts)] ================================"
