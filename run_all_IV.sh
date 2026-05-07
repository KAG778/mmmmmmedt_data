#!/bin/bash
# MIMIC-IV Full Pipeline: Main Model + 7 Baselines, 2 seeds (0, 42)
#
# Pipeline per experiment: stage1 -> stage2 -> eval
# Results with mean±std across seeds for final table.
#
# GPU allocation (GPU 0 reserved for other users):
#   GPU 1: main(s0), main(s1)
#   GPU 2: bc(s0), bc(s1), bcq(s0), bcq(s1), dqn(s0), dqn(s1)
#   GPU 3: cql(s0), cql(s1), dt(s0), dt(s1), iql(s0), iql(s1), td3bc(s0), td3bc(s1)

set -e
cd "$(dirname "$0")"

LOGDIR=logs/IV_2seeds
mkdir -p $LOGDIR

MAIN_DIR=main_model_IV/scheme3_cspdt_v2
BASELINE_DIR=baseline_Iv
DATA_TEST="data/mimic_iv_v3/test_Phys45_v3.pickle"
MODELS="bc bcq cql dqn dt iql td3bc"
SEEDS="0 42"

ts() { date +%H:%M:%S; }

# ============================================================
# Main Model: stage1 -> stage2 -> eval
# ============================================================
run_main() {
    local seed=$1 gpu=$2
    local ckpt_base="$MAIN_DIR/checkpoints_IV/seed${seed}"

    echo "[$(ts)] [MAIN seed=$seed] Stage 1 on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -u $MAIN_DIR/train_stage1_no_sem_epoch.py \
        --seed $seed \
        --logdir "${ckpt_base}_stage1" \
        > "$LOGDIR/main_seed${seed}_s1.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [MAIN seed=$seed] Stage 1 FAILED"; return 1; fi
    echo "[$(ts)] [MAIN seed=$seed] Stage 1 done"

    echo "[$(ts)] [MAIN seed=$seed] Stage 2 on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -u $MAIN_DIR/train_stage2_no_sem_sigma2.py \
        --seed $seed \
        --logdir "${ckpt_base}_stage2" \
        --policy_ckpt "${ckpt_base}_stage1/epoch_100/policy.pt" \
        --world_model_ckpt "${ckpt_base}_stage1/epoch_100/world_model.pt" \
        > "$LOGDIR/main_seed${seed}_s2.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [MAIN seed=$seed] Stage 2 FAILED"; return 1; fi
    echo "[$(ts)] [MAIN seed=$seed] Stage 2 done"

    echo "[$(ts)] [MAIN seed=$seed] Eval on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -u $MAIN_DIR/stratified_rollout_no_sem_wm.py \
        --seed $seed \
        --checkpoint "${ckpt_base}_stage2/best_checkpoint.pt" \
        --data "$DATA_TEST" \
        --output "$MAIN_DIR/results/seed${seed}_stage2_eval.json" \
        > "$LOGDIR/main_seed${seed}_eval.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [MAIN seed=$seed] Eval FAILED"; return 1; fi
    echo "[$(ts)] [MAIN seed=$seed] COMPLETE"
}

# ============================================================
# Baseline: stage1 -> stage2 -> eval
# ============================================================
run_baseline() {
    local model=$1 seed=$2 gpu=$3
    local cfg="$BASELINE_DIR/configs/${model}_sigma2.yaml"
    local base="$BASELINE_DIR/results/${model}"
    local s1dir="${base}/seed${seed}_stage1"
    local s2dir="${base}/seed${seed}_stage2"

    echo "[$(ts)] [$model seed=$seed] Stage 1 on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -u $BASELINE_DIR/train/train_stage1.py \
        --config "$cfg" --seed $seed --output_dir "$s1dir" \
        > "$LOGDIR/${model}_seed${seed}_s1.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [$model seed=$seed] Stage 1 FAILED"; return 1; fi
    echo "[$(ts)] [$model seed=$seed] Stage 1 done"

    echo "[$(ts)] [$model seed=$seed] Stage 2 on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -u $BASELINE_DIR/train/train_stage2.py \
        --config "$cfg" --seed $seed \
        --checkpoint "$s1dir/best_checkpoint.pt" --output_dir "$s2dir" \
        > "$LOGDIR/${model}_seed${seed}_s2.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [$model seed=$seed] Stage 2 FAILED"; return 1; fi
    echo "[$(ts)] [$model seed=$seed] Stage 2 done"

    echo "[$(ts)] [$model seed=$seed] Eval on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -u $BASELINE_DIR/evaluate/stratified_rollout_v3v7.py \
        --config "$cfg" --seed $seed \
        --checkpoint "$s2dir/best_checkpoint.pt" \
        --data "$DATA_TEST" \
        --output "${base}/seed${seed}_stage2_eval.json" \
        > "$LOGDIR/${model}_seed${seed}_eval.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [$model seed=$seed] Eval FAILED"; return 1; fi
    echo "[$(ts)] [$model seed=$seed] COMPLETE"
}

echo "[$(ts)] ================================"
echo "[$(ts)] MIMIC-IV 2-Seed Full Pipeline"
echo "[$(ts)] Seeds: $SEEDS"
echo "[$(ts)] Main Model: SeMDT (no_sem_wm)"
echo "[$(ts)] Baselines: $MODELS"
echo "[$(ts)] ================================"

PIDS=()

# GPU 1: main model (sequential, heavy)
run_main 0 1 &
PIDS+=($!)
sleep 3
run_main 42 1 &
PIDS+=($!)
sleep 3

# GPU 2: bc, bcq, dqn (4 seeds total)
run_baseline bc 0 2 &
PIDS+=($!)
sleep 3
run_baseline bc 42 2 &
PIDS+=($!)
sleep 3
run_baseline bcq 0 2 &
PIDS+=($!)
sleep 3
run_baseline bcq 42 2 &
PIDS+=($!)
sleep 3
run_baseline dqn 0 2 &
PIDS+=($!)
sleep 3
run_baseline dqn 42 2 &
PIDS+=($!)
sleep 3

# GPU 3: cql, dt, iql, td3bc (8 seeds total)
run_baseline cql 0 3 &
PIDS+=($!)
sleep 3
run_baseline cql 42 3 &
PIDS+=($!)
sleep 3
run_baseline dt 0 3 &
PIDS+=($!)
sleep 3
run_baseline dt 42 3 &
PIDS+=($!)
sleep 3
run_baseline iql 0 3 &
PIDS+=($!)
sleep 3
run_baseline iql 42 3 &
PIDS+=($!)
sleep 3
run_baseline td3bc 0 3 &
PIDS+=($!)
sleep 3
run_baseline td3bc 42 3 &
PIDS+=($!)
sleep 3

echo "[$(ts)] All 16 experiments launched. PIDs: ${PIDS[*]}"

FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid || FAIL=1
done

echo "[$(ts)] ================================"
if [ $FAIL -eq 0 ]; then echo "[$(ts)] ALL DONE"; else echo "[$(ts)] SOME FAILURES - check $LOGDIR/"; fi
echo "[$(ts)] ================================"
