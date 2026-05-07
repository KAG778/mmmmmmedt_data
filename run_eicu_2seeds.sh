#!/bin/bash
# Run eICU experiments: 7 baselines × 2 seeds + main model × 2 seeds
# Seeds: 0 and 1 (consistent with MIMIC-III run_2seeds.sh)
#
# GPU allocation:
#   GPU 1: main(seed0), main(seed1)
#   GPU 2: bc(s0,s1), bcq(s0,s1)
#   GPU 3: cql(s0,s1), dqn(s0,s1), dt(s0,s1)
#   GPU 0: iql(s0), td3bc(s0)  (GPU0 has ~20GB free, fewer tasks)

set -e
cd "$(dirname "$0")"

LOGDIR=logs/eicu_2seeds
mkdir -p $LOGDIR

BASELINE_DIR=baseline_III_eicu
MAIN_DIR=main_model_eicu/scheme3_cspdt_v2
DATA_TEST="data/eicu_v3/test_Phys45_v3.pickle"
MODELS="bc bcq cql dqn dt iql td3bc"
SEEDS="0 1"

ts() { date +%H:%M:%S; }

# ── Baseline: stage1 -> stage2 -> eval ──
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

# ── Main model: stage1 -> stage2 -> eval ──
run_main() {
    local seed=$1 gpu=$2
    local ckpt_base="$MAIN_DIR/checkpoints_seed${seed}"
    local res_base="$MAIN_DIR/results_seed${seed}"

    echo "[$(ts)] [MAIN seed=$seed] Stage 1 on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -u $MAIN_DIR/train_stage1_no_sem_epoch.py \
        --seed $seed \
        --logdir "${ckpt_base}/stage1" \
        > "$LOGDIR/main_seed${seed}_s1.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [MAIN seed=$seed] Stage 1 FAILED"; return 1; fi
    echo "[$(ts)] [MAIN seed=$seed] Stage 1 done"

    # Find latest stage1 checkpoint
    STAGE1_CKPT=$(ls -d "${ckpt_base}/stage1"/epoch_* 2>/dev/null | sed 's/.*epoch_//' | sort -n | tail -1)
    if [ -z "$STAGE1_CKPT" ]; then
        echo "[$(ts)] [MAIN seed=$seed] ERROR: No Stage1 checkpoint found"
        return 1
    fi
    STAGE1_CKPT="${ckpt_base}/stage1/epoch_${STAGE1_CKPT}"
    echo "[$(ts)] [MAIN seed=$seed] Using checkpoint: $STAGE1_CKPT"

    echo "[$(ts)] [MAIN seed=$seed] Stage 2 on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -u $MAIN_DIR/train_stage2_no_sem_epoch.py \
        --seed $seed \
        --policy_ckpt "$STAGE1_CKPT/policy.pt" \
        --world_model_ckpt "$STAGE1_CKPT/world_model.pt" \
        --logdir "${ckpt_base}/stage2" \
        > "$LOGDIR/main_seed${seed}_s2.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [MAIN seed=$seed] Stage 2 FAILED"; return 1; fi
    echo "[$(ts)] [MAIN seed=$seed] Stage 2 done"

    echo "[$(ts)] [MAIN seed=$seed] Eval on GPU $gpu"
    STAGE2_CKPT="${ckpt_base}/stage2/best_checkpoint.pt"
    if [ ! -f "$STAGE2_CKPT" ]; then
        STAGE2_CKPT=$(ls -d "${ckpt_base}/stage2"/epoch_* 2>/dev/null | sort | tail -1)
    fi
    CUDA_VISIBLE_DEVICES=$gpu python -u $MAIN_DIR/stratified_rollout_no_sem.py \
        --checkpoint "$STAGE2_CKPT" \
        --data "$DATA_TEST" \
        --output "${res_base}/stage2_eval.json" \
        > "$LOGDIR/main_seed${seed}_eval.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [MAIN seed=$seed] Eval FAILED"; return 1; fi
    echo "[$(ts)] [MAIN seed=$seed] COMPLETE"
}

echo "[$(ts)] ================================"
echo "[$(ts)] eICU 2-Seed Experiments"
echo "[$(ts)] Models: $MODELS"
echo "[$(ts)] Seeds: $SEEDS"
echo "[$(ts)] GPU 1: main(s0, s1)"
echo "[$(ts)] GPU 2: bc(s0,s1), bcq(s0,s1)"
echo "[$(ts)] GPU 3: cql(s0,s1), dqn(s0,s1), dt(s0,s1)"
echo "[$(ts)] GPU 0: iql(s0,s1), td3bc(s0,s1)"
echo "[$(ts)] ================================"

PIDS=()

# GPU 1: Main model (needs more memory for Qwen encoder)
run_main 0 1 &
PIDS+=($!)
sleep 5
run_main 1 1 &
PIDS+=($!)
sleep 5

# GPU 2: bc, bcq
run_baseline bc 0 2 &
PIDS+=($!)
sleep 3
run_baseline bc 1 2 &
PIDS+=($!)
sleep 3
run_baseline bcq 0 2 &
PIDS+=($!)
sleep 3
run_baseline bcq 1 2 &
PIDS+=($!)
sleep 3

# GPU 3: cql, dqn, dt
run_baseline cql 0 3 &
PIDS+=($!)
sleep 3
run_baseline cql 1 3 &
PIDS+=($!)
sleep 3
run_baseline dqn 0 3 &
PIDS+=($!)
sleep 3
run_baseline dqn 1 3 &
PIDS+=($!)
sleep 3
run_baseline dt 0 3 &
PIDS+=($!)
sleep 3
run_baseline dt 1 3 &
PIDS+=($!)
sleep 3

# GPU 0: iql, td3bc (less free memory)
run_baseline iql 0 0 &
PIDS+=($!)
sleep 3
run_baseline iql 1 0 &
PIDS+=($!)
sleep 3
run_baseline td3bc 0 0 &
PIDS+=($!)
sleep 3
run_baseline td3bc 1 0 &
PIDS+=($!)
sleep 3

echo "[$(ts)] All 16 experiments launched (7 baselines × 2 seeds + 1 main × 2 seeds). PIDs: ${PIDS[*]}"

FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid || FAIL=1
done

echo "[$(ts)] ================================"
if [ $FAIL -eq 0 ]; then
    echo "[$(ts)] ALL DONE"
else
    echo "[$(ts)] SOME FAILURES - check $LOGDIR/"
fi
echo "[$(ts)] ================================"
