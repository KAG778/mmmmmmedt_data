#!/bin/bash
# Run all 7 baselines with 2 seeds (seed=0, seed=1) in parallel
# Each model+seed runs: stage1 -> stage2 -> eval independently
# Results saved to baseline_III/results/{model}/seed{N}_stage{X}/
#
# GPU allocation:
#   GPU 0: bc(s0), cql(s0)
#   GPU 1: bc(s1), cql(s1), bcq(s0)
#   GPU 2: bcq(s1), dqn(s0), dqn(s1)
#   GPU 3: dt(s0), dt(s1), iql(s0), iql(s1), td3bc(s0), td3bc(s1)

set -e
cd "$(dirname "$0")"

LOGDIR=logs/2seeds
mkdir -p $LOGDIR

BASELINE_DIR=baseline_III
DATA_TEST="data/v3/test_Phys45_v3.pickle"
MODELS="bc bcq cql dqn dt iql td3bc"
SEEDS="0 1"

ts() { date +%H:%M:%S; }

run_one() {
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
echo "[$(ts)] 2-Seed Baseline Training"
echo "[$(ts)] Models: $MODELS"
echo "[$(ts)] Seeds: $SEEDS"
echo "[$(ts)] ================================"

PIDS=()

# GPU 0: bc seed0, cql seed0
run_one bc 0 0 &
PIDS+=($!)
sleep 3
run_one cql 0 0 &
PIDS+=($!)
sleep 3

# GPU 1: bc seed1, cql seed1, bcq seed0
run_one bc 1 1 &
PIDS+=($!)
sleep 3
run_one cql 1 1 &
PIDS+=($!)
sleep 3
run_one bcq 0 1 &
PIDS+=($!)
sleep 3

# GPU 2: bcq seed1, dqn seed0, dqn seed1
run_one bcq 1 2 &
PIDS+=($!)
sleep 3
run_one dqn 0 2 &
PIDS+=($!)
sleep 3
run_one dqn 1 2 &
PIDS+=($!)
sleep 3

# GPU 3: dt seed0, dt seed1, iql seed0, iql seed1, td3bc seed0, td3bc seed1
run_one dt 0 3 &
PIDS+=($!)
sleep 3
run_one dt 1 3 &
PIDS+=($!)
sleep 3
run_one iql 0 3 &
PIDS+=($!)
sleep 3
run_one iql 1 3 &
PIDS+=($!)
sleep 3
run_one td3bc 0 3 &
PIDS+=($!)
sleep 3
run_one td3bc 1 3 &
PIDS+=($!)
sleep 3

echo "[$(ts)] All 14 experiments launched. PIDs: ${PIDS[*]}"

FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid || FAIL=1
done

echo "[$(ts)] ================================"
if [ $FAIL -eq 0 ]; then echo "[$(ts)] ALL DONE"; else echo "[$(ts)] SOME FAILURES"; fi
echo "[$(ts)] ================================"
