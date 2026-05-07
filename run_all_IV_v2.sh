#!/bin/bash
# MIMIC-IV Full Pipeline: Main Model + 7 Baselines, 2 seeds (0, 42)
# Adapted for current GPU availability: GPU 1 (32GB free) + GPU 3 (23GB free)
#
# 4 parallel tracks:
#   GPU 1 Track A: main(s0) → main(s42)            [~12GB peak]
#   GPU 1 Track B: bc×2 → bcq×2                    [~4GB, alongside Track A]
#   GPU 3 Track C: dqn×2 → cql×2                   [~5GB]
#   GPU 3 Track D: dt×2 → iql×2 → td3bc×2          [~6GB]

set -e
cd "$(dirname "$0")"

LOGDIR=logs/IV_2seeds
mkdir -p $LOGDIR

MAIN_DIR=main_model_IV/scheme3_cspdt_v2
BASELINE_DIR=baseline_Iv
DATA_TEST="data/mimic_iv_v3/test_Phys45_v3.pickle"
SEEDS="0 42"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# ============================================================
# Main Model: stage1 -> stage2 -> eval
# ============================================================
run_main() {
    local seed=$1 gpu=$2
    local ckpt_base="$MAIN_DIR/checkpoints_IV/seed${seed}"

    echo "[$(ts)] [MAIN seed=$seed GPU=$gpu] Stage 1 starting"
    CUDA_VISIBLE_DEVICES=$gpu python -u $MAIN_DIR/train_stage1_no_sem_epoch.py \
        --seed $seed \
        --logdir "${ckpt_base}_stage1" \
        > "$LOGDIR/main_seed${seed}_s1.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [MAIN seed=$seed] Stage 1 FAILED"; return 1; fi
    echo "[$(ts)] [MAIN seed=$seed] Stage 1 done"

    echo "[$(ts)] [MAIN seed=$seed GPU=$gpu] Stage 2 starting"
    CUDA_VISIBLE_DEVICES=$gpu python -u $MAIN_DIR/train_stage2_no_sem_sigma2.py \
        --seed $seed \
        --logdir "${ckpt_base}_stage2" \
        --policy_ckpt "${ckpt_base}_stage1/epoch_100/policy.pt" \
        --world_model_ckpt "${ckpt_base}_stage1/epoch_100/world_model.pt" \
        > "$LOGDIR/main_seed${seed}_s2.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [MAIN seed=$seed] Stage 2 FAILED"; return 1; fi
    echo "[$(ts)] [MAIN seed=$seed] Stage 2 done"

    echo "[$(ts)] [MAIN seed=$seed GPU=$gpu] Eval starting"
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

    echo "[$(ts)] [$model seed=$seed GPU=$gpu] Stage 1 starting"
    CUDA_VISIBLE_DEVICES=$gpu python -u $BASELINE_DIR/train/train_stage1.py \
        --config "$cfg" --seed $seed --output_dir "$s1dir" \
        > "$LOGDIR/${model}_seed${seed}_s1.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [$model seed=$seed] Stage 1 FAILED"; return 1; fi
    echo "[$(ts)] [$model seed=$seed] Stage 1 done"

    echo "[$(ts)] [$model seed=$seed GPU=$gpu] Stage 2 starting"
    CUDA_VISIBLE_DEVICES=$gpu python -u $BASELINE_DIR/train/train_stage2.py \
        --config "$cfg" --seed $seed \
        --checkpoint "$s1dir/best_checkpoint.pt" --output_dir "$s2dir" \
        > "$LOGDIR/${model}_seed${seed}_s2.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [$model seed=$seed] Stage 2 FAILED"; return 1; fi
    echo "[$(ts)] [$model seed=$seed] Stage 2 done"

    echo "[$(ts)] [$model seed=$seed GPU=$gpu] Eval starting"
    CUDA_VISIBLE_DEVICES=$gpu python -u $BASELINE_DIR/evaluate/stratified_rollout_v3v7.py \
        --config "$cfg" --seed $seed \
        --checkpoint "$s2dir/best_checkpoint.pt" \
        --data "$DATA_TEST" \
        --output "${base}/seed${seed}_stage2_eval.json" \
        > "$LOGDIR/${model}_seed${seed}_eval.log" 2>&1
    if [ $? -ne 0 ]; then echo "[$(ts)] [$model seed=$seed] Eval FAILED"; return 1; fi
    echo "[$(ts)] [$model seed=$seed] COMPLETE"
}

# ============================================================
# Sequential track runner
# ============================================================
track_main_a() {
    echo "[$(ts)] === Track A (GPU 1): Main Model ==="
    run_main 0 1
    run_main 42 1
    echo "[$(ts)] === Track A DONE ==="
}

track_gpu1_b() {
    echo "[$(ts)] === Track B (GPU 1): BC + BCQ ==="
    run_baseline bc 0 1
    run_baseline bc 42 1
    run_baseline bcq 0 1
    run_baseline bcq 42 1
    echo "[$(ts)] === Track B DONE ==="
}

track_gpu3_c() {
    echo "[$(ts)] === Track C (GPU 3): DQN + CQL ==="
    run_baseline dqn 0 3
    run_baseline dqn 42 3
    run_baseline cql 0 3
    run_baseline cql 42 3
    echo "[$(ts)] === Track C DONE ==="
}

track_gpu3_d() {
    echo "[$(ts)] === Track D (GPU 3): DT + IQL + TD3BC ==="
    run_baseline dt 0 3
    run_baseline dt 42 3
    run_baseline iql 0 3
    run_baseline iql 42 3
    run_baseline td3bc 0 3
    run_baseline td3bc 42 3
    echo "[$(ts)] === Track D DONE ==="
}

# ============================================================
# Launch
# ============================================================
echo "[$(ts)] ==============================================="
echo "[$(ts)] MIMIC-IV 2-Seed Full Pipeline (GPU-adapted)"
echo "[$(ts)] GPU 1: Track A (main×2) + Track B (bc×2, bcq×2)"
echo "[$(ts)] GPU 3: Track C (dqn×2, cql×2) + Track D (dt×2, iql×2, td3bc×2)"
echo "[$(ts)] Total: 16 experiments across 4 tracks"
echo "[$(ts)] ==============================================="

mkdir -p $LOGDIR

# GPU 1: two parallel tracks
track_main_a > "$LOGDIR/track_A_main.log" 2>&1 &
PID_A=$!
sleep 5

track_gpu1_b > "$LOGDIR/track_B_bc_bcq.log" 2>&1 &
PID_B=$!
sleep 5

# GPU 3: two parallel tracks
track_gpu3_c > "$LOGDIR/track_C_dqn_cql.log" 2>&1 &
PID_C=$!
sleep 5

track_gpu3_d > "$LOGDIR/track_D_dt_iql_td3bc.log" 2>&1 &
PID_D=$!
sleep 5

echo "[$(ts)] All 4 tracks launched."
echo "[$(ts)]   Track A (main×2, GPU 1):  PID=$PID_A"
echo "[$(ts)]   Track B (bc+bcq, GPU 1):  PID=$PID_B"
echo "[$(ts)]   Track C (dqn+cql, GPU 3): PID=$PID_C"
echo "[$(ts)]   Track D (dt+iql+td3bc, GPU 3): PID=$PID_D"
echo "[$(ts)] Monitor: tail -f $LOGDIR/track_*.log"

FAIL=0
for pid in $PID_A $PID_B $PID_C $PID_D; do
    wait $pid || FAIL=1
done

echo ""
echo "[$(ts)] ==============================================="
if [ $FAIL -eq 0 ]; then
    echo "[$(ts)] ALL 16 EXPERIMENTS DONE"
else
    echo "[$(ts)] SOME FAILURES - check $LOGDIR/"
    echo "[$(ts)] Track A (main):    $(kill -0 $PID_A 2>/dev/null && echo 'RUNNING' || echo 'DONE')"
    echo "[$(ts)] Track B (bc/bcq):  $(kill -0 $PID_B 2>/dev/null && echo 'RUNNING' || echo 'DONE')"
    echo "[$(ts)] Track C (dqn/cql): $(kill -0 $PID_C 2>/dev/null && echo 'RUNNING' || echo 'DONE')"
    echo "[$(ts)] Track D (dt/iql/td3bc): $(kill -0 $PID_D 2>/dev/null && echo 'RUNNING' || echo 'DONE')"
fi
echo "[$(ts)] ==============================================="
