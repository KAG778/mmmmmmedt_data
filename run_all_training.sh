#!/bin/bash
# ============================================================
# Master Training Script with Error Monitoring & Retry
#
# GPU allocation:
#   GPU 0: scheme3_cspdt_v2
#   GPU 1: scheme3_cspdt_v2 _ATG
#   GPU 2: scheme3_cspdt_v2 _ATG_B
#   GPU 3: scheme3_cspdt_v2 双dt
#
# After scheme3 finish (GPU 3 freed):
#   GPU 3: SemDT_优化 -> baseline (BC, BCQ, DT, IQL)
#
# Usage:
#   bash run_all_training.sh              # run all
#   bash run_all_training.sh stage1       # only stage1
#   bash run_all_training.sh stage2       # only stage2
#   bash run_all_training.sh monitor      # check status
#   bash run_all_training.sh kill         # stop everything
# ============================================================

BASE="/home/wangmeiyi/AuctionNet/medical/重构之后的代码/优化主要模型"
BASELINE="/home/wangmeiyi/AuctionNet/medical/重构之后的代码/baseline"
RUNDIR="/home/wangmeiyi/AuctionNet/medical/重构之后的代码"
STATUS_FILE="/tmp/training_status.txt"
LOGDIR="/tmp/training_logs_persistent"
mkdir -p "$LOGDIR"

STAGE="${1:-all}"

# ============================================================
# Status tracking
# ============================================================
update_status() {
    local model="$1" stage="$2" status="$3" extra="$4"
    local ts=$(date '+%Y-%m-%d %H:%M:%S')
    # Remove old entry for this model+stage
    if [ -f "$STATUS_FILE" ]; then
        grep -v "^${model}|${stage}|" "$STATUS_FILE" > "${STATUS_FILE}.tmp" 2>/dev/null || true
        mv "${STATUS_FILE}.tmp" "$STATUS_FILE" 2>/dev/null || true
    fi
    echo "${model}|${stage}|${status}|${ts}|${extra}" >> "$STATUS_FILE"
}

get_status() {
    local model="$1" stage="$2"
    if [ -f "$STATUS_FILE" ]; then
        grep "^${model}|${stage}|" "$STATUS_FILE" | tail -1
    fi
}

log_error() {
    local model="$1" msg="$2"
    local ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[ERROR][$ts][$model] $msg" >> "$LOGDIR/errors.log"
    echo "[ERROR][$ts][$model] $msg" >> "$LOGDIR/master.log"
}

log_info() {
    local msg="$1"
    local ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[INFO][$ts] $msg" | tee -a "$LOGDIR/master.log"
}

# ============================================================
# Helper: run with retry
# ============================================================
MAX_RETRIES=2

run_with_retry() {
    local model="$1" gpu="$2" shift_count=2
    shift 2
    local cmd="$@"
    local attempt=1

    while [ $attempt -le $MAX_RETRIES ]; do
        log_info "[$model] Attempt $attempt/$MAX_RETRIES: $cmd (GPU $gpu)"
        if CUDA_VISIBLE_DEVICES=$gpu $cmd >> "$LOGDIR/${model}.log" 2>&1; then
            return 0
        else
            local exit_code=$?
            log_error "$model" "Attempt $attempt failed (exit=$exit_code): $cmd"
            if [ $attempt -lt $MAX_RETRIES ]; then
                log_info "[$model] Retrying in 30s..."
                sleep 30
            fi
            attempt=$((attempt + 1))
        fi
    done
    log_error "$model" "All $MAX_RETRIES attempts exhausted: $cmd"
    return 1
}

# ============================================================
# Helper: find latest stage1 checkpoint
# ============================================================
find_latest_stage1() {
    local dir="$1"
    # Find step dirs, sort by step number descending, pick highest
    find "$dir/checkpoints/stage1" -maxdepth 1 -name "step_*" -type d 2>/dev/null \
        | sed 's/.*step_//' | sort -rn | head -1 \
        | xargs -I{} echo "$dir/checkpoints/stage1/step_{}"
}

# ============================================================
# Per-model training function
# scheme3_cspdt_v2, ATG, ATG_B: use train_stage1.py + train_stage2.py
# 双dt: same but uses WorldDT internally
# ============================================================
run_scheme3_model() {
    local model_name="$1"
    local gpu="$2"
    local model_dir="$3"
    local exit_code=0

    log_info "[$model_name] === Starting (GPU $gpu) ==="
    cd "$model_dir"

    # --- Stage 1 ---
    if [ "$STAGE" != "stage2" ]; then
        update_status "$model_name" "stage1" "RUNNING" "GPU $gpu"
        log_info "[$model_name] Stage1 starting..."

        if run_with_retry "$model_name" "$gpu" python train_stage1.py --logdir ./checkpoints/stage1; then
            update_status "$model_name" "stage1" "DONE" ""
            log_info "[$model_name] Stage1 complete."
        else
            update_status "$model_name" "stage1" "FAILED" "see $LOGDIR/${model_name}.log"
            log_error "$model_name" "Stage1 failed after retries"
            exit_code=1
        fi
    fi

    # --- Stage 2 ---
    if [ "$STAGE" != "stage1" ] && [ $exit_code -eq 0 ]; then
        local CKPT=$(find_latest_stage1 ".")
        if [ -z "$CKPT" ] || [ ! -f "$CKPT/policy.pt" ]; then
            log_error "$model_name" "No valid stage1 checkpoint found"
            update_status "$model_name" "stage2" "FAILED" "no stage1 ckpt"
            exit_code=1
        else
            update_status "$model_name" "stage2" "RUNNING" "GPU $gpu, ckpt=$CKPT"
            log_info "[$model_name] Stage2 starting with $CKPT..."

            if run_with_retry "$model_name" "$gpu" python train_stage2.py \
                --logdir ./checkpoints/stage2 \
                --policy_ckpt "$CKPT/policy.pt" \
                --world_model_ckpt "$CKPT/world_model.pt"; then
                update_status "$model_name" "stage2" "DONE" ""
                log_info "[$model_name] Stage2 complete."
            else
                update_status "$model_name" "stage2" "FAILED" "see $LOGDIR/${model_name}.log"
                log_error "$model_name" "Stage2 failed after retries"
                exit_code=1
            fi
        fi
    fi

    log_info "[$model_name] === Finished (exit=$exit_code) ==="
    return $exit_code
}

# ============================================================
# SemDT_优化
# ============================================================
run_semdt() {
    local model_name="semdt"
    local gpu=3
    local exit_code=0

    log_info "[$model_name] === Starting (GPU $gpu) ==="
    cd "$BASE/SemDT_优化"

    if [ "$STAGE" != "stage2" ]; then
        update_status "$model_name" "stage1" "RUNNING" "GPU $gpu"
        log_info "[$model_name] Stage1 starting..."

        if run_with_retry "$model_name" "$gpu" python train_stage1_with_semantic.py --config semdt_v3.yaml; then
            update_status "$model_name" "stage1" "DONE" ""
            log_info "[$model_name] Stage1 complete."
        else
            update_status "$model_name" "stage1" "FAILED" ""
            exit_code=1
        fi
    fi

    if [ "$STAGE" != "stage1" ] && [ $exit_code -eq 0 ]; then
        local CKPT="../results2/semdt_v3/stage1/best_checkpoint.pt"
        if [ ! -f "$CKPT" ]; then
            log_error "$model_name" "No stage1 checkpoint at $CKPT"
            update_status "$model_name" "stage2" "FAILED" "no ckpt"
            exit_code=1
        else
            update_status "$model_name" "stage2" "RUNNING" "GPU $gpu"
            log_info "[$model_name] Stage2 starting..."

            if run_with_retry "$model_name" "$gpu" python train_stage2_with_semantic.py \
                --config semdt_v3.yaml --checkpoint "$CKPT"; then
                update_status "$model_name" "stage2" "DONE" ""
                log_info "[$model_name] Stage2 complete."
            else
                update_status "$model_name" "stage2" "FAILED" ""
                exit_code=1
            fi
        fi
    fi

    log_info "[$model_name] === Finished (exit=$exit_code) ==="
    return $exit_code
}

# ============================================================
# Baseline models (BC, BCQ, DT, IQL)
# ============================================================
run_baselines() {
    local gpu=3
    cd "$BASELINE"

    for model in bc bcq dt iql; do
        local model_name="baseline_${model}"
        local exit_code=0

        log_info "[$model_name] === Starting (GPU $gpu) ==="

        if [ "$STAGE" != "stage2" ]; then
            update_status "$model_name" "stage1" "RUNNING" "GPU $gpu"
            log_info "[$model_name] Stage1 starting..."

            if run_with_retry "$model_name" "$gpu" python train/train_stage1.py --config "configs/${model}.yaml"; then
                update_status "$model_name" "stage1" "DONE" ""
                log_info "[$model_name] Stage1 complete."
            else
                update_status "$model_name" "stage1" "FAILED" ""
                exit_code=1
            fi
        fi

        if [ "$STAGE" != "stage1" ] && [ $exit_code -eq 0 ]; then
            local CKPT="results/${model}/stage1/best_checkpoint.pt"
            if [ ! -f "$CKPT" ]; then
                log_error "$model_name" "No stage1 checkpoint at $CKPT"
                update_status "$model_name" "stage2" "FAILED" "no ckpt"
                exit_code=1
            else
                update_status "$model_name" "stage2" "RUNNING" "GPU $gpu"
                log_info "[$model_name] Stage2 starting..."

                if run_with_retry "$model_name" "$gpu" python train/train_stage2.py \
                    --config "configs/${model}.yaml" --checkpoint "$CKPT" --semantic; then
                    update_status "$model_name" "stage2" "DONE" ""
                    log_info "[$model_name] Stage2 complete."
                else
                    update_status "$model_name" "stage2" "FAILED" ""
                    exit_code=1
                fi
            fi
        fi

        log_info "[$model_name] === Finished (exit=$exit_code) ==="
    done
}

# ============================================================
# Monitor: show status of all models
# ============================================================
show_monitor() {
    echo "=========================================="
    echo "  Training Monitor - $(date)"
    echo "=========================================="
    echo ""

    # Show status file
    if [ -f "$STATUS_FILE" ]; then
        printf "%-18s %-8s %-10s %-20s %s\n" "MODEL" "STAGE" "STATUS" "TIME" "INFO"
        echo "-----------------------------------------------------------------------------------"
        while IFS='|' read -r model stage status ts extra; do
            printf "%-18s %-8s %-10s %-20s %s\n" "$model" "$stage" "$status" "$ts" "$extra"
        done < "$STATUS_FILE"
    else
        echo "No status file found. Training not started?"
    fi

    # Show GPU status
    echo ""
    echo "--- GPU Status ---"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"

    # Show latest log line per model
    echo ""
    echo "--- Latest Log Lines ---"
    for logf in "$LOGDIR"/*.log; do
        [ "$(basename $logf)" = "master.log" ] && continue
        [ "$(basename $logf)" = "errors.log" ] && continue
        local lastline=$(tail -1 "$logf" 2>/dev/null)
        if [ -n "$lastline" ]; then
            echo "  $(basename $logf .log): $lastline"
        fi
    done

    # Show errors
    if [ -f "$LOGDIR/errors.log" ] && [ -s "$LOGDIR/errors.log" ]; then
        echo ""
        echo "--- Recent Errors ---"
        tail -5 "$LOGDIR/errors.log"
    fi

    # Show running processes
    echo ""
    echo "--- Running Processes ---"
    ps aux | grep "train_stage" | grep -v grep | awk '{print "  PID="$2, "CMD="$11, $12, $13, $14}' | head -10

    echo ""
    echo "=========================================="
}

# ============================================================
# Kill: stop all training processes
# ============================================================
kill_all() {
    echo "Stopping all training processes..."
    pkill -f "train_stage" 2>/dev/null && echo "Killed train_stage processes" || echo "No train_stage processes found"
    pkill -f "run_all_training" 2>/dev/null && echo "Killed run_all_training processes" || echo "No run_all_training processes found"
    echo "Done. Check 'bash run_all_training.sh monitor' to confirm."
}

# ============================================================
# Main dispatch
# ============================================================
case "$STAGE" in
    monitor)
        show_monitor
        ;;
    kill)
        kill_all
        ;;
    *)
        # --- Actual training ---
        log_info "=========================================="
        log_info "Master Training Launch"
        log_info "Stage: $STAGE"
        log_info "=========================================="

        # Clean old status
        > "$STATUS_FILE"
        > "$LOGDIR/errors.log"

        # Phase 1: 4 scheme3 models in parallel (GPU 0-3)
        log_info "Phase 1: Launching 4 scheme3 models in parallel..."

        run_scheme3_model "v2"       0 "$BASE/scheme3_cspdt_v2"        >> "$LOGDIR/master.log" 2>&1 &
        P_V2=$!

        run_scheme3_model "atg"      1 "$BASE/scheme3_cspdt_v2 _ATG"   >> "$LOGDIR/master.log" 2>&1 &
        P_ATG=$!

        run_scheme3_model "atg_b"    2 "$BASE/scheme3_cspdt_v2 _ATG_B" >> "$LOGDIR/master.log" 2>&1 &
        P_ATGB=$!

        run_scheme3_model "dual_dt"  3 "$BASE/scheme3_cspdt_v2 双dt"   >> "$LOGDIR/master.log" 2>&1 &
        P_DDT=$!

        log_info "Waiting for scheme3 models (PIDs: $P_V2 $P_ATG $P_ATGB $P_DDT)..."

        # Wait and track results
        FAILURES=0
        wait $P_V2   && log_info "[v2]      Complete"      || { log_info "[v2]      FAILED";  FAILURES=$((FAILURES+1)); }
        wait $P_ATG   && log_info "[ATG]     Complete"      || { log_info "[ATG]     FAILED";  FAILURES=$((FAILURES+1)); }
        wait $P_ATGB  && log_info "[ATG_B]   Complete"      || { log_info "[ATG_B]   FAILED";  FAILURES=$((FAILURES+1)); }
        wait $P_DDT   && log_info "[双dt]    Complete"      || { log_info "[双dt]    FAILED";  FAILURES=$((FAILURES+1)); }

        log_info "Phase 1 done. Failures: $FAILURES / 4"

        # Phase 2: SemDT_优化 on GPU 3
        log_info "Phase 2: SemDT_优化 on GPU 3..."
        run_semdt >> "$LOGDIR/master.log" 2>&1
        log_info "[SemDT] Done."

        # Phase 3: Baseline models on GPU 3 (sequential)
        log_info "Phase 3: Baseline models on GPU 3..."
        run_baselines >> "$LOGDIR/master.log" 2>&1
        log_info "[Baselines] Done."

        # Final summary
        log_info ""
        log_info "=========================================="
        log_info "ALL TRAINING COMPLETE"
        log_info "=========================================="
        show_monitor
        ;;
esac
