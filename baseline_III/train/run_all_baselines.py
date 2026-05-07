"""
Automated pipeline: Stage1 -> Stage2 -> Evaluation for all baselines.
Parallelizes across available GPUs.
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
LOG_DIR = Path("/tmp/training_logs_persistent")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# All baseline models to train
MODELS = ["bc", "dt", "iql", "bcq", "cql", "dqn", "td3bc"]

# GPU assignments (will be filled dynamically)
GPU_MAP = {}


def get_free_gpus(min_memory_free=20000):
    """Get list of GPU indices with enough free memory."""
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        idx, mem_free = line.split(",")
        if int(mem_free.strip()) >= min_memory_free:
            gpus.append(int(idx.strip()))
    return gpus


def run_cmd(cmd, log_file, gpu_id):
    """Run a command with CUDA_VISIBLE_DEVICES set."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU {gpu_id}] {cmd}")
    print(f"  Log: {log_file}")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR)
        )
    return proc


def wait_for_procs(procs_info):
    """Wait for all processes to finish. Return list of (model, stage, success)."""
    results = []
    while procs_info:
        for info in list(procs_info):
            model, stage, gpu_id, proc, log_file = info
            ret = proc.poll()
            if ret is not None:
                success = (ret == 0)
                status = "OK" if success else "FAILED"
                print(f"[GPU {gpu_id}] {model} {stage}: {status} (exit={ret})")
                results.append((model, stage, success))
                procs_info.remove(info)
        time.sleep(10)
    return results


def check_stage1_done(model):
    """Check if stage1 checkpoint exists."""
    ckpt = BASE_DIR / "results" / model / "stage1" / "best_checkpoint.pt"
    return ckpt.exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to train")
    parser.add_argument("--semantic", action="store_true", help="Use semantic stage2")
    parser.add_argument("--skip_stage1", action="store_true", help="Skip stage1 if checkpoint exists")
    parser.add_argument("--skip_stage2", action="store_true", help="Skip stage2")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation")
    args = parser.parse_args()

    models = args.models
    print(f"\n{'='*60}")
    print(f"Automated Baseline Training Pipeline")
    print(f"Models: {models}")
    print(f"Semantic stage2: {args.semantic}")
    print(f"{'='*60}\n")

    # ============================================================
    # Phase 1: Stage 1 Training (parallel across GPUs)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Phase 1: Stage 1 Training")
    print(f"{'='*60}\n")

    gpus = get_free_gpus(min_memory_free=10000)
    print(f"Available GPUs: {gpus}")

    if not gpus:
        print("No free GPUs! Exiting.")
        return

    stage1_procs = []
    gpu_idx = 0

    for model in models:
        if args.skip_stage1 and check_stage1_done(model):
            print(f"[SKIP] {model} stage1 - checkpoint exists")
            continue

        gpu_id = gpus[gpu_idx % len(gpus)]
        config = f"configs/{model}.yaml"
        log_file = LOG_DIR / f"{model}_stage1.log"

        cmd = f"python -u train/train_stage1.py --config {config}"
        proc = run_cmd(cmd, str(log_file), gpu_id)
        stage1_procs.append((model, "stage1", gpu_id, proc, str(log_file)))

        gpu_idx += 1
        # Stagger launches to avoid memory spikes
        time.sleep(5)

    # Wait for all stage1 to complete
    stage1_results = wait_for_procs(stage1_procs)

    # Check which models succeeded
    successful_models = [m for m, s, ok in stage1_results if s == "stage1" and ok]
    skipped_models = [m for m in models if args.skip_stage1 and check_stage1_done(m)]
    all_ready = list(set(successful_models + skipped_models))

    print(f"\nStage1 complete. Ready for stage2: {all_ready}")

    if not all_ready:
        print("No models ready for stage2. Exiting.")
        return

    # ============================================================
    # Phase 2: Stage 2 Training (parallel across GPUs)
    # ============================================================
    if not args.skip_stage2:
        print(f"\n{'='*60}")
        print(f"Phase 2: Stage 2 Training")
        print(f"{'='*60}\n")

        # Re-check free GPUs
        gpus = get_free_gpus(min_memory_free=10000)
        print(f"Available GPUs: {gpus}")

        stage2_procs = []
        gpu_idx = 0

        for model in all_ready:
            gpu_id = gpus[gpu_idx % len(gpus)]
            config = f"configs/{model}.yaml"
            checkpoint = f"results/{model}/stage1/best_checkpoint.pt"
            log_file = LOG_DIR / f"{model}_stage2.log"

            semantic_flag = "--semantic" if args.semantic else ""
            cmd = f"python -u train/train_stage2.py --config {config} --checkpoint {checkpoint} {semantic_flag}"
            proc = run_cmd(cmd, str(log_file), gpu_id)
            stage2_procs.append((model, "stage2", gpu_id, proc, str(log_file)))

            gpu_idx += 1
            time.sleep(5)

        stage2_results = wait_for_procs(stage2_procs)
        eval_ready = [m for m, s, ok in stage2_results if s == "stage2" and ok]
        print(f"\nStage2 complete. Ready for evaluation: {eval_ready}")
    else:
        eval_ready = all_ready

    # ============================================================
    # Phase 3: Evaluation
    # ============================================================
    if not args.skip_eval and eval_ready:
        print(f"\n{'='*60}")
        print(f"Phase 3: Evaluation")
        print(f"{'='*60}\n")

        gpus = get_free_gpus(min_memory_free=5000)
        print(f"Available GPUs: {gpus}")

        eval_procs = []
        gpu_idx = 0

        for model in eval_ready:
            gpu_id = gpus[gpu_idx % len(gpus)] if gpus else 0
            config = f"configs/{model}.yaml"
            checkpoint = f"results/{model}/stage2/best_checkpoint.pt"
            log_file = LOG_DIR / f"{model}_eval.log"

            eval_script = BASE_DIR / "evaluate" / "stratified_rollout_v3v7.py"
            if eval_script.exists():
                cmd = f"python -u {eval_script} --config {config} --checkpoint {checkpoint}"
                proc = run_cmd(cmd, str(log_file), gpu_id)
                eval_procs.append((model, "eval", gpu_id, proc, str(log_file)))
                gpu_idx += 1
            else:
                print(f"[SKIP] {model} eval - script not found: {eval_script}")

        if eval_procs:
            eval_results = wait_for_procs(eval_procs)
            for model, stage, ok in eval_results:
                status = "OK" if ok else "FAILED"
                print(f"  {model} eval: {status}")

    print(f"\n{'='*60}")
    print(f"All done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
