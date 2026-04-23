"""
生成v3格式的语义嵌入用于SEMDT训练
使用定性描述而非数字，与SEMDT设计理念一致
"""
import sys
import os
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts.saps2_qualitative_prompts import build_prompt_sequences_for_trajectory
from prompts.text_encoder import PromptTextEncoder


def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum


def load_and_prepare(path):
    with open(path, "rb") as f:
        trajectories = pickle.load(f)

    # Add returns_to_go if not present
    for traj in trajectories:
        if "returns_to_go" not in traj and "rewards" in traj:
            traj["returns_to_go"] = discount_cumsum(traj["rewards"], 1.0)

    return trajectories


def encode_and_assign(trajectories, encoder, batch_size, output_dtype):
    task_texts = []
    hindsight_texts = []
    foresight_texts = []
    lengths = []

    for traj_idx, traj in enumerate(trajectories):
        rtgs = traj.get("returns_to_go")
        if rtgs is None:
            print(f"Warning: No returns_to_go in traj {traj_idx}, using zeros", flush=True)
            rtgs = np.zeros(traj["dem_observations"].shape[0])

        # V3 requires acuities
        if "acuities" not in traj:
            print(f"Warning: No acuities in traj {traj_idx}, skipping", flush=True)
            continue

        # Convert actions to int64 if needed
        actions = traj.get("actions")
        if actions is not None:
            actions = actions.astype(np.int64)

        try:
            sequences = build_prompt_sequences_for_trajectory(
                states=traj["dem_observations"],
                acuities=traj["acuities"],
                rtgs=rtgs,
                actions=actions,
                max_timestep=max(20, traj["dem_observations"].shape[0]),
            )

            task_texts.extend(sequences["task_prompts"])
            hindsight_texts.extend(sequences["hindsight_prompts"])
            foresight_texts.extend(sequences["foresight_prompts"])
            lengths.append(len(sequences["task_prompts"]))

            # 保存原始prompt用于检查
            traj["v3_task_prompts"] = sequences["task_prompts"]
            traj["v3_hindsight_prompts"] = sequences["hindsight_prompts"]
            traj["v3_foresight_prompts"] = sequences["foresight_prompts"]

        except Exception as e:
            print(f"Error processing traj {traj_idx}: {e}", flush=True)
            lengths.append(0)
            continue

        if (traj_idx + 1) % 100 == 0:
            print(f"[INFO] Processed {traj_idx + 1}/{len(trajectories)} trajectories...", flush=True)

    print(f"[INFO] Encoding {len(task_texts)} task prompts")
    task_embeddings = encoder.encode_texts(task_texts, batch_size=batch_size, output_dtype=output_dtype)
    print(f"[INFO] Encoding {len(hindsight_texts)} hindsight prompts")
    hindsight_embeddings = encoder.encode_texts(
        hindsight_texts, batch_size=batch_size, output_dtype=output_dtype
    )
    print(f"[INFO] Encoding {len(foresight_texts)} foresight prompts")
    foresight_embeddings = encoder.encode_texts(
        foresight_texts, batch_size=batch_size, output_dtype=output_dtype
    )

    offset = 0
    for traj, traj_len in zip(trajectories, lengths):
        if traj_len == 0:
            continue
        next_offset = offset + traj_len
        # 使用v6格式的键名（兼容现有训练脚本），但内容是v3的定性描述
        traj["v6_task_embeddings"] = task_embeddings[offset:next_offset]
        traj["v6_hindsight_embeddings"] = hindsight_embeddings[offset:next_offset]
        traj["v6_foresight_embeddings"] = foresight_embeddings[offset:next_offset]
        traj["v6_prompt_encoder_model"] = encoder.model_name
        # 标记这是v3定性描述
        traj["v6_prompt_version"] = "v3_qualitative"
        offset = next_offset


def main():
    input_dir = "/home/wangmeiyi/AuctionNet/medical/data/phys45"
    output_dir = "../data/v3"
    splits = "train,test"

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading encoder: Qwen/Qwen2.5-0.5B-Instruct")
    encoder = PromptTextEncoder(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda:0",
        max_length=512,
    )
    print(f"[INFO] Hidden size: {encoder.hidden_size}")

    for split in [item.strip() for item in splits.split(",") if item.strip()]:
        input_path = os.path.join(input_dir, f"{split}_Phys45.pickle")
        output_path = os.path.join(output_dir, f"{split}_Phys45_v3.pickle")

        if not os.path.exists(input_path):
            print(f"[WARN] Input file not found: {input_path}, skipping")
            continue

        print(f"[INFO] Processing split={split}: {input_path}")
        trajectories = load_and_prepare(input_path)
        print(f"[INFO] Loaded {len(trajectories)} trajectories")

        encode_and_assign(
            trajectories=trajectories,
            encoder=encoder,
            batch_size=32,
            output_dtype="float16",
        )

        with open(output_path, "wb") as f:
            pickle.dump(trajectories, f)
        print(f"[INFO] Wrote V3 dataset to {output_path}")

        # 打印示例
        print(f"\n[INFO] Sample V3 task prompt: {trajectories[0]['v3_task_prompts'][0][:200]}...")
        print(f"[INFO] Sample V3 hindsight prompt: {trajectories[0]['v3_hindsight_prompts'][0][:200]}...")
        print(f"[INFO] Sample V3 foresight prompt: {trajectories[0]['v3_foresight_prompts'][0][:200]}...")
        print()


if __name__ == "__main__":
    main()
