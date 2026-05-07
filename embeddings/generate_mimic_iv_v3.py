"""
Generate v3 embeddings for MIMIC-IV phys45 data.
Input:  other_data/mimic_iv_phys45_deliver/{train,val,test}_Phys45.pickle
Output: data/mimic_iv_v3/{train,val,test}_Phys45_v3.pickle + normalization_params.pkl
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

        if "acuities" not in traj:
            print(f"Warning: No acuities in traj {traj_idx}, skipping", flush=True)
            lengths.append(0)
            continue

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

            traj["v3_task_prompts"] = sequences["task_prompts"]
            traj["v3_hindsight_prompts"] = sequences["hindsight_prompts"]
            traj["v3_foresight_prompts"] = sequences["foresight_prompts"]

        except Exception as e:
            print(f"Error processing traj {traj_idx}: {e}", flush=True)
            lengths.append(0)
            continue

        if (traj_idx + 1) % 500 == 0:
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
        traj["v6_task_embeddings"] = task_embeddings[offset:next_offset]
        traj["v6_hindsight_embeddings"] = hindsight_embeddings[offset:next_offset]
        traj["v6_foresight_embeddings"] = foresight_embeddings[offset:next_offset]
        traj["v6_prompt_encoder_model"] = encoder.model_name
        traj["v6_prompt_version"] = "v3_qualitative"
        # Also set without v6 prefix for compatibility
        traj["task_embeddings"] = task_embeddings[offset:next_offset]
        traj["hindsight_embeddings"] = hindsight_embeddings[offset:next_offset]
        traj["foresight_embeddings"] = foresight_embeddings[offset:next_offset]
        offset = next_offset


def compute_normalization(trajectories):
    all_states = np.concatenate([t["dem_observations"] for t in trajectories], axis=0)
    return {
        "state_mean": all_states.mean(axis=0).astype(np.float32),
        "state_std": all_states.std(axis=0).astype(np.float32),
    }


def main():
    input_dir = "/home/wangmeiyi/AuctionNet/medical/last_exp/other_data/mimic_iv_phys45_deliver"
    output_dir = "/home/wangmeiyi/AuctionNet/medical/last_exp/data/mimic_iv_v3"
    splits = "train,val,test"
    device = "cuda:2"

    os.makedirs(output_dir, exist_ok=True)

    model_path = "/home/wangmeiyi/models/Qwen2.5-0.5B-Instruct"
    print(f"[INFO] Loading encoder: {model_path} on {device}")
    encoder = PromptTextEncoder(
        model_name=model_path,
        device=device,
        max_length=512,
    )
    print(f"[INFO] Hidden size: {encoder.hidden_size}")

    for split in [s.strip() for s in splits.split(",")]:
        input_path = os.path.join(input_dir, f"{split}_Phys45.pickle")
        output_path = os.path.join(output_dir, f"{split}_Phys45_v3.pickle")

        if not os.path.exists(input_path):
            print(f"[WARN] Input not found: {input_path}, skipping")
            continue

        print(f"\n[INFO] === Processing {split} ===")
        trajectories = load_and_prepare(input_path)
        print(f"[INFO] Loaded {len(trajectories)} trajectories")

        encode_and_assign(
            trajectories=trajectories,
            encoder=encoder,
            batch_size=64,
            output_dtype="float16",
        )

        with open(output_path, "wb") as f:
            pickle.dump(trajectories, f)
        print(f"[INFO] Saved {output_path} ({os.path.getsize(output_path)/1e9:.2f} GB)")

        # Print sample
        if trajectories and "v3_task_prompts" in trajectories[0]:
            print(f"[INFO] Sample task prompt: {trajectories[0]['v3_task_prompts'][0][:150]}...")

    # Compute normalization from training set
    print("\n[INFO] Computing normalization params from training set...")
    train_path = os.path.join(output_dir, "train_Phys45_v3.pickle")
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    norm_params = compute_normalization(train_data)
    norm_path = os.path.join(output_dir, "normalization_params.pkl")
    with open(norm_path, "wb") as f:
        pickle.dump(norm_params, f)
    print(f"[INFO] Saved {norm_path}")
    print(f"[INFO] state_mean range: [{norm_params['state_mean'].min():.3f}, {norm_params['state_mean'].max():.3f}]")
    print(f"[INFO] state_std range: [{norm_params['state_std'].min():.3f}, {norm_params['state_std'].max():.3f}]")

    print("\n[INFO] Done!")


if __name__ == "__main__":
    main()
