"""
Generate V6 template embeddings for CSP-DT models

Uses saps6_clinical_scenario_prompts.py to generate text prompts,
then encodes them using Qwen model.
"""

import argparse
import os
import sys
import pickle
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts.saps6_clinical_scenario_prompts import build_prompt_sequences_for_trajectory
from prompts.text_encoder import PromptTextEncoder


def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum


def _load_and_prepare(path):
    with open(path, "rb") as f:
        trajectories = pickle.load(f)

    # Add returns_to_go if not present
    for traj in trajectories:
        if "returns_to_go" not in traj and "rewards" in traj:
            traj["returns_to_go"] = discount_cumsum(traj["rewards"], 1.0)

    return trajectories


def _encode_and_assign(trajectories, encoder, batch_size, output_dtype):
    task_texts = []
    hindsight_texts = []
    foresight_texts = []
    lengths = []

    for traj in trajectories:
        rtgs = traj.get("returns_to_go")
        if rtgs is None:
            print("Warning: No returns_to_go, using zeros")
            rtgs = np.zeros(traj["dem_observations"].shape[0])

        # V6 template requires acuities
        if "acuities" not in traj:
            print("Warning: No acuities in trajectory, skipping")
            continue

        # Convert actions to int64 if needed
        actions = traj.get("actions")
        if actions is not None:
            actions = actions.astype(np.int64)

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
        traj["v6_task_prompts"] = sequences["task_prompts"]
        traj["v6_hindsight_prompts"] = sequences["hindsight_prompts"]
        traj["v6_foresight_prompts"] = sequences["foresight_prompts"]

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
        next_offset = offset + traj_len
        traj["v6_task_embeddings"] = task_embeddings[offset:next_offset]
        traj["v6_hindsight_embeddings"] = hindsight_embeddings[offset:next_offset]
        traj["v6_foresight_embeddings"] = foresight_embeddings[offset:next_offset]
        traj["v6_prompt_encoder_model"] = encoder.model_name
        offset = next_offset


def main():
    parser = argparse.ArgumentParser(
        description="Generate V6 template embeddings for CSP-DT models"
    )
    parser.add_argument("--input_dir", type=str, default="/home/wangmeiyi/AuctionNet/medical/data/phys45")
    parser.add_argument("--output_dir", type=str, default="../data/v6")
    parser.add_argument("--splits", type=str, default="train")
    parser.add_argument("--encoder_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Loading encoder: {args.encoder_model}")
    encoder = PromptTextEncoder(
        model_name=args.encoder_model,
        device=args.device,
        max_length=args.max_length,
    )
    print(f"[INFO] Hidden size: {encoder.hidden_size}")

    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        input_path = os.path.join(args.input_dir, f"{split}_Phys45.pickle")
        output_path = os.path.join(args.output_dir, f"{split}_Phys45_v6.pickle")

        if not os.path.exists(input_path):
            print(f"[WARN] Input file not found: {input_path}, skipping")
            continue

        print(f"[INFO] Processing split={split}: {input_path}")
        trajectories = _load_and_prepare(input_path)
        print(f"[INFO] Loaded {len(trajectories)} trajectories")

        _encode_and_assign(
            trajectories=trajectories,
            encoder=encoder,
            batch_size=args.batch_size,
            output_dtype=args.output_dtype,
        )

        with open(output_path, "wb") as f:
            pickle.dump(trajectories, f)
        print(f"[INFO] Wrote V6 dataset to {output_path}")
        print(f"[INFO] Sample V6 task prompt: {trajectories[0]['v6_task_prompts'][0][:200]}...")
        print(f"[INFO] Sample V6 foresight prompt: {trajectories[0]['v6_foresight_prompts'][0][:200]}...")
        print()


if __name__ == "__main__":
    main()
