"""
Generate v7 state-only embeddings for SeMDT training.
Uses only 45-dim state features (no acuities needed).
"""
import sys
import os
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts.state_only_prompts_v7 import build_prompt_sequences_v7
from prompts.text_encoder import PromptTextEncoder


def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum


def generate_v7_embeddings(data_path, output_path, batch_size=32, output_dtype="float16"):
    print(f"Loading data from: {data_path}")
    with open(data_path, 'rb') as f:
        trajectories = pickle.load(f)

    # Add returns_to_go if missing
    for traj in trajectories:
        if "returns_to_go" not in traj and "rewards" in traj:
            traj["returns_to_go"] = discount_cumsum(traj["rewards"], 1.0)

    print(f"Loaded {len(trajectories)} trajectories")

    # Initialize encoder
    print("Loading Qwen encoder...")
    encoder = PromptTextEncoder(model_name="Qwen/Qwen2.5-0.5B-Instruct", device="cuda")

    # Generate prompts and encode
    task_texts = []
    hindsight_texts = []
    foresight_texts = []
    lengths = []

    for i, traj in enumerate(trajectories):
        states = traj['dem_observations']
        actions = traj.get('actions', None)

        sequences = build_prompt_sequences_v7(states, actions=actions)

        task_texts.extend(sequences["task_prompts"])
        hindsight_texts.extend(sequences["hindsight_prompts"])
        foresight_texts.extend(sequences["foresight_prompts"])
        lengths.append(len(sequences["task_prompts"]))

        if (i + 1) % 1000 == 0:
            print(f"  Generated prompts for {i+1}/{len(trajectories)} trajectories")

    print(f"Total prompts: task={len(task_texts)}, hindsight={len(hindsight_texts)}, foresight={len(foresight_texts)}")

    # Encode
    print(f"Encoding {len(task_texts)} task prompts...")
    task_embeddings = encoder.encode_texts(task_texts, batch_size=batch_size, output_dtype=output_dtype)

    print(f"Encoding {len(hindsight_texts)} hindsight prompts...")
    hindsight_embeddings = encoder.encode_texts(hindsight_texts, batch_size=batch_size, output_dtype=output_dtype)

    print(f"Encoding {len(foresight_texts)} foresight prompts...")
    foresight_embeddings = encoder.encode_texts(foresight_texts, batch_size=batch_size, output_dtype=output_dtype)

    # Assign back to trajectories
    offset = 0
    for i, traj in enumerate(trajectories):
        n = lengths[i]
        next_offset = offset + n

        traj["v7_task_prompts"] = task_texts[offset:next_offset]
        traj["v7_hindsight_prompts"] = hindsight_texts[offset:next_offset]
        traj["v7_foresight_prompts"] = foresight_texts[offset:next_offset]
        traj["v7_task_embeddings"] = task_embeddings[offset:next_offset]
        traj["v7_hindsight_embeddings"] = hindsight_embeddings[offset:next_offset]
        traj["v7_foresight_embeddings"] = foresight_embeddings[offset:next_offset]
        traj["v7_prompt_version"] = "v7_state_only"

        offset = next_offset

    # Save
    print(f"Saving to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)

    print(f"Done! Saved {len(trajectories)} trajectories with v7 embeddings")
    print(f"Sample v7 task prompt: {trajectories[0]['v7_task_prompts'][0][:200]}...")
    print(f"Task embedding shape: {trajectories[0]['v7_task_embeddings'].shape}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    generate_v7_embeddings(args.data, args.output, batch_size=args.batch_size)
