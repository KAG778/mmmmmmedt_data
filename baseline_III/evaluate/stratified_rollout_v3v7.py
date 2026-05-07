import os
import random
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

"""
10-step Rollout Evaluation for Baseline models (BC, BCQ, DT, IQL)
with v3+v7 hybrid semantic embeddings.
  Step 0: precomputed v3 embeddings from pickle data
  Step 1+: v7 online embeddings from predicted state
  Reads model config from YAML to instantiate the correct model type.
"""
import sys
import pickle
import torch
import numpy as np
import json
import argparse
import yaml
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))
from models.policies import get_policy
from models.world_models import get_world_model
# prompts is at ../../prompts relative to baseline/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from prompts.state_only_prompts_v7 import (
    build_task_v7, build_hindsight_v7_first, build_hindsight_v7, build_foresight_v7
)
from prompts.text_encoder import PromptTextEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROLL_STEPS = 10
MIN_TRAJ_LENGTH = 11


def stratify(initial_saps2):
    if initial_saps2 < 75: return 'low'
    elif initial_saps2 < 85: return 'mid'
    else: return 'high'


def encode_single_prompt(encoder, texts):
    """Encode a small batch of prompts, return tensor."""
    emb = encoder.encode_texts(texts, batch_size=len(texts))
    return torch.FloatTensor(emb)


def evaluate_baseline_v3v7(config_path, checkpoint_path, data_path, output_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model_cfg = config['model']
    policy_cfg = model_cfg['policy']
    world_cfg = model_cfg['world_model']
    model_name = model_cfg.get('name', policy_cfg['type'])
    print(f"Model: {model_name}, Policy type: {policy_cfg['type']}, World type: {world_cfg['type']}")

    # Load encoder for v7 prompts (used from step 1 onwards)
    print("Loading Qwen encoder for v7 prompts...")
    encoder = PromptTextEncoder(model_name="Qwen/Qwen2.5-0.5B-Instruct", device=str(device))
    print("Encoder ready.")

    # Build models from config
    policy = get_policy(policy_cfg).to(device)
    world_model = get_world_model(world_cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    world_model.load_state_dict(checkpoint['world_model_state_dict'])
    world_model.eval()

    # Load test data
    with open(data_path, 'rb') as f:
        trajectories = pickle.load(f)

    trajectories = [t for t in trajectories if len(t['dem_observations']) >= MIN_TRAJ_LENGTH]
    print(f"测试轨迹数: {len(trajectories)}")

    results = {'high': [], 'mid': [], 'low': []}

    for idx, traj in enumerate(trajectories):
        acuities = traj['acuities']
        if isinstance(acuities, torch.Tensor):
            acuities = acuities.cpu().numpy()
        if acuities.ndim == 2:
            acuities = acuities[:, 2]

        init_saps2 = float(acuities[0])
        stratum = stratify(init_saps2)

        states = torch.FloatTensor(traj['dem_observations']).to(device)
        actions_discrete = traj['actions']

        current_state = states[0:1]
        prev_state_np = states[0].cpu().numpy()
        saps2_deltas = []

        for step in range(min(ROLL_STEPS, len(states) - 1)):
            if step == 0:
                # Step 0: use precomputed v3 embeddings
                task_emb = torch.FloatTensor(traj['v6_task_embeddings']).to(device)
                h_emb = torch.FloatTensor(traj['v6_hindsight_embeddings']).to(device)
                f_emb = torch.FloatTensor(traj['v6_foresight_embeddings']).to(device)
                sem_tensor = torch.cat([task_emb[0:1], h_emb[0:1], f_emb[0:1]], dim=-1)
            else:
                # Step 1+: generate v7 embeddings from predicted state
                curr_state_np = current_state[0].cpu().detach().numpy()

                task_text = build_task_v7(curr_state_np)
                if step == 1:
                    hindsight_text = build_hindsight_v7_first(curr_state_np)
                else:
                    action_id = actions_discrete[step - 1] if step - 1 < len(actions_discrete) else 15
                    hindsight_text = build_hindsight_v7(prev_state_np, curr_state_np, action_id)
                foresight_text = build_foresight_v7(curr_state_np)

                # Encode with Qwen
                texts = [task_text, hindsight_text, foresight_text]
                embs = encode_single_prompt(encoder, texts)  # (3, 896)
                sem_tensor = torch.cat([embs[0:1], embs[1:2], embs[2:3]], dim=-1).to(device)  # (1, 2688)

                prev_state_np = curr_state_np

            sem_ctx = {'semantic': sem_tensor}

            with torch.no_grad():
                action_logits = policy.get_action(current_state, context=sem_ctx)
                pred_action = torch.argmax(action_logits, dim=-1).item()

                action_onehot = torch.zeros(1, 25).to(device)
                action_onehot[0, pred_action] = 1.0

                delta = world_model.predict_saps2_delta(
                    current_state, action_onehot,
                    semantic_context=sem_tensor
                ).item()
                saps2_deltas.append(delta)

                world_out = world_model(current_state, action_onehot, semantic_context=sem_tensor)
                current_state = world_out['mu']

        total_delta = sum(saps2_deltas)
        results[stratum].append(total_delta)

        if (idx + 1) % 200 == 0:
            print(f"  Evaluated {idx+1}/{len(trajectories)} trajectories...")

    stats = {}
    for stratum in ['high', 'mid', 'low']:
        values = results[stratum]
        if values:
            stats[stratum] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'count': len(values)
            }

    print(f"\n{model_name} ({policy_cfg['type']}) v3+v7 10步Rollout结果:")
    for group in ['high', 'mid', 'low']:
        if group in stats:
            print(f"  {group.upper()}: {stats[group]['mean']:.4f} ± {stats[group]['std']:.4f}, n={stats[group]['count']}")

    all_vals = [v for vs in results.values() for v in vs]
    if all_vals:
        stats['overall'] = {
            'mean': float(np.mean(all_vals)),
            'std': float(np.std(all_vals)),
            'median': float(np.median(all_vals)),
            'count': len(all_vals)
        }
        print(f"  ALL:   {stats['overall']['mean']:.4f} ± {stats['overall']['std']:.4f}, n={stats['overall']['count']}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"结果已保存到: {output_path}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Baseline config YAML (e.g. configs/bc.yaml)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint file (e.g. results/bc/stage1/best_checkpoint.pt)')
    parser.add_argument('--data', type=str, required=True,
                        help='Test pickle file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Seed: {args.seed}")
    evaluate_baseline_v3v7(args.config, args.checkpoint, args.data, args.output)
