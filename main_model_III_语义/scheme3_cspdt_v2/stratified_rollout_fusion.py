"""
10-step Rollout Evaluation — Fusion Ablation variant.
  - Policy: SeMDT_Concat/Residual/Gated/CrossAttn (3-token)
  - WorldModel: NO semantic (use_semantic=False)
  - sigma=2.0

Usage:
  python stratified_rollout_fusion.py --fusion concat \
    --checkpoint checkpoints_fusion/concat/stage2/best_checkpoint.pt \
    --data data/v3/test_Phys45_v3.pickle \
    --output results_fusion/concat/rollout.json
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import pickle
import random
import torch
import numpy as np
import json
import argparse
from pathlib import Path

# Add paths: models/datasets from main_model_III, config from current dir
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, '..', '..', 'main_model_III', 'scheme3_cspdt_v2')
sys.path.insert(0, script_dir)
sys.path.insert(0, model_dir)

from prompts.state_only_prompts_v7 import (
    build_task_v7, build_hindsight_v7_first, build_hindsight_v7, build_foresight_v7
)
from prompts.text_encoder import PromptTextEncoder

import config_fusion as C
from models.policy import build_policy
from models.world_model import WorldModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROLL_STEPS = 10
MIN_TRAJ_LENGTH = 11


def stratify(initial_saps2):
    if initial_saps2 < 75: return 'low'
    elif initial_saps2 < 85: return 'mid'
    else: return 'high'


def encode_single_prompt(encoder, texts):
    emb = encoder.encode_texts(texts, batch_size=len(texts))
    return torch.FloatTensor(emb)


def _resolve_ckpt(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if 'policy_state_dict' in ckpt:
        return ckpt['policy_state_dict'], ckpt['world_model_state_dict']

    if os.path.isdir(checkpoint_path):
        policy_sd = torch.load(os.path.join(checkpoint_path, 'policy.pt'), map_location=device)
        world_sd = torch.load(os.path.join(checkpoint_path, 'world_model.pt'), map_location=device)
        return policy_sd, world_sd

    raise ValueError(f"Cannot resolve checkpoint format: {checkpoint_path}")


def evaluate_fusion_ablation(checkpoint_path, data_path, output_path, variant_name):
    # Configure fusion variant
    model_type, block_size, tokens = C.configure_variant(variant_name)
    tag = f"[{variant_name}]"

    print(f"{tag} Model type: {model_type}, tokens: {tokens}, block_size: {block_size}")

    # Load Qwen encoder for v7 prompts
    print(f"{tag} Loading Qwen encoder for v7 prompts...")
    encoder = PromptTextEncoder(model_name="Qwen/Qwen2.5-0.5B-Instruct", device="cuda")
    print(f"{tag} Encoder ready.")

    policy = build_policy(
        vocab_size=C.VOCAB_SIZE, block_size=block_size,
        n_layer=C.N_LAYER, n_head=C.N_HEAD, n_embd=C.N_EMBD,
        language_emb_dim=C.LANGUAGE_EMB_DIM, max_timestep=C.CONTEXT_LENGTH,
        model_type=model_type,
    ).to(device)

    world_model = WorldModel(
        state_dim=C.STATE_DIM, action_dim=C.VOCAB_SIZE,
        hidden_dim=C.O_HIDDEN, dropout=C.MC_DROPOUT,
        use_semantic=False,
    ).to(device)

    policy_sd, world_sd = _resolve_ckpt(checkpoint_path, device)
    policy.load_state_dict(policy_sd)
    policy.eval()
    world_model.load_state_dict(world_sd)
    world_model.eval()

    with open(data_path, 'rb') as f:
        trajectories = pickle.load(f)

    trajectories = [t for t in trajectories if len(t['dem_observations']) >= MIN_TRAJ_LENGTH]
    print(f"{tag} Test trajectories: {len(trajectories)}")

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
        rtgs_np = traj['returns_to_go']

        cur_states = states[0:1].unsqueeze(0)
        cur_actions = None
        cur_rtgs = torch.FloatTensor(rtgs_np[0:1]).unsqueeze(-1).unsqueeze(0).to(device)
        cur_timesteps = torch.LongTensor([[0]]).unsqueeze(-1).to(device)

        prev_state_np = states[0].cpu().numpy()
        saps2_deltas = []

        for step in range(min(ROLL_STEPS, len(states) - 1)):
            # Build semantic embeddings
            if step == 0:
                emb_key = 'v6_task_embeddings' if 'v6_task_embeddings' in traj else 'task_embeddings'
                h_key = 'v6_hindsight_embeddings' if 'v6_hindsight_embeddings' in traj else 'hindsight_embeddings'
                f_key = 'v6_foresight_embeddings' if 'v6_foresight_embeddings' in traj else 'foresight_embeddings'
                task_emb = torch.FloatTensor(np.asarray(traj[emb_key][0:1])).unsqueeze(0).to(device)
                h_emb = torch.FloatTensor(np.asarray(traj[h_key][0:1])).unsqueeze(0).to(device)
                f_emb = torch.FloatTensor(np.asarray(traj[f_key][0:1])).unsqueeze(0).to(device)
            else:
                curr_state_np = cur_states[0, -1, :].cpu().detach().numpy()

                task_text = build_task_v7(curr_state_np)
                if step == 1:
                    hindsight_text = build_hindsight_v7_first(curr_state_np)
                else:
                    action_id = actions_discrete[step - 1] if step - 1 < len(actions_discrete) else 15
                    hindsight_text = build_hindsight_v7(prev_state_np, curr_state_np, action_id)
                foresight_text = build_foresight_v7(curr_state_np)

                texts = [task_text, hindsight_text, foresight_text]
                embs = encode_single_prompt(encoder, texts)
                task_emb = embs[0:1].unsqueeze(0).to(device)
                h_emb = embs[1:2].unsqueeze(0).to(device)
                f_emb = embs[2:3].unsqueeze(0).to(device)

                prev_state_np = curr_state_np

            # Policy: get action logits (fused semantic, no channel masking)
            with torch.no_grad():
                logits, _, _ = policy(
                    states=cur_states,
                    actions=cur_actions,
                    targets=None,
                    rtgs=cur_rtgs,
                    timesteps=cur_timesteps,
                    task_embeddings=task_emb,
                    hindsight_embeddings=h_emb,
                    foresight_embeddings=f_emb,
                )
                pred_action = logits[:, -1, :].argmax(dim=-1).item()

            # World model: predict (no semantic)
            s_t = cur_states[0, -1:, :]
            pred_a_t = torch.tensor([pred_action], device=device)

            with torch.no_grad():
                mu, sigma, delta_saps2 = world_model.predict(
                    s_t, pred_a_t, mc_samples=C.MC_SAMPLES)
                saps2_deltas.append(delta_saps2.item())

            # Extend context window
            next_state = mu.unsqueeze(0)
            cur_states = torch.cat([cur_states, next_state], dim=1)[:, -C.CONTEXT_LENGTH:, :]

            next_action = torch.LongTensor([[pred_action]]).unsqueeze(-1).to(device)
            if cur_actions is None:
                cur_actions = next_action
            else:
                cur_actions = torch.cat([cur_actions, next_action], dim=1)[:, -C.CONTEXT_LENGTH:, :]

            next_rtg = cur_rtgs[:, -1:, :]
            cur_rtgs = torch.cat([cur_rtgs, next_rtg], dim=1)[:, -C.CONTEXT_LENGTH:, :]

            next_ts = (cur_timesteps[:, -1:, :] + 1).clamp(max=C.CONTEXT_LENGTH)
            cur_timesteps = torch.cat([cur_timesteps, next_ts], dim=1)[:, -C.CONTEXT_LENGTH:, :]

        total_delta = sum(saps2_deltas)
        results[stratum].append(total_delta)

        if (idx + 1) % 200 == 0:
            print(f"  Evaluated {idx+1}/{len(trajectories)} trajectories...")

    # Report
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

    print(f"\n{tag} Fusion Ablation ({variant_name}) 10-step Rollout Results:")
    for group in ['high', 'mid', 'low']:
        if group in stats:
            print(f"  {group.upper()}: {stats[group]['mean']:.4f} +/- {stats[group]['std']:.4f}, n={stats[group]['count']}")

    all_vals = [v for vs in results.values() for v in vs]
    if all_vals:
        stats['overall'] = {
            'mean': float(np.mean(all_vals)),
            'std': float(np.std(all_vals)),
            'median': float(np.median(all_vals)),
            'count': len(all_vals)
        }
        print(f"  ALL:   {stats['overall']['mean']:.4f} +/- {stats['overall']['std']:.4f}, n={stats['overall']['count']}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"{tag} Results saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion', type=str, required=True,
                        choices=['concat', 'residual', 'gated', 'cross_attn'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    evaluate_fusion_ablation(args.checkpoint, args.data, args.output, args.fusion)
