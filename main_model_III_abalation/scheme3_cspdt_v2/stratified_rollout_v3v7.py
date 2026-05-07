import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

"""
10步 Rollout 评估 — CSP-DT (SeMDT 6-token) with v3+v7 hybrid semantic embeddings
  Step 0: 预计算的 v3 嵌入 (从 pickle 数据文件)
  Step 1+: v7 在线生成嵌入 (仅基于 World Model 预测的 45 维 state)
  World model 提供: next_state + delta_saps2

模型接口与 train_stage1/stage2 一致 (build_policy / WorldModel)
"""
import sys
import pickle
import random
import torch
import numpy as np
import json
import argparse
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts.state_only_prompts_v7 import (
    build_task_v7, build_hindsight_v7_first, build_hindsight_v7, build_foresight_v7
)
from prompts.text_encoder import PromptTextEncoder

import config as C
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
    """Encode a small batch of prompts, return tensor."""
    emb = encoder.encode_texts(texts, batch_size=len(texts))
    return torch.FloatTensor(emb)


def _resolve_ckpt(checkpoint_path, device):
    """Load checkpoint — supports both single-file dict and separate policy/world_model."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    if 'policy_state_dict' in ckpt:
        return ckpt['policy_state_dict'], ckpt['world_model_state_dict']

    # Separate files: checkpoint_path is a directory like checkpoints/stage2/step_200000/
    if os.path.isdir(checkpoint_path):
        policy_sd = torch.load(os.path.join(checkpoint_path, 'policy.pt'), map_location=device)
        world_sd = torch.load(os.path.join(checkpoint_path, 'world_model.pt'), map_location=device)
        return policy_sd, world_sd

    raise ValueError(f"Cannot resolve checkpoint format: {checkpoint_path}")


def evaluate_cspdt_v3v7(checkpoint_path, data_path, output_path):
    print(f"Model type: {C.MODEL_TYPE}, block_size: {C.BLOCK_SIZE}")

    # Load encoder for v7 prompts (used from step 1 onwards)
    print("Loading Qwen encoder for v7 prompts...")
    encoder = PromptTextEncoder(model_name="Qwen/Qwen2.5-0.5B-Instruct", device="cuda")
    print("Encoder ready.")

    # Build models (same interface as train_stage1/stage2)
    policy = build_policy(
        vocab_size=C.VOCAB_SIZE, block_size=C.BLOCK_SIZE,
        n_layer=C.N_LAYER, n_head=C.N_HEAD, n_embd=C.N_EMBD,
        language_emb_dim=C.LANGUAGE_EMB_DIM, max_timestep=C.CONTEXT_LENGTH,
        model_type=C.MODEL_TYPE,
    ).to(device)

    world_model = WorldModel(
        state_dim=C.STATE_DIM, action_dim=C.VOCAB_SIZE,
        hidden_dim=C.O_HIDDEN, dropout=C.MC_DROPOUT,
    ).to(device)

    policy_sd, world_sd = _resolve_ckpt(checkpoint_path, device)
    policy.load_state_dict(policy_sd)
    policy.eval()
    world_model.load_state_dict(world_sd)

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
        rtgs_np = traj['returns_to_go']

        # Build incremental context for autoregressive rollout
        cur_states = states[0:1].unsqueeze(0)               # (1, 1, 45)
        cur_actions = None
        cur_rtgs = torch.FloatTensor(rtgs_np[0:1]).unsqueeze(-1).unsqueeze(0).to(device)  # (1,1,1)
        cur_timesteps = torch.LongTensor([[0]]).unsqueeze(-1).to(device)  # (1,1,1)

        prev_state_np = states[0].cpu().numpy()
        saps2_deltas = []

        for step in range(min(ROLL_STEPS, len(states) - 1)):
            # --- Get semantic embeddings ---
            if step == 0:
                # Step 0: precomputed v3 embeddings from pickle
                emb_key = 'v6_task_embeddings' if 'v6_task_embeddings' in traj else 'task_embeddings'
                h_key = 'v6_hindsight_embeddings' if 'v6_hindsight_embeddings' in traj else 'hindsight_embeddings'
                f_key = 'v6_foresight_embeddings' if 'v6_foresight_embeddings' in traj else 'foresight_embeddings'
                task_emb = torch.FloatTensor(np.asarray(traj[emb_key][0:1])).unsqueeze(0).to(device)   # (1,1,896)
                h_emb = torch.FloatTensor(np.asarray(traj[h_key][0:1])).unsqueeze(0).to(device)
                f_emb = torch.FloatTensor(np.asarray(traj[f_key][0:1])).unsqueeze(0).to(device)
            else:
                # Step 1+: generate v7 embeddings from predicted state
                curr_state_np = cur_states[0, -1, :].cpu().detach().numpy()

                task_text = build_task_v7(curr_state_np)
                if step == 1:
                    hindsight_text = build_hindsight_v7_first(curr_state_np)
                else:
                    action_id = actions_discrete[step - 1] if step - 1 < len(actions_discrete) else 15
                    hindsight_text = build_hindsight_v7(prev_state_np, curr_state_np, action_id)
                foresight_text = build_foresight_v7(curr_state_np)

                texts = [task_text, hindsight_text, foresight_text]
                embs = encode_single_prompt(encoder, texts)  # (3, 896)
                task_emb = embs[0:1].unsqueeze(0).to(device)   # (1,1,896)
                h_emb = embs[1:2].unsqueeze(0).to(device)
                f_emb = embs[2:3].unsqueeze(0).to(device)

                prev_state_np = curr_state_np

            # --- Policy: get action logits ---
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

            # --- World model: predict next state + delta_saps2 ---
            s_t = cur_states[0, -1:, :]  # (1, 45) instead of (45,)
            pred_a_t = torch.tensor([pred_action], device=device)
            sem_for_wm = torch.cat([task_emb[0], h_emb[0], f_emb[0]], dim=-1)  # (1, 2688)
            with torch.no_grad():
                mu, sigma, delta_saps2 = world_model.predict(s_t, pred_a_t, semantic=sem_for_wm, mc_samples=C.MC_SAMPLES)
                saps2_deltas.append(delta_saps2.item())

            # --- Extend context window ---
            next_state = mu.unsqueeze(0)  # mu is (1, 45), unsqueeze to (1, 1, 45)
            cur_states = torch.cat([cur_states, next_state], dim=1)[:, -C.CONTEXT_LENGTH:, :]

            next_action = torch.LongTensor([[pred_action]]).unsqueeze(-1).to(device)  # (1,1,1)
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

    # --- Report ---
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

    print(f"\nCSP-DT ({C.MODEL_TYPE}) v3+v7 10步Rollout结果:")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint dir (e.g. checkpoints/stage2/step_200000/) or single file')
    parser.add_argument('--data', type=str, required=True,
                        help='Test pickle file (e.g. data/v3/test_Phys45_v3.pickle)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON path')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    evaluate_cspdt_v3v7(args.checkpoint, args.data, args.output)
