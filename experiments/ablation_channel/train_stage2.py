"""
Stage 2 (Channel Ablation): Counterfactual Self-Play with masked semantic channels.
  - Policy: SeMDT with channel-masked semantic
  - WorldModel: WITH semantic but masked channels zeroed
  - sigma=2.0, search_radius=±2

Usage:
  python train_stage2.py --variant A1_no_task
  python train_stage2.py --variant A4_only_task
"""
import argparse
import os
import sys
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, '..', '..', 'main_model', 'scheme3_cspdt_v2')
sys.path.insert(0, script_dir)
sys.path.insert(0, model_dir)

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from channel_mask import mask_embeddings, mask_semantic_concat, get_active_channels, variant_tag
from models.policy import build_policy
from models.world_model import WorldModel
from datasets.v3_semantic_dataset import V3SemanticDataset
from prompts.semantic_generator import SemanticGenerator
from prompts.text_encoder import PromptTextEncoder


def search_better_action(policy, world_model, state, current_action_idx,
                         task_emb, h_emb, f_emb, variant,
                         search_radius=2, mc_samples=5, sigma_threshold=2.0):
    device = state.device

    # Mask channels for world model input
    sem_concat = torch.cat([task_emb, h_emb, f_emb], dim=-1)  # (1, 2688)
    sem_concat = mask_semantic_concat(sem_concat, variant)

    # Doctor action's predicted delta
    with torch.no_grad():
        doc_action_t = torch.tensor([current_action_idx], device=device)
        doc_mu, doc_sigma, doctor_delta = world_model.predict(
            state, doc_action_t, semantic=sem_concat, mc_samples=mc_samples)
        doctor_delta = doctor_delta.item()

    # Generate discrete action candidates
    candidates = set()
    for offset in range(-search_radius, search_radius + 1):
        idx = current_action_idx + offset
        if 0 <= idx < 25:
            candidates.add(idx)

    # Policy's suggested action (with masked semantic)
    task_m, h_m, f_m = mask_embeddings(task_emb, h_emb, f_emb, variant)
    with torch.no_grad():
        policy_logits = policy.get_action_logits(
            state.unsqueeze(0),
            None,
            torch.zeros(1, 1, 1, device=device),
            torch.zeros(1, 1, 1, dtype=torch.long, device=device),
            task_m.unsqueeze(0),
            h_m.unsqueeze(0),
            f_m.unsqueeze(0),
        )
        policy_action_idx = policy_logits.argmax(dim=-1).item()
        candidates.add(policy_action_idx)

    best_delta = doctor_delta
    best_action_idx = current_action_idx
    best_mu_pred = doc_mu
    best_sigma_pred = doc_sigma
    best_sigma_mean = doc_sigma.mean().item()

    for action_idx in candidates:
        action_t = torch.tensor([action_idx], device=device)
        with torch.no_grad():
            mu_cand, sigma_cand, candidate_delta = world_model.predict(
                state, action_t, semantic=sem_concat, mc_samples=mc_samples)
        candidate_delta_val = candidate_delta.item()
        sigma_mean = sigma_cand.mean().item()

        if sigma_mean >= sigma_threshold:
            continue

        if candidate_delta_val < best_delta:
            best_delta = candidate_delta_val
            best_action_idx = action_idx
            best_mu_pred = mu_cand
            best_sigma_pred = sigma_cand
            best_sigma_mean = sigma_mean

    advantage = doctor_delta - best_delta
    confidence = max(0.0, min(1.0, 1.0 - best_sigma_mean / sigma_threshold))

    search_info = {
        'confidence': confidence,
        'mu_pred': best_mu_pred,
        'sigma_pred': best_sigma_pred,
        'sigma_mean': best_sigma_mean,
    }
    return best_action_idx, advantage, search_info


def train_stage2(args):
    config_name = f'config_{args.variant}'
    C = __import__(config_name)
    variant = C.VARIANT

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logdir = args.logdir or os.path.join(C.CHECKPOINT_DIR, 'stage2')
    os.makedirs(logdir, exist_ok=True)

    active = get_active_channels(variant)
    tag = variant_tag(variant)
    print(f"{tag} Stage 2 Channel Ablation Training")
    print(f"{tag} Active channels: {active}")

    print(f"{tag} Loading v3 semantic datasets...")
    train_dataset = V3SemanticDataset(
        os.path.join(C.DATA_DIR, 'train_Phys45_v3.pickle'),
        C.CONTEXT_LENGTH, C.RTG_SCALE, gamma=C.RTG_GAMMA,
        language_emb_dim=C.LANGUAGE_EMB_DIM,
    )
    trajectories = train_dataset.trajectories
    print(f"{tag} Loaded {len(trajectories)} trajectories")

    # Stratified sampling
    low_indices, mid_indices, high_indices = [], [], []
    for i, traj in enumerate(trajectories):
        init_saps2 = float(traj['acuities_original_init'][2])
        if init_saps2 < 75:
            low_indices.append(i)
        elif init_saps2 < 85:
            mid_indices.append(i)
        else:
            high_indices.append(i)

    print(f"{tag} Stratified sampling:")
    print(f"  Low (<75):   {len(low_indices):5d}")
    print(f"  Mid (75-85): {len(mid_indices):5d}")
    print(f"  High (≥85):  {len(high_indices):5d}")

    stratified_indices = low_indices + mid_indices + high_indices
    n_traj = len(stratified_indices)

    # Build models
    lang_dim = train_dataset.language_emb_dim or C.LANGUAGE_EMB_DIM
    use_atg = 'ATG' in C.MODEL_TYPE
    policy = build_policy(
        vocab_size=C.VOCAB_SIZE, block_size=C.BLOCK_SIZE,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        language_emb_dim=lang_dim, max_timestep=C.CONTEXT_LENGTH,
        model_type=C.MODEL_TYPE,
    ).to(device)

    world_model = WorldModel(
        state_dim=C.STATE_DIM, action_dim=C.VOCAB_SIZE,
        hidden_dim=C.O_HIDDEN, dropout=C.MC_DROPOUT,
        use_semantic=True,
    ).to(device)

    policy.load_state_dict(torch.load(args.policy_ckpt, map_location=device))
    world_model.load_state_dict(torch.load(args.world_model_ckpt, map_location=device))
    print(f"{tag} Loaded stage1 checkpoints from:\n  {args.policy_ckpt}\n  {args.world_model_ckpt}")

    # Load normalization params
    norm_path = os.path.join(C.DATA_DIR, 'train_Phys45_v3_norm.pkl')
    if not os.path.exists(norm_path):
        norm_path = os.path.join(C.DATA_DIR, 'normalization_params.pkl')

    if os.path.exists(norm_path):
        with open(norm_path, 'rb') as f:
            norm_params = pickle.load(f)
        state_mean = norm_params['state_mean']
        state_std = norm_params['state_std'] + 1e-8
        print(f"{tag} Loaded normalization params from {norm_path}")
    else:
        print(f"{tag} Warning: normalization params not found")
        state_mean = np.zeros(C.STATE_DIM)
        state_std = np.ones(C.STATE_DIM)

    # Semantic generator for better_action embeddings
    text_encoder = PromptTextEncoder(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device=device,
        max_length=256,
    )
    semantic_generator = SemanticGenerator(text_encoder, device=device)
    print(f"{tag} Initialized semantic generator")

    opt_pi = optim.Adam(policy.parameters(), lr=args.lr_pi)
    opt_O = optim.Adam(world_model.parameters(), lr=args.lr_O * 0.5)
    lambda_O = args.lambda_O
    search_radius = args.search_radius

    epochs = args.epochs
    selfplay_iterations = args.selfplay_iterations

    print(f"{tag} Starting: {epochs} epochs x {selfplay_iterations} iterations, "
          f"search_radius=±{search_radius}")

    best_advantage = 0.0
    global_step = 0

    for epoch in range(epochs):
        policy.train()
        world_model.train()

        total_advantage = 0.0
        total_improvements = 0
        total_policy_loss = 0.0
        total_world_loss = 0.0
        total_confidence = 0.0
        total_alpha = 0.0
        total_bootstrap_count = 0

        for iteration in range(selfplay_iterations):
            sample_idx = np.random.randint(n_traj)
            traj_idx = stratified_indices[sample_idx]
            traj = trajectories[traj_idx]

            states_np = traj['dem_observations']
            actions_np = traj['actions']
            T = len(states_np)

            if T < 2:
                continue

            t = np.random.randint(0, T - 1)

            state_t = torch.FloatTensor(states_np[t:t+1]).to(device)
            next_state_t = torch.FloatTensor(states_np[t+1:t+2]).to(device)
            doctor_action_idx = int(actions_np[t])

            task_emb = torch.FloatTensor(
                np.asarray(traj['task_embeddings'][t:t+1])).to(device)
            doctor_h_emb = torch.FloatTensor(
                np.asarray(traj['hindsight_embeddings'][t:t+1])).to(device)
            doctor_f_emb = torch.FloatTensor(
                np.asarray(traj['foresight_embeddings'][t:t+1])).to(device)

            # Search with masked channels
            better_action_idx, advantage, search_info = search_better_action(
                policy, world_model, state_t, doctor_action_idx,
                task_emb, doctor_h_emb, doctor_f_emb, variant,
                search_radius=search_radius,
                mc_samples=C.MC_SAMPLES,
                sigma_threshold=C.SIGMA_THRESHOLD_SEARCH,
            )

            if advantage > 0:
                if t == 0:
                    continue

                # Generate better_action's semantic embeddings
                prev_state_normalized = states_np[t-1]
                curr_state_normalized = states_np[t]
                prev_state_raw = prev_state_normalized * state_std + state_mean
                curr_state_raw = curr_state_normalized * state_std + state_mean
                next_acuities_np = traj['acuities'][t]

                better_h_emb, better_f_emb = semantic_generator.generate_for_action(
                    prev_state_raw, curr_state_raw, next_acuities_np, better_action_idx
                )

                # Build masked semantic concat for world model
                sem_concat = torch.cat([task_emb, better_h_emb, better_f_emb], dim=-1)
                sem_concat = mask_semantic_concat(sem_concat, variant)

                # Mask policy embeddings
                task_m, h_m, f_m = mask_embeddings(task_emb, better_h_emb, better_f_emb, variant)

                confidence = search_info['confidence']
                base_alpha = max(C.ALPHA_MIN, 1.0 - global_step / C.ALPHA_DECAY_STEPS)
                if confidence >= C.CONFIDENCE_GATE:
                    alpha = base_alpha * confidence
                else:
                    alpha = 0.0

                opt_pi.zero_grad()
                opt_O.zero_grad()

                # Get world model predicted delta for ATG
                with torch.no_grad():
                    better_a_t = torch.tensor([better_action_idx], device=device)
                    _, _, pred_delta = world_model.predict(
                        state_t, better_a_t, semantic=sem_concat, mc_samples=C.MC_SAMPLES)

                # Policy loss
                rtg_t = torch.FloatTensor(
                    traj['returns_to_go'][t:t+1]).unsqueeze(-1).unsqueeze(0).to(device)
                ts_t = torch.LongTensor([t]).unsqueeze(-1).unsqueeze(0).to(device)
                action_input = torch.LongTensor([[doctor_action_idx]]).unsqueeze(-1).to(device)

                forward_kwargs = dict(
                    states=state_t.unsqueeze(0),
                    actions=action_input,
                    targets=None,
                    rtgs=rtg_t,
                    timesteps=ts_t,
                    task_embeddings=task_m.unsqueeze(0),
                    hindsight_embeddings=h_m.unsqueeze(0),
                    foresight_embeddings=f_m.unsqueeze(0),
                )
                if use_atg:
                    forward_kwargs['delta_saps2'] = pred_delta.unsqueeze(0)

                logits, _, _ = policy(**forward_kwargs)
                policy_pred = logits[:, -1, :]

                target_onehot = torch.zeros(1, C.VOCAB_SIZE, device=device)
                target_onehot[0, better_action_idx] = 1.0
                policy_loss = ((policy_pred - target_onehot.detach()) ** 2).mean()

                # World model loss
                a_onehot = torch.zeros(1, C.VOCAB_SIZE, device=device)
                a_onehot[0, better_action_idx] = 1.0
                mu_pred, log_sigma, _ = world_model(state_t, a_onehot, semantic=sem_concat)
                real_loss = ((mu_pred - next_state_t) ** 2).mean()

                if confidence >= C.CONFIDENCE_GATE:
                    mu_target = search_info['mu_pred'].detach()
                    bootstrap_loss = ((mu_pred - mu_target) ** 2).mean()
                    world_loss = alpha * bootstrap_loss + (1 - alpha) * real_loss
                    total_bootstrap_count += 1
                else:
                    world_loss = real_loss

                total_loss = policy_loss + lambda_O * world_loss
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)

                opt_pi.step()
                opt_O.step()

                total_policy_loss += policy_loss.item()
                total_world_loss += world_loss.item()
                total_improvements += 1
                total_confidence += confidence
                total_alpha += alpha

            total_advantage += advantage
            global_step += 1

        avg_advantage = total_advantage / selfplay_iterations
        improvement_rate = total_improvements / selfplay_iterations
        avg_policy_loss = total_policy_loss / max(total_improvements, 1)
        avg_world_loss = total_world_loss / max(total_improvements, 1)
        avg_confidence = total_confidence / max(total_improvements, 1)
        avg_alpha = total_alpha / max(total_improvements, 1)
        bootstrap_pct = total_bootstrap_count / max(total_improvements, 1)

        print(f"{tag} Epoch {epoch+1}/{epochs} (global_step={global_step}): "
              f"Avg Advantage={avg_advantage:.4f}, "
              f"Improvement Rate={improvement_rate:.2%}, "
              f"Policy Loss={avg_policy_loss:.4f}, "
              f"World Loss={avg_world_loss:.4f}, "
              f"Confidence={avg_confidence:.3f}, "
              f"Alpha={avg_alpha:.3f}, "
              f"Bootstrap%={bootstrap_pct:.2%}")

        is_best = avg_advantage > best_advantage
        if is_best:
            best_advantage = avg_advantage

        if is_best:
            best_path = os.path.join(logdir, 'best_checkpoint.pt')
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'policy_state_dict': policy.state_dict(),
                'world_model_state_dict': world_model.state_dict(),
                'advantage': avg_advantage,
            }, best_path)
            print(f"  -> Saved best checkpoint (advantage={best_advantage:.4f})")

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(logdir, f'epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'policy_state_dict': policy.state_dict(),
                'world_model_state_dict': world_model.state_dict(),
                'advantage': avg_advantage,
            }, ckpt_path)

    print(f"\n{tag} Training complete. Best advantage: {best_advantage:.4f} "
          f"(global_steps={global_step})")
    print(f"{tag} Checkpoints in {logdir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True,
                        choices=['A1_no_task', 'A2_no_hindsight', 'A3_no_foresight',
                                 'A4_only_task', 'A5_only_hindsight', 'A6_only_foresight'])
    parser.add_argument('--policy_ckpt', type=str, required=True)
    parser.add_argument('--world_model_ckpt', type=str, required=True)
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--selfplay_iterations', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--search_radius', type=int, default=2)
    parser.add_argument('--lr_pi', type=float, default=6e-4)
    parser.add_argument('--lr_O', type=float, default=1e-3)
    parser.add_argument('--lambda_O', type=float, default=0.5)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    train_stage2(args)
