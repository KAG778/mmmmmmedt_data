"""
Stage 2: Counterfactual Self-Play with Confidence-Gated Semi-Bootstrap.
ABLATION: BOTH Policy (DT) and WM WITHOUT semantic embeddings.

Key differences from full SeMDT:
  - No SemanticGenerator / PromptTextEncoder (saves ~1GB GPU memory)
  - Policy (DT) ignores semantic, uses precomputed embeddings for tensor compat
  - WM uses semantic=None everywhere
  - No online embedding generation needed
"""
import argparse
import os
import sys

# Force offline mode before any HuggingFace imports
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config_no_sem_full as C
from models.policy import build_policy
from models.world_model import WorldModel
from datasets.v3_semantic_dataset import V3SemanticDataset


def search_better_action(policy, world_model, state, current_action_idx,
                         task_emb, h_emb, f_emb,
                         search_radius=2, mc_samples=5, sigma_threshold=2.0):
    """
    Search for better action. WM uses semantic=None. Policy (DT) ignores semantic.
    """
    device = state.device

    # Doctor action's predicted delta (WM no semantic)
    with torch.no_grad():
        doc_action_t = torch.tensor([current_action_idx], device=device)
        doc_mu, doc_sigma, doctor_delta = world_model.predict(
            state, doc_action_t, semantic=None, mc_samples=mc_samples)
        doctor_delta = doctor_delta.item()

    # Generate discrete action candidates
    candidates = set()

    # 1. Neighborhood actions (±radius)
    for offset in range(-search_radius, search_radius + 1):
        idx = current_action_idx + offset
        if 0 <= idx < C.VOCAB_SIZE:
            candidates.add(idx)

    # 2. Policy's suggested action
    with torch.no_grad():
        policy_logits = policy.get_action_logits(
            state.unsqueeze(0),
            None,
            torch.zeros(1, 1, 1, device=device),
            torch.zeros(1, 1, 1, dtype=torch.long, device=device),
            task_emb.unsqueeze(0),
            h_emb.unsqueeze(0),
            f_emb.unsqueeze(0),
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
                state, action_t, semantic=None, mc_samples=mc_samples)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    os.makedirs(args.logdir, exist_ok=True)

    print(f"[Stage2-NoSemFull] Loading v3 semantic datasets...")
    train_dataset = V3SemanticDataset(
        os.path.join(args.datadir, 'train_Phys45_v3.pickle'),
        C.CONTEXT_LENGTH, C.RTG_SCALE, gamma=C.RTG_GAMMA,
        language_emb_dim=C.LANGUAGE_EMB_DIM,
    )
    trajectories = train_dataset.trajectories
    print(f"[Stage2-NoSemFull] Loaded {len(trajectories)} trajectories")

    # Stratified sampling by initial SAPS2
    low_indices, mid_indices, high_indices = [], [], []
    for i, traj in enumerate(trajectories):
        init_saps2 = float(traj['acuities_original_init'][2])
        if init_saps2 < 75:
            low_indices.append(i)
        elif init_saps2 < 85:
            mid_indices.append(i)
        else:
            high_indices.append(i)

    print(f"[Stage2-NoSemFull] Stratified sampling:")
    print(f"  Low (<75):   {len(low_indices):5d}")
    print(f"  Mid (75-85): {len(mid_indices):5d}")
    print(f"  High (≥85):  {len(high_indices):5d}")

    stratified_indices = low_indices + mid_indices + high_indices
    n_traj = len(stratified_indices)

    # Build models
    lang_dim = train_dataset.language_emb_dim or C.LANGUAGE_EMB_DIM
    policy = build_policy(
        vocab_size=C.VOCAB_SIZE, block_size=C.BLOCK_SIZE,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        language_emb_dim=lang_dim, max_timestep=C.CONTEXT_LENGTH,
        model_type=C.MODEL_TYPE,
    ).to(device)

    # World Model WITHOUT semantic
    world_model = WorldModel(
        state_dim=C.STATE_DIM, action_dim=C.VOCAB_SIZE,
        hidden_dim=C.O_HIDDEN, dropout=C.MC_DROPOUT,
        use_semantic=False,
    ).to(device)

    policy.load_state_dict(torch.load(args.policy_ckpt, map_location=device))
    world_model.load_state_dict(torch.load(args.world_model_ckpt, map_location=device))
    print(f"[Stage2-NoSemFull] Loaded stage1 checkpoints from:\n  {args.policy_ckpt}\n  {args.world_model_ckpt}")
    print(f"[Stage2-NoSemFull] No SemanticGenerator/TextEncoder — full ablation mode")

    opt_pi = optim.Adam(policy.parameters(), lr=args.lr_pi)
    opt_O = optim.Adam(world_model.parameters(), lr=args.lr_O * 0.5)
    lambda_O = args.lambda_O
    search_radius = args.search_radius

    epochs = args.epochs
    selfplay_iterations = args.selfplay_iterations

    print(f"[Stage2-NoSemFull] Starting: {epochs} epochs x {selfplay_iterations} iterations, "
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

            # Use precomputed embeddings from dataset (DT ignores, but needs valid tensors)
            task_emb = torch.FloatTensor(
                np.asarray(traj['task_embeddings'][t:t+1])).to(device)
            doctor_h_emb = torch.FloatTensor(
                np.asarray(traj['hindsight_embeddings'][t:t+1])).to(device)
            doctor_f_emb = torch.FloatTensor(
                np.asarray(traj['foresight_embeddings'][t:t+1])).to(device)

            better_action_idx, advantage, search_info = search_better_action(
                policy, world_model, state_t, doctor_action_idx,
                task_emb, doctor_h_emb, doctor_f_emb,
                search_radius=search_radius,
                mc_samples=C.MC_SAMPLES,
                sigma_threshold=C.SIGMA_THRESHOLD_SEARCH,
            )

            if advantage > 0:
                if t == 0:
                    continue

                # No semantic generation needed — use precomputed embeddings
                # DT branch ignores these, WM uses semantic=None

                confidence = search_info['confidence']
                base_alpha = max(C.ALPHA_MIN, 1.0 - global_step / C.ALPHA_DECAY_STEPS)
                if confidence >= C.CONFIDENCE_GATE:
                    alpha = base_alpha * confidence
                else:
                    alpha = 0.0

                opt_pi.zero_grad()
                opt_O.zero_grad()

                with torch.no_grad():
                    better_a_t = torch.tensor([better_action_idx], device=device)
                    _, _, pred_delta = world_model.predict(
                        state_t, better_a_t, semantic=None, mc_samples=C.MC_SAMPLES)

                # Policy loss: MSE on one-hot
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
                    task_embeddings=task_emb.unsqueeze(0),
                    hindsight_embeddings=doctor_h_emb.unsqueeze(0),  # DT ignores, precomputed
                    foresight_embeddings=doctor_f_emb.unsqueeze(0),  # DT ignores, precomputed
                )
                logits, _, _ = policy(**forward_kwargs)
                policy_pred = logits[:, -1, :]

                target_onehot = torch.zeros(1, C.VOCAB_SIZE, device=device)
                target_onehot[0, better_action_idx] = 1.0

                policy_loss = ((policy_pred - target_onehot.detach()) ** 2).mean()

                # World model loss: confidence-gated semi-bootstrap (WITHOUT semantic)
                a_onehot = torch.zeros(1, C.VOCAB_SIZE, device=device)
                a_onehot[0, better_action_idx] = 1.0
                mu_pred, log_sigma, _ = world_model(state_t, a_onehot, semantic=None)
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

        print(f"[Stage2-NoSemFull] Epoch {epoch+1}/{epochs} (global_step={global_step}): "
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
            best_path = os.path.join(args.logdir, 'best_checkpoint.pt')
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'policy_state_dict': policy.state_dict(),
                'world_model_state_dict': world_model.state_dict(),
                'advantage': avg_advantage,
                'policy_loss': avg_policy_loss,
                'world_loss': avg_world_loss,
                'confidence': avg_confidence,
                'alpha': avg_alpha,
                'bootstrap_pct': bootstrap_pct,
            }, best_path)
            print(f"  -> Saved best checkpoint (advantage={best_advantage:.4f})")

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.logdir, f'epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'policy_state_dict': policy.state_dict(),
                'world_model_state_dict': world_model.state_dict(),
                'advantage': avg_advantage,
            }, ckpt_path)

    print(f"\n[Stage2-NoSemFull] Training complete. Best advantage: {best_advantage:.4f} "
          f"(global_steps={global_step})")
    print(f"[Stage2-NoSemFull] Checkpoints in {args.logdir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=C.DATA_DIR)
    parser.add_argument('--logdir', default='./checkpoints_no_sem_full/stage2')
    parser.add_argument('--policy_ckpt', default='./checkpoints_no_sem_full/stage1/best_checkpoint/policy.pt')
    parser.add_argument('--world_model_ckpt',
                        default='./checkpoints_no_sem_full/stage1/best_checkpoint/world_model.pt')
    parser.add_argument('--epochs', type=int, default=C.STAGE2_EPOCHS)
    parser.add_argument('--selfplay_iterations', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--search_radius', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=C.BATCH_SIZE)
    parser.add_argument('--lr_pi', type=float, default=C.LR_PI)
    parser.add_argument('--lr_O', type=float, default=C.LR_O)
    parser.add_argument('--lambda_O', type=float, default=C.LAMBDA_O)
    parser.add_argument('--n_layer', type=int, default=C.N_LAYER)
    parser.add_argument('--n_head', type=int, default=C.N_HEAD)
    parser.add_argument('--n_embd', type=int, default=C.N_EMBD)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    train_stage2(args)
