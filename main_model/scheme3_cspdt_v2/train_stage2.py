"""
Stage 2: Counterfactual Self-Play with Confidence-Gated Semi-Bootstrap.

Key strategy (matching reference script):
  - Per-sample training: randomly sample one (trajectory, timestep) per step
  - Action search: discrete neighborhood ±2 + policy suggested action
  - Advantage: doctor_delta - best_candidate_delta
  - Confidence filtering: reject high-uncertainty action candidates
  - Policy loss: MSE (pred_onehot - better_action_onehot)^2
  - World model loss: semi-bootstrap (alpha * bootstrap + (1-alpha) * real)
  - Stratified sampling by SAPS2 severity
  - Best checkpoint by advantage
"""
import argparse
import os
import sys
import pickle

# Force offline mode before any HuggingFace imports
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config as C
from models.policy import build_policy
from models.world_model import WorldModel
from datasets.v3_semantic_dataset import V3SemanticDataset
from prompts.semantic_generator import SemanticGenerator
from prompts.text_encoder import PromptTextEncoder


def search_better_action(policy, world_model, state, current_action_idx,
                         task_emb, h_emb, f_emb,
                         search_radius=2, mc_samples=5, sigma_threshold=2.0):
    """
    Search for better action using world model saps2_head with confidence filtering.
    Discrete neighborhood exploration (±radius) + policy suggestion.

    Args:
        state: (1, 45) state tensor
        current_action_idx: int, doctor's action index
        task_emb, h_emb, f_emb: (1, 896) semantic embeddings
        search_radius: ±range for neighborhood search
        mc_samples: MC Dropout samples for uncertainty
        sigma_threshold: only accept candidates with sigma_mean < this value

    Returns:
        best_action_idx: int, best action found
        advantage: float, doctor_delta - best_delta (positive = improvement)
        search_info: dict with confidence, mu_pred, sigma_pred, sigma_mean
    """
    device = state.device
    sem_concat = torch.cat([task_emb, h_emb, f_emb], dim=-1)  # (1, 2688)

    # Doctor action's predicted delta (with confidence)
    with torch.no_grad():
        doc_action_t = torch.tensor([current_action_idx], device=device)
        doc_mu, doc_sigma, doctor_delta = world_model.predict(
            state, doc_action_t, semantic=sem_concat, mc_samples=mc_samples)
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
            state.unsqueeze(0),     # (1, 1, 45)
            None,                   # no actions at first step
            torch.zeros(1, 1, 1, device=device),  # dummy RTG
            torch.zeros(1, 1, 1, dtype=torch.long, device=device),  # dummy timestep
            task_emb.unsqueeze(0),  # (1, 1, 896)
            h_emb.unsqueeze(0),
            f_emb.unsqueeze(0),
        )  # (1, vocab_size)
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

        # Confidence filtering: skip candidates with high uncertainty
        if sigma_mean >= sigma_threshold:
            continue

        # Lower delta = better (SAPS2 decreased => patient improved)
        if candidate_delta_val < best_delta:
            best_delta = candidate_delta_val
            best_action_idx = action_idx
            best_mu_pred = mu_cand
            best_sigma_pred = sigma_cand
            best_sigma_mean = sigma_mean

    advantage = doctor_delta - best_delta

    # Confidence: inverse of normalised sigma, clipped to [0, 1]
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

    print(f"[Stage2] Loading v3 semantic datasets...")
    train_dataset = V3SemanticDataset(
        os.path.join(args.datadir, 'train_Phys45_v3.pickle'),
        C.CONTEXT_LENGTH, C.RTG_SCALE, gamma=C.RTG_GAMMA,
        language_emb_dim=C.LANGUAGE_EMB_DIM,
    )
    trajectories = train_dataset.trajectories
    print(f"[Stage2] Loaded {len(trajectories)} trajectories")

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

    print(f"[Stage2] Stratified sampling:")
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
    ).to(device)

    policy.load_state_dict(torch.load(args.policy_ckpt, map_location=device))
    world_model.load_state_dict(torch.load(args.world_model_ckpt, map_location=device))
    print(f"Loaded stage1 checkpoints from:\n  {args.policy_ckpt}\n  {args.world_model_ckpt}")

    # Load normalization parameters for denormalization during semantic generation
    norm_path = os.path.join(args.datadir, 'train_Phys45_v3_norm.pkl')
    if not os.path.exists(norm_path):
        norm_path = os.path.join(args.datadir, 'normalization_params.pkl')

    if os.path.exists(norm_path):
        with open(norm_path, 'rb') as f:
            norm_params = pickle.load(f)
        state_mean = norm_params['state_mean']
        state_std = norm_params['state_std'] + 1e-8
        print(f"[Stage2] Loaded normalization params from {norm_path}")
    else:
        print(f"[Stage2] Warning: normalization params not found, semantic generation may be inaccurate")
        state_mean = np.zeros(C.STATE_DIM)
        state_std = np.ones(C.STATE_DIM)

    # Initialize semantic generator for better_action embeddings
    text_encoder = PromptTextEncoder(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device=device,
        max_length=256,
    )
    semantic_generator = SemanticGenerator(text_encoder, device=device)
    print(f"[Stage2] Initialized semantic generator for better_action embeddings")

    opt_pi = optim.Adam(policy.parameters(), lr=args.lr_pi)
    opt_O = optim.Adam(world_model.parameters(), lr=args.lr_O * 0.5)
    lambda_O = args.lambda_O
    search_radius = args.search_radius

    epochs = args.epochs
    selfplay_iterations = args.selfplay_iterations

    print(f"[Stage2] Starting: {epochs} epochs x {selfplay_iterations} iterations, "
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
            # Sample one (trajectory, timestep)
            sample_idx = np.random.randint(n_traj)
            traj_idx = stratified_indices[sample_idx]
            traj = trajectories[traj_idx]

            states_np = traj['dem_observations']
            actions_np = traj['actions']
            T = len(states_np)

            if T < 2:
                continue

            t = np.random.randint(0, T - 1)

            state_t = torch.FloatTensor(states_np[t:t+1]).to(device)        # (1, 45)
            next_state_t = torch.FloatTensor(states_np[t+1:t+2]).to(device)  # (1, 45)
            doctor_action_idx = int(actions_np[t])

            # Get task embedding (action-agnostic, can be reused)
            task_emb = torch.FloatTensor(
                np.asarray(traj['task_embeddings'][t:t+1])).to(device)       # (1, 896)

            # Get doctor action's semantic embeddings for search
            doctor_h_emb = torch.FloatTensor(
                np.asarray(traj['hindsight_embeddings'][t:t+1])).to(device)   # (1, 896)
            doctor_f_emb = torch.FloatTensor(
                np.asarray(traj['foresight_embeddings'][t:t+1])).to(device)   # (1, 896)

            # Search for better action using doctor's semantics
            better_action_idx, advantage, search_info = search_better_action(
                policy, world_model, state_t, doctor_action_idx,
                task_emb, doctor_h_emb, doctor_f_emb,
                search_radius=search_radius,
                mc_samples=C.MC_SAMPLES,
                sigma_threshold=C.SIGMA_THRESHOLD_SEARCH,
            )

            if advantage > 0:
                # Skip t=0 to avoid invalid prev_state
                if t == 0:
                    continue

                # ✅ FIX: Regenerate semantic embeddings for better_action
                # Get previous state and denormalize for semantic generation
                prev_state_normalized = states_np[t-1]
                curr_state_normalized = states_np[t]

                # Denormalize states for clinical threshold matching
                prev_state_raw = prev_state_normalized * state_std + state_mean
                curr_state_raw = curr_state_normalized * state_std + state_mean

                # Fix: acuities is already shifted forward by 1 in dataset, so use [t] not [t+1]
                next_acuities_np = traj['acuities'][t]

                # Generate better_action's hindsight and foresight embeddings
                better_h_emb, better_f_emb = semantic_generator.generate_for_action(
                    prev_state_raw, curr_state_raw, next_acuities_np, better_action_idx
                )

                # Concatenate with task embedding (action-agnostic)
                sem_concat = torch.cat([task_emb, better_h_emb, better_f_emb], dim=-1)  # (1, 2688)

                # Compute confidence-gated dynamic alpha
                confidence = search_info['confidence']
                base_alpha = max(C.ALPHA_MIN, 1.0 - global_step / C.ALPHA_DECAY_STEPS)
                if confidence >= C.CONFIDENCE_GATE:
                    alpha = base_alpha * confidence
                else:
                    alpha = 0.0

                opt_pi.zero_grad()
                opt_O.zero_grad()

                # Get world model's predicted delta_saps2 for ATG token (stage 2 uses prediction)
                with torch.no_grad():
                    better_a_t = torch.tensor([better_action_idx], device=device)
                    _, _, pred_delta = world_model.predict(
                        state_t, better_a_t, semantic=sem_concat, mc_samples=C.MC_SAMPLES)

                # Policy loss: MSE on one-hot (matching reference script)
                # Use a minimal context window of just this timestep
                rtg_t = torch.FloatTensor(
                    traj['returns_to_go'][t:t+1]).unsqueeze(-1).unsqueeze(0).to(device)  # (1,1,1)
                ts_t = torch.LongTensor([t]).unsqueeze(-1).unsqueeze(0).to(device)      # (1,1,1)
                action_input = torch.LongTensor([[doctor_action_idx]]).unsqueeze(-1).to(device)  # (1,1,1)

                forward_kwargs = dict(
                    states=state_t.unsqueeze(0),      # (1, 1, 45)
                    actions=action_input,              # (1, 1, 1)
                    targets=None,
                    rtgs=rtg_t,                        # (1, 1, 1)
                    timesteps=ts_t,                    # (1, 1, 1)
                    task_embeddings=task_emb.unsqueeze(0),        # (1, 1, 896)
                    hindsight_embeddings=better_h_emb.unsqueeze(0),  # ✅ Use better_action semantics
                    foresight_embeddings=better_f_emb.unsqueeze(0),  # ✅ Use better_action semantics
                )
                # Pass delta_saps2 for ATG model types
                if use_atg:
                    forward_kwargs['delta_saps2'] = pred_delta.unsqueeze(0)  # (1, 1, 1)

                logits, _, _ = policy(**forward_kwargs)
                policy_pred = logits[:, -1, :]  # (1, vocab_size)

                # Target: one-hot of better action
                target_onehot = torch.zeros(1, C.VOCAB_SIZE, device=device)
                target_onehot[0, better_action_idx] = 1.0

                policy_loss = ((policy_pred - target_onehot.detach()) ** 2).mean()

                # World model loss: confidence-gated semi-bootstrap
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

        print(f"Epoch {epoch+1}/{epochs} (global_step={global_step}): "
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

        # Save best checkpoint
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

        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.logdir, f'epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'policy_state_dict': policy.state_dict(),
                'world_model_state_dict': world_model.state_dict(),
                'advantage': avg_advantage,
            }, ckpt_path)

    print(f"\n[Stage2] Training complete. Best advantage: {best_advantage:.4f} "
          f"(global_steps={global_step})")
    print(f"[Stage2] Checkpoints in {args.logdir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=C.DATA_DIR)
    parser.add_argument('--logdir', default='./checkpoints/stage2')
    parser.add_argument('--policy_ckpt', default='./checkpoints/stage1/best_checkpoint/policy.pt',
                        help='Path to Stage1 policy checkpoint (default: best_checkpoint)')
    parser.add_argument('--world_model_ckpt',
                        default='./checkpoints/stage1/best_checkpoint/world_model.pt',
                        help='Path to Stage1 world model checkpoint (default: best_checkpoint)')
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
