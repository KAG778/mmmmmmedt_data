"""
Stage 2 ablation: Counterfactual Self-Play with zeroed semantic components in Policy.
WM without semantic embeddings.
Usage: python train_stage2_ablation.py --variant no_task
"""
import argparse
import os
import sys
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config_ablation_semantic as C
from config_ablation_semantic import ABLATE_VARIANTS
from models.policy import build_policy
from models.world_model import WorldModel
from datasets.v3_semantic_dataset import V3SemanticDataset
from prompts.semantic_generator import SemanticGenerator
from prompts.text_encoder import PromptTextEncoder


def _apply_ablation(task_emb, h_emb, f_emb, ablate_cfg):
    if ablate_cfg.get('task', False):
        task_emb = torch.zeros_like(task_emb)
    if ablate_cfg.get('hindsight', False):
        h_emb = torch.zeros_like(h_emb)
    if ablate_cfg.get('foresight', False):
        f_emb = torch.zeros_like(f_emb)
    return task_emb, h_emb, f_emb


def search_better_action(policy, world_model, state, current_action_idx,
                         task_emb, h_emb, f_emb, ablate_cfg,
                         search_radius=2, mc_samples=5, sigma_threshold=2.0):
    device = state.device

    with torch.no_grad():
        doc_action_t = torch.tensor([current_action_idx], device=device)
        doc_mu, doc_sigma, doctor_delta = world_model.predict(
            state, doc_action_t, semantic=None, mc_samples=mc_samples)
        doctor_delta = doctor_delta.item()

    candidates = set()
    for offset in range(-search_radius, search_radius + 1):
        idx = current_action_idx + offset
        if 0 <= idx < C.VOCAB_SIZE:
            candidates.add(idx)

    task_pol, h_pol, f_pol = _apply_ablation(
        task_emb.clone(), h_emb.clone(), f_emb.clone(), ablate_cfg)

    with torch.no_grad():
        policy_logits = policy.get_action_logits(
            state.unsqueeze(0),
            None,
            torch.zeros(1, 1, 1, device=device),
            torch.zeros(1, 1, 1, dtype=torch.long, device=device),
            task_pol.unsqueeze(0),
            h_pol.unsqueeze(0),
            f_pol.unsqueeze(0),
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
    ablate_cfg = ABLATE_VARIANTS[args.variant]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logdir = os.path.join(args.logdir_base, args.variant, 'stage2')
    os.makedirs(logdir, exist_ok=True)

    print(f"[Stage2-Ablation] Variant: {args.variant}")
    print(f"[Stage2-Ablation] Ablate config: {ablate_cfg}")

    print(f"[Stage2-Ablation] Loading v3 semantic datasets...")
    train_dataset = V3SemanticDataset(
        os.path.join(args.datadir, 'train_Phys45_v3.pickle'),
        C.CONTEXT_LENGTH, C.RTG_SCALE, gamma=C.RTG_GAMMA,
        language_emb_dim=C.LANGUAGE_EMB_DIM,
    )
    trajectories = train_dataset.trajectories
    print(f"[Stage2-Ablation] Loaded {len(trajectories)} trajectories")

    low_indices, mid_indices, high_indices = [], [], []
    for i, traj in enumerate(trajectories):
        init_saps2 = float(traj['acuities_original_init'][2])
        if init_saps2 < 75:
            low_indices.append(i)
        elif init_saps2 < 85:
            mid_indices.append(i)
        else:
            high_indices.append(i)

    print(f"[Stage2-Ablation] Stratified: Low={len(low_indices)}, Mid={len(mid_indices)}, High={len(high_indices)}")
    stratified_indices = low_indices + mid_indices + high_indices
    n_traj = len(stratified_indices)

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
        use_semantic=False,
    ).to(device)

    stage1_dir = os.path.join(args.logdir_base, args.variant, 'stage1', args.stage1_ckpt)
    policy_path = os.path.join(stage1_dir, 'policy.pt')
    wm_path = os.path.join(stage1_dir, 'world_model.pt')
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    world_model.load_state_dict(torch.load(wm_path, map_location=device))
    print(f"Loaded stage1: {policy_path} + {wm_path}")

    norm_path = os.path.join(args.datadir, 'train_Phys45_v3_norm.pkl')
    if not os.path.exists(norm_path):
        norm_path = os.path.join(args.datadir, 'normalization_params.pkl')
    if os.path.exists(norm_path):
        with open(norm_path, 'rb') as f:
            norm_params = pickle.load(f)
        state_mean = norm_params['state_mean']
        state_std = norm_params['state_std'] + 1e-8
        print(f"[Stage2-Ablation] Loaded normalization params")
    else:
        print(f"[Stage2-Ablation] Warning: normalization params not found")
        state_mean = np.zeros(C.STATE_DIM)
        state_std = np.ones(C.STATE_DIM)

    need_sem_gen = not ablate_cfg.get('hindsight', True) or not ablate_cfg.get('foresight', True)
    if need_sem_gen:
        text_encoder = PromptTextEncoder(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            device=device, max_length=256,
        )
        semantic_generator = SemanticGenerator(text_encoder, device=device)
        print(f"[Stage2-Ablation] Initialized semantic generator")
    else:
        semantic_generator = None
        print(f"[Stage2-Ablation] Both h+f ablated, skipping semantic generator")

    opt_pi = optim.Adam(policy.parameters(), lr=args.lr_pi)
    opt_O = optim.Adam(world_model.parameters(), lr=args.lr_O * 0.5)
    lambda_O = args.lambda_O
    search_radius = args.search_radius

    epochs = args.epochs
    selfplay_iterations = args.selfplay_iterations
    print(f"[Stage2-Ablation] Starting: {epochs} epochs x {selfplay_iterations} iter")

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

            better_action_idx, advantage, search_info = search_better_action(
                policy, world_model, state_t, doctor_action_idx,
                task_emb, doctor_h_emb, doctor_f_emb, ablate_cfg,
                search_radius=search_radius,
                mc_samples=C.MC_SAMPLES,
                sigma_threshold=C.SIGMA_THRESHOLD_SEARCH,
            )

            if advantage > 0:
                if t == 0:
                    continue

                if semantic_generator is not None:
                    prev_state_normalized = states_np[t-1]
                    curr_state_normalized = states_np[t]
                    prev_state_raw = prev_state_normalized * state_std + state_mean
                    curr_state_raw = curr_state_normalized * state_std + state_mean
                    next_acuities_np = traj['acuities'][t]
                    better_h_emb, better_f_emb = semantic_generator.generate_for_action(
                        prev_state_raw, curr_state_raw, next_acuities_np, better_action_idx
                    )
                else:
                    better_h_emb = doctor_h_emb
                    better_f_emb = doctor_f_emb

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

                task_fwd, h_fwd, f_fwd = _apply_ablation(
                    task_emb.clone(), better_h_emb.clone(), better_f_emb.clone(), ablate_cfg)

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
                    task_embeddings=task_fwd.unsqueeze(0),
                    hindsight_embeddings=h_fwd.unsqueeze(0),
                    foresight_embeddings=f_fwd.unsqueeze(0),
                )
                if use_atg:
                    forward_kwargs['delta_saps2'] = pred_delta.unsqueeze(0)

                logits, _, _ = policy(**forward_kwargs)
                policy_pred = logits[:, -1, :]

                target_onehot = torch.zeros(1, C.VOCAB_SIZE, device=device)
                target_onehot[0, better_action_idx] = 1.0
                policy_loss = ((policy_pred - target_onehot.detach()) ** 2).mean()

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

        print(f"[{args.variant}] Epoch {epoch+1}/{epochs} (step={global_step}): "
              f"Adv={avg_advantage:.4f}, ImpRate={improvement_rate:.2%}, "
              f"PiLoss={avg_policy_loss:.4f}, WLoss={avg_world_loss:.4f}, "
              f"Conf={avg_confidence:.3f}, Alpha={avg_alpha:.3f}, Boot={bootstrap_pct:.2%}")

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
                'policy_loss': avg_policy_loss,
                'world_loss': avg_world_loss,
                'confidence': avg_confidence,
                'alpha': avg_alpha,
                'bootstrap_pct': bootstrap_pct,
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

    print(f"\n[Stage2-{args.variant}] Complete. Best advantage: {best_advantage:.4f}")
    print(f"[Stage2-{args.variant}] Checkpoints in {logdir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True,
                        choices=list(ABLATE_VARIANTS.keys()))
    parser.add_argument('--datadir', default=C.DATA_DIR)
    parser.add_argument('--logdir_base', default=os.path.join(SCRIPT_DIR, 'checkpoints_ablation'))
    parser.add_argument('--stage1_ckpt', default='best_checkpoint')
    parser.add_argument('--epochs', type=int, default=C.STAGE2_EPOCHS)
    parser.add_argument('--selfplay_iterations', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--search_radius', type=int, default=2)
    parser.add_argument('--lr_pi', type=float, default=C.LR_PI)
    parser.add_argument('--lr_O', type=float, default=C.LR_O)
    parser.add_argument('--lambda_O', type=float, default=C.LAMBDA_O)
    parser.add_argument('--n_layer', type=int, default=C.N_LAYER)
    parser.add_argument('--n_head', type=int, default=C.N_HEAD)
    parser.add_argument('--n_embd', type=int, default=C.N_EMBD)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    stage1_base = os.path.join(args.logdir_base, args.variant, 'stage1')
    if args.stage1_ckpt == 'best_checkpoint':
        if os.path.exists(os.path.join(stage1_base, 'epoch_100')):
            args.stage1_ckpt = 'epoch_100'
        else:
            import glob
            epoch_dirs = sorted(glob.glob(os.path.join(stage1_base, 'epoch_*')))
            if epoch_dirs:
                args.stage1_ckpt = os.path.basename(epoch_dirs[-1])
            else:
                args.stage1_ckpt = 'best_checkpoint'

    train_stage2(args)
