"""
Stage 1 Training: Algorithm-specific pretraining for all baselines.

BC  - Behavior Cloning (MSE on action prediction)
DT  - Decision Transformer with RTG conditioning
IQL - Implicit Q-Learning: expectile-regressed V(s) + advantage-weighted policy
BCQ - Batch-Constrained Q-learning: twin Q-nets + perturbation network
CQL - Conservative Q-Learning: twin Q-nets + target nets + CQL penalty
"""

import os
import sys
import random
import argparse
import yaml
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.policies import get_policy, IQLPolicy, BCQPolicy, CQLPolicy, DTPolicy, DQNPolicy, TD3BCPolicy, MLP
from models.world_models import get_world_model


class TrajectoryDataset(Dataset):
    """Dataset for medical trajectories with stratified sampling support"""

    def __init__(self, data_path, max_length=20, action_dim=25, stratified_sampling=True, low_oversample=3, normalize_states=True):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.max_length = max_length
        self.action_dim = action_dim
        self.stratified_sampling = stratified_sampling
        self.low_oversample = low_oversample
        self.normalize_states = normalize_states

        # Load normalization parameters
        if self.normalize_states:
            norm_path = os.path.join(os.path.dirname(data_path), 'normalization_params.pkl')
            if os.path.exists(norm_path):
                with open(norm_path, 'rb') as f:
                    norm_params = pickle.load(f)
                self.state_mean = norm_params['state_mean']
                self.state_std = norm_params['state_std'] + 1e-8
                print(f"[TrajectoryDataset] Loaded normalization params from {norm_path}")
            else:
                print(f"[TrajectoryDataset] Warning: normalization requested but {norm_path} not found, skipping")
                self.normalize_states = False

        # Stratify by initial SAPS2
        self.indices = list(range(len(self.data)))
        self.stratum_labels = []

        for idx in self.indices:
            traj = self.data[idx]
            if 'acuities' in traj and traj['acuities'] is not None:
                init_saps2 = float(traj['acuities'][0, 2])
            else:
                init_saps2 = 70.0  # Default to low

            if init_saps2 < 75:
                self.stratum_labels.append('low')
            elif init_saps2 < 85:
                self.stratum_labels.append('mid')
            else:
                self.stratum_labels.append('high')

        # Create oversampled indices for stratum-aware sampling
        if stratified_sampling:
            low_indices = [i for i, s in enumerate(self.stratum_labels) if s == 'low']
            mid_indices = [i for i, s in enumerate(self.stratum_labels) if s == 'mid']
            high_indices = [i for i, s in enumerate(self.stratum_labels) if s == 'high']

            print(f"Stratified sampling enabled:")
            print(f"  Low (<75):   {len(low_indices):5d} -> oversampled by {low_oversample}x")
            print(f"  Mid (75-85): {len(mid_indices):5d}")
            print(f"  High (>=85):  {len(high_indices):5d}")

            # Oversample low stratum
            self.oversampled_indices = (
                low_indices * low_oversample + mid_indices + high_indices
            )
            self.traj_length = len(self.oversampled_indices)
        else:
            self.traj_length = len(self.data)

    def __len__(self):
        return self.traj_length

    def _get_original_index(self, idx):
        """Get original data index, accounting for oversampling"""
        if self.stratified_sampling:
            return self.oversampled_indices[idx]
        return idx

    def __getitem__(self, idx):
        orig_idx = self._get_original_index(idx)
        traj = self.data[orig_idx]

        # Use dem_observations for states (45 dimensions)
        states = traj['dem_observations']  # (T, 45)

        # Normalize states if enabled
        if self.normalize_states:
            states = (states - self.state_mean) / self.state_std

        # Keep actions as discrete indices (0-24)
        actions_discrete = traj['actions']  # (T,)
        actions = torch.LongTensor(actions_discrete)

        # Compute reward from SAPS2 delta: reward = -delta (improvement = positive reward)
        if 'acuities' in traj and traj['acuities'] is not None:
            acuities = traj['acuities'][:, 2]
            saps2_delta = acuities[1:] - acuities[:-1]
            rewards = -saps2_delta  # negative delta = positive reward
        else:
            rewards = traj.get('rewards', np.zeros(len(states)))
            saps2_delta = -np.array(rewards)

        # Compute returns-to-go from rewards
        rewards_full = np.concatenate([[0], rewards])  # prepend 0 for initial state
        rtg = np.cumsum(rewards_full[::-1])[::-1].copy()

        # Semantic embeddings (only if world model needs them)
        semantic = None

        result = {
            'states': torch.FloatTensor(states),
            'actions': actions,  # Now discrete indices (LongTensor)
            'saps2_delta': torch.FloatTensor(saps2_delta),
            'rewards': torch.FloatTensor(rewards),
            'returns_to_go': torch.FloatTensor(rtg),
            'length': len(states)
        }
        if semantic is not None:
            result['semantic'] = torch.FloatTensor(semantic)

        return result


def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    max_len = max([b['length'] for b in batch])

    states, actions, saps2_delta, masks = [], [], [], []
    rewards_list, rtg_list, semantic_list = [], [], []
    has_semantic = 'semantic' in batch[0]

    for item in batch:
        length = item['length']
        s = item['states']
        a = item['actions']
        d = item['saps2_delta']
        r = item['rewards']
        rtg = item['returns_to_go']

        pad_len = max_len - length
        s_pad = torch.cat([s, torch.zeros(pad_len, s.shape[1])], dim=0)
        # Actions are now discrete indices (LongTensor), pad with 0
        a_pad = torch.cat([a, torch.zeros(pad_len, dtype=torch.long)], dim=0)
        d_pad = torch.cat([d, torch.zeros(pad_len)], dim=0)
        r_pad = torch.cat([r, torch.zeros(pad_len)], dim=0)
        rtg_pad = torch.cat([rtg, torch.zeros(pad_len)], dim=0)

        states.append(s_pad)
        actions.append(a_pad)
        saps2_delta.append(d_pad)
        rewards_list.append(r_pad)
        rtg_list.append(rtg_pad)

        if has_semantic:
            sem = item['semantic']
            sem_pad = torch.cat([sem, torch.zeros(pad_len, sem.shape[1])], dim=0)
            semantic_list.append(sem_pad)

        full_mask = torch.cat([torch.ones(length), torch.zeros(pad_len)])
        mask = full_mask[:-1]
        masks.append(mask)

    result = {
        'states': torch.stack(states),
        'actions': torch.stack(actions),
        'saps2_delta': torch.stack(saps2_delta),
        'rewards': torch.stack(rewards_list),
        'returns_to_go': torch.stack(rtg_list),
        'mask': torch.stack(masks)
    }
    if has_semantic:
        result['semantic'] = torch.stack(semantic_list)
    return result


def compute_loss(pred, target, mask, loss_type='mse', focal_alpha=None, focal_gamma=None):
    """Compute loss with optional focal weighting"""
    if loss_type == 'mse':
        if mask.dim() == 2 and pred.dim() == 3:
            mask = mask.unsqueeze(-1)
        loss = ((pred - target) ** 2) * mask
        return loss.sum() / mask.sum()

    elif loss_type == 'focal':
        mse = (pred - target) ** 2
        weight = torch.abs(target) ** focal_gamma
        if focal_alpha is not None:
            weight = focal_alpha * weight + (1 - focal_alpha)
        if mask.dim() == 2 and pred.dim() == 3:
            mask = mask.unsqueeze(-1)
        loss = weight * mse * mask
        return loss.sum() / mask.sum()

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def expectile_loss(pred, target, expectile=0.8):
    """Asymmetric L2 loss for IQL value function expectile regression."""
    diff = target - pred
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * (diff ** 2)).mean()


def _get_semantic(batch, device):
    """Extract semantic tensor from batch for world model, reshaped to 2D."""
    if 'semantic' not in batch:
        return None
    sem = batch['semantic'].to(device)  # (B, T, sem_dim)
    B, T, D = sem.shape
    return sem.reshape(B * T, D)


# ============================================================================
# Algorithm-specific training functions
# ============================================================================

def train_bc_stage1(policy, world_model, dataloader, device, epochs, lr, output_dir, loss_config):
    """Standard BC + World Model training."""
    policy_opt = optim.Adam(policy.parameters(), lr=lr)
    world_opt = optim.Adam(world_model.parameters(), lr=lr)
    loss_type = loss_config.get('type', 'mse')
    best_loss = float('inf')

    for epoch in range(epochs):
        policy.train()
        world_model.train()
        total_policy_loss = total_world_loss = total_saps2_loss = num_batches = 0

        for batch in dataloader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            saps2_delta = batch['saps2_delta'].to(device)
            mask = batch['mask'].to(device)

            # Policy: BC action prediction (discrete actions, use CrossEntropy)
            policy_opt.zero_grad()
            action_logits = policy(states[:, :-1])  # (B, T-1, num_actions)
            target_actions = actions[:, 1:].long()  # (B, T-1)
            # Reshape for CrossEntropyLoss: (B*T-1, num_actions) and (B*T-1,)
            action_logits_flat = action_logits.reshape(-1, action_logits.shape[-1])  # (B*T, 25)
            target_actions_flat = target_actions.reshape(-1)
            policy_loss = F.cross_entropy(action_logits_flat, target_actions_flat)
            policy_loss.backward()
            policy_opt.step()

            # World Model
            # World Model
            world_opt.zero_grad()
            sem_wm = None
            if 'semantic' in batch:
                sem_wm = batch['semantic'][:, :-1].to(device)  # (B, T-1, sem_dim)
            world_out = world_model(states[:, :-1], actions[:, :-1], semantic_context=sem_wm)
            state_loss = compute_loss(world_out['mu'], states[:, 1:], mask, loss_type,
                                      loss_config.get('focal_alpha'), loss_config.get('focal_gamma'))
            saps2_loss = compute_loss(world_out['saps2_delta'], saps2_delta, mask, loss_type,
                                      loss_config.get('focal_alpha'), loss_config.get('focal_gamma'))
            world_loss = state_loss + saps2_loss
            world_loss.backward()
            world_opt.step()

            total_policy_loss += policy_loss.item()
            total_world_loss += state_loss.item()
            total_saps2_loss += saps2_loss.item()
            num_batches += 1

        avg_p = total_policy_loss / num_batches
        avg_w = total_world_loss / num_batches
        avg_s = total_saps2_loss / num_batches
        avg_total = avg_p + avg_w + avg_s

        print(f"Epoch {epoch+1}/{epochs}: Policy={avg_p:.4f}, World={avg_w:.4f}, "
              f"SAPS2={avg_s:.4f}, Total={avg_total:.4f}")

        if avg_total < best_loss:
            best_loss = avg_total
            _save_checkpoint(policy, world_model, policy_opt, world_opt, epoch, avg_total, output_dir)
            print(f"  -> Saved best checkpoint (loss={best_loss:.4f})")

    return output_dir / 'best_checkpoint.pt'


def train_dt_stage1(policy, world_model, dataloader, device, epochs, lr, output_dir, loss_config):
    """Decision Transformer with RTG conditioning + World Model."""
    policy_opt = optim.Adam(policy.parameters(), lr=lr)
    world_opt = optim.Adam(world_model.parameters(), lr=lr)
    loss_type = loss_config.get('type', 'mse')
    best_loss = float('inf')
    use_rtg = getattr(policy, 'use_rtg', True)

    for epoch in range(epochs):
        policy.train()
        world_model.train()
        total_policy_loss = total_world_loss = total_saps2_loss = num_batches = 0

        for batch in dataloader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            saps2_delta = batch['saps2_delta'].to(device)
            rtg = batch['returns_to_go'].to(device)
            mask = batch['mask'].to(device)

            B, T, _ = states.shape

            # ---- DT: Policy update with RTG conditioning ----
            policy_opt.zero_grad()

            input_states = states[:, :-1]    # s_0 .. s_{T-2}
            input_actions = actions[:, :-1]  # a_0 .. a_{T-2}
            target_actions = actions[:, 1:]  # a_1 .. a_{T-1}

            # Pass RTG if DT supports it
            input_rtg = rtg[:, :-1] if use_rtg else None

            action_logits = policy(input_states, actions=input_actions, rtg=input_rtg)  # (B, T-1, num_actions)
            target_actions = target_actions.long()  # (B, T-1)
            if action_logits.dim() == 3:
                seq_len = min(action_logits.shape[1], target_actions.shape[1])
                action_logits = action_logits[:, :seq_len]
                target_actions = target_actions[:, :seq_len]

            # Use CrossEntropy for discrete actions
            action_logits_flat = action_logits.reshape(-1, action_logits.shape[-1])  # (B*T, 25)
            target_actions_flat = target_actions.reshape(-1)
            policy_loss = F.cross_entropy(action_logits_flat, target_actions_flat)
            policy_loss.backward()
            policy_opt.step()

            # World Model (same as BC)
            # World Model
            world_opt.zero_grad()
            sem_wm = None
            if 'semantic' in batch:
                sem_wm = batch['semantic'][:, :-1].to(device)  # (B, T-1, sem_dim)
            world_out = world_model(states[:, :-1], actions[:, :-1], semantic_context=sem_wm)
            state_loss = compute_loss(world_out['mu'], states[:, 1:], mask, loss_type,
                                      loss_config.get('focal_alpha'), loss_config.get('focal_gamma'))
            saps2_loss = compute_loss(world_out['saps2_delta'], saps2_delta, mask, loss_type,
                                      loss_config.get('focal_alpha'), loss_config.get('focal_gamma'))
            world_loss = state_loss + saps2_loss
            world_loss.backward()
            world_opt.step()

            total_policy_loss += policy_loss.item()
            total_world_loss += state_loss.item()
            total_saps2_loss += saps2_loss.item()
            num_batches += 1

        avg_p = total_policy_loss / num_batches
        avg_w = total_world_loss / num_batches
        avg_s = total_saps2_loss / num_batches
        avg_total = avg_p + avg_w + avg_s

        print(f"Epoch {epoch+1}/{epochs}: Policy={avg_p:.4f}, World={avg_w:.4f}, "
              f"SAPS2={avg_s:.4f}, Total={avg_total:.4f}")

        if avg_total < best_loss:
            best_loss = avg_total
            _save_checkpoint(policy, world_model, policy_opt, world_opt, epoch, avg_total, output_dir)
            print(f"  -> Saved best checkpoint (loss={best_loss:.4f})")

    return output_dir / 'best_checkpoint.pt'


def train_iql_stage1(policy, world_model, dataloader, device, epochs, lr, output_dir, loss_config):
    """IQL: Value function (expectile regression) + advantage-weighted policy + World Model."""
    policy_opt = optim.Adam(policy.network.parameters(), lr=lr)
    value_opt = optim.Adam(policy.value_net.parameters(), lr=lr * 0.1)  # 10x lower for stability
    world_opt = optim.Adam(world_model.parameters(), lr=lr)

    expectile = policy.config.get("expectile", 0.7)
    beta = policy.config.get("beta", 3.0)
    gamma = policy.config.get("gamma", 0.99)
    loss_type = loss_config.get('type', 'mse')
    best_loss = float('inf')

    # Running mean/std for V-target normalization
    v_running_mean = 0.0
    v_running_std = 1.0

    print(f"  IQL params: expectile={expectile}, beta={beta}, gamma={gamma}")

    for epoch in range(epochs):
        policy.train()
        world_model.train()
        total_v_loss = total_pi_loss = total_world_loss = total_saps2_loss = num_batches = 0

        for batch in dataloader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            saps2_delta = batch['saps2_delta'].to(device)
            rewards = batch['rewards'].to(device)
            mask = batch['mask'].to(device)

            B, T, _ = states.shape

            # ---- IQL: Value function update via expectile regression ----
            value_opt.zero_grad()
            v_pred = policy.get_value(states[:, :-1]).squeeze(-1)  # (B, T-1)
            with torch.no_grad():
                v_next = policy.get_value(states[:, 1:]).squeeze(-1)  # (B, T-1)
                # Clip v_next to prevent explosion
                v_next = v_next.clamp(-50, 50)
                # Reward normalization: scale down large SAPS2 deltas
                r_scaled = rewards[:, :T-1].clamp(-20, 20)
                q_target = r_scaled + gamma * v_next
                # Normalize target
                qt_mean = q_target.mean()
                qt_std = q_target.std().clamp(min=1.0)
                q_target_norm = (q_target - qt_mean) / qt_std

            v_loss = expectile_loss(v_pred, q_target_norm.detach(), expectile)
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.value_net.parameters(), 1.0)
            value_opt.step()

            # ---- IQL: Policy update with advantage weighting ----
            policy_opt.zero_grad()
            action_logits = policy(states[:, :-1])  # (B, T-1, num_actions)
            target_actions = actions[:, 1:]       # (B, T-1, A)

            # Handle dimension mismatch
            if action_logits.dim() == 2:
                # BCPolicy with 3D input returns 3D, but 2D means batch flatten
                # Reshape: assume action_logits is (B*(T-1), A), reshape to (B, T-1, A)
                B_exp = target_actions.shape[0]
                T_exp = target_actions.shape[1]
                action_logits = action_logits.reshape(B_exp, T_exp, -1)

            seq_len = min(action_logits.shape[1], target_actions.shape[1])
            action_logits = action_logits[:, :seq_len]
            target_actions = target_actions[:, :seq_len]

            with torch.no_grad():
                v_curr = policy.get_value(states[:, :-1]).squeeze(-1)[:, :seq_len]
                adv = (q_target_norm[:, :seq_len] - v_curr).clamp(-5, 5)
                adv_weights = torch.softmax(adv * beta, dim=-1)  # (B, seq_len)
                adv_weights = adv_weights.unsqueeze(-1)  # (B, seq_len, 1)

            # Use CrossEntropy for discrete actions
            target_actions = target_actions.long()  # (B, T-1)
            action_logits_flat = action_logits.reshape(-1, action_logits.shape[-1])  # (B*T, 25)
            target_actions_flat = target_actions.reshape(-1)  # (B*T,)
            adv_weights_flat = adv_weights.reshape(-1)  # (B*T,)
            pi_loss = F.cross_entropy(action_logits_flat, target_actions_flat, reduction='none')
            pi_loss = (pi_loss * adv_weights_flat).mean()
            pi_loss.backward()
            policy_opt.step()

            # ---- World Model ----
            # World Model
            world_opt.zero_grad()
            sem_wm = None
            if 'semantic' in batch:
                sem_wm = batch['semantic'][:, :-1].to(device)  # (B, T-1, sem_dim)
            world_out = world_model(states[:, :-1], actions[:, :-1], semantic_context=sem_wm)
            state_loss = compute_loss(world_out['mu'], states[:, 1:], mask, loss_type,
                                      loss_config.get('focal_alpha'), loss_config.get('focal_gamma'))
            saps2_loss = compute_loss(world_out['saps2_delta'], saps2_delta, mask, loss_type,
                                      loss_config.get('focal_alpha'), loss_config.get('focal_gamma'))
            world_loss = state_loss + saps2_loss
            world_loss.backward()
            world_opt.step()

            total_v_loss += v_loss.item()
            total_pi_loss += pi_loss.item()
            total_world_loss += state_loss.item()
            total_saps2_loss += saps2_loss.item()
            num_batches += 1

        avg_v = total_v_loss / num_batches
        avg_pi = total_pi_loss / num_batches
        avg_w = total_world_loss / num_batches
        avg_s = total_saps2_loss / num_batches
        avg_total = avg_v + avg_pi + avg_w + avg_s

        print(f"Epoch {epoch+1}/{epochs}: VLoss={avg_v:.4f}, PiLoss={avg_pi:.4f}, "
              f"World={avg_w:.4f}, SAPS2={avg_s:.4f}, Total={avg_total:.4f}")

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'world_model_state_dict': world_model.state_dict(),
                'policy_opt_state_dict': policy_opt.state_dict(),
                'world_opt_state_dict': world_opt.state_dict(),
                'value_opt_state_dict': value_opt.state_dict(),
                'loss': avg_total,
            }, output_dir / 'best_checkpoint.pt')
            print(f"  -> Saved best checkpoint (loss={best_loss:.4f})")

    return output_dir / 'best_checkpoint.pt'


def train_bcq_stage1(policy, world_model, dataloader, device, epochs, lr, output_dir, loss_config):
    """BCQ: Twin Q-networks with target networks + perturbation network + BC policy + World Model."""
    policy_opt = optim.Adam(policy.network.parameters(), lr=lr)
    q_opt = optim.Adam(list(policy.q1_net.parameters()) + list(policy.q2_net.parameters()), lr=lr)
    perturb_opt = optim.Adam(policy.perturb_net.parameters(), lr=lr)
    world_opt = optim.Adam(world_model.parameters(), lr=lr)

    gamma = policy.config.get("gamma", 0.99)
    tau = policy.config.get("tau", 0.005)
    loss_type = loss_config.get('type', 'mse')
    best_loss = float('inf')

    # Create proper target Q-networks (deep copies, independent parameters)
    hidden_dim = policy.config.get("hidden_dim", 256)
    num_layers = policy.config.get("num_layers", 3)
    target_q1 = MLP(policy.state_dim + policy.action_dim, hidden_dim, 1, num_layers).to(device)
    target_q2 = MLP(policy.state_dim + policy.action_dim, hidden_dim, 1, num_layers).to(device)
    target_q1.load_state_dict(policy.q1_net.state_dict())
    target_q2.load_state_dict(policy.q2_net.state_dict())
    for p in list(target_q1.parameters()) + list(target_q2.parameters()):
        p.requires_grad = False

    print(f"  BCQ params: gamma={gamma}, tau={tau}")

    for epoch in range(epochs):
        policy.train()
        world_model.train()
        total_q_loss = total_pi_loss = total_perturb_loss = 0
        total_world_loss = total_saps2_loss = num_batches = 0

        for batch in dataloader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            saps2_delta = batch['saps2_delta'].to(device)
            rewards = batch['rewards'].to(device)
            mask = batch['mask'].to(device)

            B, T, _ = states.shape
            s_t = states[:, :-1]
            a_t = actions[:, :-1]
            s_next = states[:, 1:]
            r_t = rewards[:, :T-1]
            a_next = actions[:, 1:]

            flat_s = s_t.reshape(-1, s_t.shape[-1])  # (B*(T-1), state_dim)
            flat_a = a_t.reshape(-1)  # Discrete actions: (B*(T-1),)
            flat_s_next = s_next[:, :T-1].reshape(-1, s_next.shape[-1])
            flat_a_next = a_next.reshape(-1)  # Discrete actions: (B*(T-1),)

            # ---- BCQ: Q-network update with target networks ----
            q_opt.zero_grad()
            # Use _get_q_input to handle discrete action conversion
            q1 = policy.q1_net(policy._get_q_input(flat_s, flat_a)).reshape(B, -1)
            q2 = policy.q2_net(policy._get_q_input(flat_s, flat_a)).reshape(B, -1)

            with torch.no_grad():
                # Use TARGET Q-networks for Bellman target (not online!)
                q1_target = target_q1(policy._get_q_input(flat_s_next, flat_a_next)).reshape(B, -1)
                q2_target = target_q2(policy._get_q_input(flat_s_next, flat_a_next)).reshape(B, -1)
                q_target = r_t + gamma * torch.min(q1_target, q2_target)

            q1_loss = ((q1 - q_target) ** 2).mean()
            q2_loss = ((q2 - q_target) ** 2).mean()
            q_loss = q1_loss + q2_loss
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(policy.q1_net.parameters()) + list(policy.q2_net.parameters()), 1.0)
            q_opt.step()

            # Soft update target Q-networks: target = tau * online + (1-tau) * target
            for o, t in zip(policy.q1_net.parameters(), target_q1.parameters()):
                t.data.copy_(tau * o.data + (1 - tau) * t.data)
            for o, t in zip(policy.q2_net.parameters(), target_q2.parameters()):
                t.data.copy_(tau * o.data + (1 - tau) * t.data)

            # ---- BCQ: Policy update (BC loss with CrossEntropy) ----
            policy_opt.zero_grad()
            action_logits = policy(s_t)  # (B, T-1, 25)
            a_t_long = a_t.long()  # (B, T-1)
            # Reshape for CrossEntropy: (B*T-1, 25) and (B*T-1,)
            action_logits_flat = action_logits.reshape(-1, action_logits.shape[-1])
            a_t_flat = a_t_long.reshape(-1)
            pi_loss = F.cross_entropy(action_logits_flat, a_t_flat)
            pi_loss.backward()
            policy_opt.step()

            # ---- BCQ: Perturbation network (maximize Q) ----
            perturb_opt.zero_grad()
            perturbed_a = policy.perturb_action(flat_s, flat_a)
            # Use _get_q_input to handle the perturbed action properly
            q1_perturb = policy.q1_net(policy._get_q_input(flat_s, flat_a))
            perturb_loss = -q1_perturb.mean()
            perturb_loss.backward()
            perturb_opt.step()

            # ---- World Model ----
            # World Model
            world_opt.zero_grad()
            sem_wm = None
            if 'semantic' in batch:
                sem_wm = batch['semantic'][:, :-1].to(device)  # (B, T-1, sem_dim)
            world_out = world_model(states[:, :-1], actions[:, :-1], semantic_context=sem_wm)
            state_loss = compute_loss(world_out['mu'], states[:, 1:], mask, loss_type,
                                      loss_config.get('focal_alpha'), loss_config.get('focal_gamma'))
            saps2_loss = compute_loss(world_out['saps2_delta'], saps2_delta, mask, loss_type,
                                      loss_config.get('focal_alpha'), loss_config.get('focal_gamma'))
            world_loss = state_loss + saps2_loss
            world_loss.backward()
            world_opt.step()

            total_q_loss += q_loss.item()
            total_pi_loss += pi_loss.item()
            total_perturb_loss += perturb_loss.item()
            total_world_loss += state_loss.item()
            total_saps2_loss += saps2_loss.item()
            num_batches += 1

        avg_q = total_q_loss / num_batches
        avg_pi = total_pi_loss / num_batches
        avg_p = total_perturb_loss / num_batches
        avg_w = total_world_loss / num_batches
        avg_s = total_saps2_loss / num_batches
        avg_total = avg_q + avg_pi + avg_p + avg_w + avg_s

        print(f"Epoch {epoch+1}/{epochs}: QLoss={avg_q:.4f}, PiLoss={avg_pi:.4f}, "
              f"PerturbLoss={avg_p:.4f}, World={avg_w:.4f}, SAPS2={avg_s:.4f}, "
              f"Total={avg_total:.4f}")

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'world_model_state_dict': world_model.state_dict(),
                'policy_opt_state_dict': policy_opt.state_dict(),
                'world_opt_state_dict': world_opt.state_dict(),
                'q_opt_state_dict': q_opt.state_dict(),
                'perturb_opt_state_dict': perturb_opt.state_dict(),
                'loss': avg_total,
            }, output_dir / 'best_checkpoint.pt')
            print(f"  -> Saved best checkpoint (loss={best_loss:.4f})")

    return output_dir / 'best_checkpoint.pt'


def train_cql_stage1(policy, world_model, dataloader, device, epochs, lr, output_dir, loss_config):
    """CQL: Conservative Q-Learning with twin Q-nets, target nets, and CQL penalty."""
    q_params = list(policy.q1_net.parameters()) + list(policy.q2_net.parameters())
    policy_opt = optim.Adam(policy.network.parameters(), lr=lr)
    q_opt = optim.Adam(q_params, lr=lr)
    world_opt = optim.Adam(world_model.parameters(), lr=lr)

    gamma = policy.gamma
    tau = policy.tau
    cql_alpha = policy.cql_alpha
    loss_type = loss_config.get('type', 'mse')
    best_loss = float('inf')

    print(f"  CQL params: alpha={cql_alpha}, gamma={gamma}, tau={tau}")

    for epoch in range(epochs):
        policy.train()
        world_model.train()
        total_q_loss = total_cql_penalty = total_pi_loss = 0
        total_world_loss = total_saps2_loss = num_batches = 0

        for batch in dataloader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            saps2_delta = batch['saps2_delta'].to(device)
            rewards = batch['rewards'].to(device)
            mask = batch['mask'].to(device)

            B, T, _ = states.shape
            s_t = states[:, :-1].reshape(-1, states.shape[-1])
            a_t = actions[:, :-1].reshape(-1)  # Discrete actions: (B*(T-1),)
            r_t = rewards[:, :T-1].reshape(-1)
            s_next = states[:, 1:].reshape(-1, states.shape[-1])
            a_next = actions[:, 1:].reshape(-1)  # Discrete actions: (B*(T-1),)

            # ---- CQL: Q-network update with conservative penalty ----
            q_opt.zero_grad()

            # Bellman target using target Q-networks
            with torch.no_grad():
                q1_target, q2_target = policy.get_target_q_values(s_next, a_next)
                q_target = r_t + gamma * torch.min(q1_target, q2_target).squeeze(-1)

            q1, q2 = policy.get_q_values(s_t, a_t)
            q1_loss = ((q1.squeeze(-1) - q_target) ** 2).mean()
            q2_loss = ((q2.squeeze(-1) - q_target) ** 2).mean()
            bellman_loss = q1_loss + q2_loss

            # CQL conservative penalty: logsumexp_Q(random_a) - Q(data_a)
            cql_penalty = policy.compute_cql_penalty(s_t, a_t)

            q_loss = bellman_loss + cql_penalty
            q_loss.backward()
            q_opt.step()

            # Soft update target networks
            policy.soft_update_targets()

            # ---- CQL: Policy update (BC with CrossEntropy) ----
            policy_opt.zero_grad()
            action_logits = policy(states[:, :-1])  # (B, T-1, 25)
            target_actions = actions[:, 1:].long()  # (B, T-1)
            seq_len = min(action_logits.shape[1], target_actions.shape[1])

            # BC loss with CrossEntropy
            action_logits_flat = action_logits[:, :seq_len].reshape(-1, action_logits.shape[-1])
            target_actions_flat = target_actions[:, :seq_len].reshape(-1)
            pi_loss = F.cross_entropy(action_logits_flat, target_actions_flat)
            pi_loss.backward()
            policy_opt.step()

            # ---- World Model ----
            # World Model
            world_opt.zero_grad()
            sem_wm = None
            if 'semantic' in batch:
                sem_wm = batch['semantic'][:, :-1].to(device)  # (B, T-1, sem_dim)
            world_out = world_model(states[:, :-1], actions[:, :-1], semantic_context=sem_wm)
            state_loss = compute_loss(world_out['mu'], states[:, 1:], mask, loss_type,
                                      loss_config.get('focal_alpha'), loss_config.get('focal_gamma'))
            saps2_loss = compute_loss(world_out['saps2_delta'], saps2_delta, mask, loss_type,
                                      loss_config.get('focal_alpha'), loss_config.get('focal_gamma'))
            world_loss = state_loss + saps2_loss
            world_loss.backward()
            world_opt.step()

            total_q_loss += bellman_loss.item()
            total_cql_penalty += cql_penalty.item()
            total_pi_loss += pi_loss.item()
            total_world_loss += state_loss.item()
            total_saps2_loss += saps2_loss.item()
            num_batches += 1

        avg_q = total_q_loss / num_batches
        avg_cql = total_cql_penalty / num_batches
        avg_pi = total_pi_loss / num_batches
        avg_w = total_world_loss / num_batches
        avg_s = total_saps2_loss / num_batches
        avg_total = avg_q + avg_pi + avg_w + avg_s

        print(f"Epoch {epoch+1}/{epochs}: QLoss={avg_q:.4f}, CQLPenalty={avg_cql:.4f}, "
              f"PiLoss={avg_pi:.4f}, World={avg_w:.4f}, SAPS2={avg_s:.4f}, "
              f"Total={avg_total:.4f}")

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'world_model_state_dict': world_model.state_dict(),
                'policy_opt_state_dict': policy_opt.state_dict(),
                'world_opt_state_dict': world_opt.state_dict(),
                'q_opt_state_dict': q_opt.state_dict(),
                'loss': avg_total,
            }, output_dir / 'best_checkpoint.pt')
            print(f"  -> Saved best checkpoint (loss={best_loss:.4f})")

    return output_dir / 'best_checkpoint.pt'


def _save_checkpoint(policy, world_model, policy_opt, world_opt, epoch, loss, output_dir):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'policy_state_dict': policy.state_dict(),
        'world_model_state_dict': world_model.state_dict(),
        'policy_opt_state_dict': policy_opt.state_dict(),
        'world_opt_state_dict': world_opt.state_dict(),
        'loss': loss,
    }, output_dir / 'best_checkpoint.pt')


def train_dqn_stage1(policy, world_model, dataloader, device, epochs, lr, output_dir, loss_config):
    """DQN: Q-learning with target network for discrete actions."""
    q_opt = optim.Adam(policy.network.parameters(), lr=lr)
    world_opt = optim.Adam(world_model.parameters(), lr=lr)

    gamma = policy.gamma
    tau = policy.tau
    loss_type = loss_config.get('type', 'mse')
    best_loss = float('inf')

    print(f"  DQN params: gamma={gamma}, tau={tau}")

    for epoch in range(epochs):
        policy.train()
        world_model.train()
        total_q_loss = total_world_loss = total_saps2_loss = num_batches = 0

        for batch in dataloader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            saps2_delta = batch['saps2_delta'].to(device)
            rewards = batch['rewards'].to(device)
            mask = batch['mask'].to(device)

            B, T, _ = states.shape
            s_t = states[:, :-1].reshape(-1, states.shape[-1])
            a_t = actions[:, :-1].reshape(-1)
            r_t = rewards[:, :T-1].reshape(-1)
            s_next = states[:, 1:].reshape(-1, states.shape[-1])

            # ---- DQN: Q-network update ----
            q_opt.zero_grad()

            # Current Q values: Q(s_t, a_t) for the taken action
            q_values = policy.get_q_values(s_t)  # (N, action_dim)
            # Convert discrete actions to one-hot for indexing
            a_t_onehot = F.one_hot(a_t.long(), num_classes=policy.action_dim).float()
            q_pred = (q_values * a_t_onehot).sum(dim=-1)  # Q for taken action

            # Target: r + gamma * max_a Q_target(s', a)
            with torch.no_grad():
                target_q = policy.get_target_q_values(s_next)  # (N, action_dim)
                max_target_q = target_q.max(dim=-1)[0]
                q_target = r_t + gamma * max_target_q

            q_loss = ((q_pred - q_target) ** 2).mean()
            q_loss.backward()
            q_opt.step()

            # Soft update target network
            policy.soft_update_target()

            # ---- World Model ----
            # World Model
            world_opt.zero_grad()
            sem_wm = None
            if 'semantic' in batch:
                sem_wm = batch['semantic'][:, :-1].to(device)  # (B, T-1, sem_dim)
            world_out = world_model(states[:, :-1], actions[:, :-1], semantic_context=sem_wm)
            state_loss = compute_loss(world_out['mu'], states[:, 1:], mask, loss_type,
                                      loss_config.get('focal_alpha'), loss_config.get('focal_gamma'))
            saps2_loss = compute_loss(world_out['saps2_delta'], saps2_delta, mask, loss_type,
                                      loss_config.get('focal_alpha'), loss_config.get('focal_gamma'))
            world_loss = state_loss + saps2_loss
            world_loss.backward()
            world_opt.step()

            total_q_loss += q_loss.item()
            total_world_loss += state_loss.item()
            total_saps2_loss += saps2_loss.item()
            num_batches += 1

        avg_q = total_q_loss / num_batches
        avg_w = total_world_loss / num_batches
        avg_s = total_saps2_loss / num_batches
        avg_total = avg_q + avg_w + avg_s

        print(f"Epoch {epoch+1}/{epochs}: QLoss={avg_q:.4f}, World={avg_w:.4f}, "
              f"SAPS2={avg_s:.4f}, Total={avg_total:.4f}")

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'world_model_state_dict': world_model.state_dict(),
                'policy_opt_state_dict': q_opt.state_dict(),
                'world_opt_state_dict': world_opt.state_dict(),
                'loss': avg_total,
            }, output_dir / 'best_checkpoint.pt')
            print(f"  -> Saved best checkpoint (loss={best_loss:.4f})")

    return output_dir / 'best_checkpoint.pt'


def train_td3bc_stage1(policy, world_model, dataloader, device, epochs, lr, output_dir, loss_config):
    """TD3+BC: Twin Q-networks + delayed policy update + BC regularization."""
    actor_opt = optim.Adam(policy.network.parameters(), lr=lr)
    critic_opt = optim.Adam(
        list(policy.q1_net.parameters()) + list(policy.q2_net.parameters()), lr=lr)
    world_opt = optim.Adam(world_model.parameters(), lr=lr)

    gamma = policy.gamma
    tau = policy.tau
    alpha = policy.alpha
    policy_delay = policy.policy_delay
    loss_type = loss_config.get('type', 'mse')
    best_loss = float('inf')

    print(f"  TD3+BC params: gamma={gamma}, tau={tau}, alpha={alpha}, policy_delay={policy_delay}")

    for epoch in range(epochs):
        policy.train()
        world_model.train()
        total_q_loss = total_actor_loss = total_world_loss = total_saps2_loss = num_batches = 0

        for batch in dataloader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            saps2_delta = batch['saps2_delta'].to(device)
            rewards = batch['rewards'].to(device)
            mask = batch['mask'].to(device)

            B, T, _ = states.shape
            s_t = states[:, :-1].reshape(-1, states.shape[-1])
            a_t = actions[:, :-1].reshape(-1)
            r_t = rewards[:, :T-1].reshape(-1)
            s_next = states[:, 1:].reshape(-1, states.shape[-1])

            # ---- TD3+BC: Critic update ----
            critic_opt.zero_grad()

            with torch.no_grad():
                # Target action with smoothing noise
                next_action = torch.softmax(policy.target_policy(s_next), dim=-1)
                noise = (torch.randn_like(next_action) * policy.policy_noise).clamp(
                    -policy.noise_clip, policy.noise_clip)
                next_action = (next_action + noise).clamp(0, 1)

                target_q = policy.get_target_q(s_next, next_action).squeeze(-1)
                q_target = r_t + gamma * target_q

            q1, q2 = policy.get_q_values(s_t, a_t)
            q1_loss = ((q1.squeeze(-1) - q_target) ** 2).mean()
            q2_loss = ((q2.squeeze(-1) - q_target) ** 2).mean()
            critic_loss = q1_loss + q2_loss
            critic_loss.backward()
            critic_opt.step()

            # ---- TD3+BC: Delayed Actor update ----
            if num_batches % policy_delay == 0:
                actor_opt.zero_grad()
                # Actor loss = alpha * BC_loss - Q(s, pi(s))
                pi_action = torch.softmax(policy.network(s_t), dim=-1)
                q1_pi = policy.q1_net(torch.cat([s_t, pi_action], dim=-1))
                actor_loss = -q1_pi.mean()

                # BC regularization - convert discrete actions to one-hot
                a_t_onehot = F.one_hot(a_t.long(), num_classes=policy.action_dim).float()
                bc_loss = ((pi_action - a_t_onehot) ** 2).mean()
                with torch.no_grad():
                    q_val = q1_pi.abs().mean().clamp(min=1e-8)
                actor_loss = alpha / q_val * bc_loss + actor_loss

                actor_loss.backward()
                actor_opt.step()

                # Soft update all targets
                policy.soft_update_targets()

            total_q_loss += critic_loss.item()
            total_actor_loss += actor_loss.item() if num_batches % policy_delay == 0 else 0

            # ---- World Model ----
            # World Model
            world_opt.zero_grad()
            sem_wm = None
            if 'semantic' in batch:
                sem_wm = batch['semantic'][:, :-1].to(device)  # (B, T-1, sem_dim)
            world_out = world_model(states[:, :-1], actions[:, :-1], semantic_context=sem_wm)
            state_loss = compute_loss(world_out['mu'], states[:, 1:], mask, loss_type,
                                      loss_config.get('focal_alpha'), loss_config.get('focal_gamma'))
            saps2_loss = compute_loss(world_out['saps2_delta'], saps2_delta, mask, loss_type,
                                      loss_config.get('focal_alpha'), loss_config.get('focal_gamma'))
            world_loss = state_loss + saps2_loss
            world_loss.backward()
            world_opt.step()

            total_world_loss += state_loss.item()
            total_saps2_loss += saps2_loss.item()
            num_batches += 1

        avg_q = total_q_loss / num_batches
        avg_a = total_actor_loss / (num_batches // policy_delay) if num_batches >= policy_delay else 0
        avg_w = total_world_loss / num_batches
        avg_s = total_saps2_loss / num_batches
        avg_total = avg_q + avg_a + avg_w + avg_s

        print(f"Epoch {epoch+1}/{epochs}: QLoss={avg_q:.4f}, ActorLoss={avg_a:.4f}, "
              f"World={avg_w:.4f}, SAPS2={avg_s:.4f}, Total={avg_total:.4f}")

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'world_model_state_dict': world_model.state_dict(),
                'policy_opt_state_dict': actor_opt.state_dict(),
                'world_opt_state_dict': world_opt.state_dict(),
                'critic_opt_state_dict': critic_opt.state_dict(),
                'loss': avg_total,
            }, output_dir / 'best_checkpoint.pt')
            print(f"  -> Saved best checkpoint (loss={best_loss:.4f})")

    return output_dir / 'best_checkpoint.pt'


def train_stage1(config_path, output_dir=None):
    """Train Stage 1: dispatch to algorithm-specific training."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']
    policy_type = config['model']['policy']['type']
    print(f"\n{'='*50}")
    print(f"Stage 1 Training: {model_name} (type={policy_type})")
    print(f"{'='*50}\n")

    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / model_name / "stage1"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    train_config = config['training']['stage1']
    data_path = train_config['data_path']
    print(f"Loading data from: {data_path}")

    low_oversample = train_config.get('low_oversample', 1)
    dataset = TrajectoryDataset(data_path, stratified_sampling=True, low_oversample=low_oversample)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    print(f"Dataset size: {len(dataset)} trajectories (low_oversample={low_oversample})")

    # Initialize models
    policy_config = config['model']['policy']
    world_config = config['model']['world_model']

    print(f"\nInitializing models...")
    print(f"  Policy: {policy_config['type']}, semantic: {policy_config.get('semantic_embed')}")
    print(f"  World Model: {world_config['type']}, semantic: {world_config.get('semantic_embed')}")

    policy = get_policy(policy_config).to(device)
    world_model = get_world_model(world_config).to(device)

    # Loss config
    loss_config = config['model'].get('loss', {})
    epochs = train_config['epochs']
    lr = train_config['learning_rate']

    # Dispatch to algorithm-specific training
    dispatch = {
        'BC': train_bc_stage1, 'DT': train_dt_stage1, 'IQL': train_iql_stage1,
        'BCQ': train_bcq_stage1, 'CQL': train_cql_stage1,
        'DQN': train_dqn_stage1, 'TD3BC': train_td3bc_stage1,
    }

    trainer = dispatch.get(policy_type)
    if trainer is None:
        print(f"Unknown policy type '{policy_type}', defaulting to BC training")
        trainer = train_bc_stage1

    return trainer(policy, world_model, dataloader, device, epochs, lr, output_dir, loss_config)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1 Training')
    parser.add_argument('--config', type=str, required=True, help='Path to model config')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Seed: {args.seed}")
    train_stage1(args.config, args.output_dir)
