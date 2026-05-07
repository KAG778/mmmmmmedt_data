"""
Stage 2 Training: Self-Play Fine-tuning (Fixed Version)

Uses the World Model to search for actions that improve SAPS2.
Both Policy and World Model are jointly updated (true self-play).
"""

import os
import sys
import argparse
import yaml
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.policies import get_policy, IQLPolicy, BCQPolicy, CQLPolicy, DQNPolicy, TD3BCPolicy
from models.world_models import get_world_model


def get_policy_type(policy):
    """Get a string identifier for the algorithm type."""
    classname = type(policy).__name__
    if classname in ('DTPolicy', 'MeDTPolicy', 'CSP_DTPolicy', 'SeMDTPolicy'):
        return 'DT'
    elif classname == 'IQLPolicy':
        return 'IQL'
    elif classname == 'BCQPolicy':
        return 'BCQ'
    elif classname == 'CQLPolicy':
        return 'CQL'
    elif classname == 'DQNPolicy':
        return 'DQN'
    elif classname == 'TD3BCPolicy':
        return 'TD3BC'
    elif classname == 'BCPolicy':
        return 'BC'
    return 'BC'


def load_checkpoint(checkpoint_path, policy, world_model):
    """Load checkpoint from Stage 1"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    policy.load_state_dict(checkpoint['policy_state_dict'])
    world_model.load_state_dict(checkpoint['world_model_state_dict'])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, loss={checkpoint['loss']:.4f}")
    return policy, world_model


def is_transformer_policy(policy):
    """Check if policy is transformer-based (DT, MeDT, CSP-DT, SeMDT)"""
    return get_policy_type(policy) == 'DT'


def search_better_action(policy, world_model, state, current_action, search_horizon=10,
                         n_candidates=5, is_transformer=False, semantic_context=None,
                         semantic_tensor=None, sigma_threshold=2.0, mc_samples=10):
    """
    Search for an action that improves SAPS2 using the World Model.

    Uses DISCRETE action space search (0-24) with neighborhood exploration.
    When sigma_threshold is provided, uses MC Dropout uncertainty estimation
    to filter low-confidence candidates.

    Args:
        policy: Policy model
        world_model: World Model with saps2_head
        state: Current state (1, state_dim) or (1, seq, state_dim)
        current_action: Doctor's action (1, action_dim) - one-hot encoded
        search_horizon: How many steps ahead to consider
        n_candidates: Number of action candidates to evaluate (unused, kept for compatibility)
        is_transformer: Whether policy is transformer-based
        semantic_context: Semantic context dict for policy
        semantic_tensor: Semantic tensor for world model
        sigma_threshold: Only accept candidates with sigma below this threshold
        mc_samples: Number of MC Dropout samples for uncertainty estimation

    Returns:
        best_action: The action with best predicted SAPS2 improvement (one-hot)
        advantage: Predicted improvement over doctor's action
        search_info: dict with 'confidence', 'mu_pred', 'sigma_pred', 'sigma_mean'
    """
    device = state.device

    # Ensure state is 2D (batch, state_dim)
    if state.dim() == 3:
        state_2d = state[:, -1, :]  # Take last timestep
    else:
        state_2d = state

    # Convert discrete action index to one-hot if needed
    action_dim = 25
    if current_action.dtype in (torch.long, torch.int64, torch.int32):
        current_action_idx = current_action.item()
        current_action_onehot = torch.nn.functional.one_hot(
            current_action, num_classes=action_dim).float()  # (1, 25)
    else:
        current_action_idx = torch.argmax(current_action).item()
        current_action_onehot = current_action

    # Get doctor's SAPS2 delta prediction (with uncertainty if supported)
    with torch.no_grad():
        doc_mu, doc_sigma, doc_delta = world_model.predict_with_uncertainty(
            state_2d, current_action_onehot, semantic_tensor, n_samples=mc_samples)

    # Generate action candidates in DISCRETE space
    candidates = set()

    # 1. Add neighborhood actions (±2 range)
    radius = 2
    for offset in range(-radius, radius + 1):
        idx = current_action_idx + offset
        if 0 <= idx < 25:
            candidates.add(idx)

    # 2. Add policy's suggested action
    with torch.no_grad():
        if is_transformer:
            policy_action = policy.get_action(state, actions=current_action,
                                              context=semantic_context)
        else:
            policy_action = policy.get_action(state, context=semantic_context)

        if policy_action.dim() == 3:
            policy_action = policy_action.squeeze(1)

        policy_action_idx = torch.argmax(policy_action).item()
        candidates.add(policy_action_idx)

    # Evaluate each discrete candidate action with confidence filtering
    best_delta = doc_delta.item()
    best_action_onehot = current_action_onehot
    best_sigma_mean = doc_sigma.mean().item()
    best_confidence = max(0.0, 1.0 - best_sigma_mean / sigma_threshold)
    best_mu = doc_mu
    best_sigma = doc_sigma

    for action_idx in candidates:
        # Convert discrete action to one-hot
        action_onehot = torch.zeros(1, 25, device=device)
        action_onehot[0, action_idx] = 1.0

        with torch.no_grad():
            mu, sigma, delta = world_model.predict_with_uncertainty(
                state_2d, action_onehot, semantic_tensor, n_samples=mc_samples)

        sigma_mean = sigma.mean().item()

        # Only accept high-confidence candidates that improve over doctor
        if sigma_mean < sigma_threshold and delta.item() < best_delta:
            best_delta = delta.item()
            best_action_onehot = action_onehot
            best_sigma_mean = sigma_mean
            best_confidence = max(0.0, 1.0 - sigma_mean / sigma_threshold)
            best_mu = mu
            best_sigma = sigma

    advantage = doc_delta.item() - best_delta  # Positive = improvement

    search_info = {
        'confidence': best_confidence,
        'mu_pred': best_mu,          # (1, state_dim)
        'sigma_pred': best_sigma,    # (1, state_dim)
        'sigma_mean': best_sigma_mean,
    }
    return best_action_onehot, advantage, search_info


def _expectile_loss(pred, target, expectile=0.8):
    """Asymmetric L2 loss for IQL value function."""
    diff = target - pred
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * (diff ** 2)).mean()


def train_stage2(config_path, checkpoint_path, output_dir=None, epochs_override=None):
    """Train Stage 2: Self-Play with World Model (both Policy and World Model updated)"""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']
    print(f"\n{'='*50}")
    print(f"Stage 2 Training: {model_name}")
    print(f"{'='*50}\n")

    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / model_name / "stage2"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    train_config = config['training']['stage2']
    data_path = config['training']['stage1']['data_path']
    print(f"Loading data from: {data_path}")

    with open(data_path, 'rb') as f:
        trajectories = pickle.load(f)

    # Normalize states (must match Stage1 training)
    norm_path = os.path.join(os.path.dirname(data_path), 'normalization_params.pkl')
    if os.path.exists(norm_path):
        with open(norm_path, 'rb') as f:
            norm_params = pickle.load(f)
        state_mean = norm_params['state_mean']
        state_std = norm_params['state_std'] + 1e-8
        print(f"Loaded normalization params from {norm_path}")
        for traj in trajectories:
            traj['dem_observations'] = (traj['dem_observations'] - state_mean) / state_std
        print(f"Normalized {len(trajectories)} trajectories")
    else:
        print(f"Warning: normalization params not found at {norm_path}, using raw states")

    # Stratified sampling: oversample low-risk trajectories
    low_indices = []
    mid_indices = []
    high_indices = []

    for i, traj in enumerate(trajectories):
        init_saps2 = float(traj['acuities'][0, 2])
        if init_saps2 < 50:
            low_indices.append(i)
        elif init_saps2 < 57:
            mid_indices.append(i)
        else:
            high_indices.append(i)

    print(f"Loaded {len(trajectories)} trajectories")
    print(f"Stratified sampling (NO oversampling):")
    print(f"  Low (<50):   {len(low_indices):5d}")
    print(f"  Mid (50-57): {len(mid_indices):5d}")
    print(f"  High (≥57):  {len(high_indices):5d}")

    # Create stratified sampling indices
    low_oversample = 1  # No oversampling (consistent with other experiments)
    stratified_indices = low_indices * low_oversample + mid_indices + high_indices
    print(f"Effective training size: {len(stratified_indices)} trajectories")

    # Initialize models
    policy_config = config['model']['policy']
    world_config = config['model']['world_model']

    print(f"\nInitializing models...")
    policy = get_policy(policy_config).to(device)
    world_model = get_world_model(world_config).to(device)

    # Check if policy is transformer-based
    is_transformer = is_transformer_policy(policy)
    print(f"Policy type: {'Transformer' if is_transformer else 'MLP'}")

    # Load Stage 1 checkpoint
    print(f"\nLoading Stage 1 checkpoint from: {checkpoint_path}")
    policy, world_model = load_checkpoint(checkpoint_path, policy, world_model)

    # Optimizers (lower learning rate for fine-tuning)
    policy_opt = optim.Adam(policy.parameters(), lr=train_config['learning_rate'])
    world_opt = optim.Adam(world_model.parameters(), lr=train_config['learning_rate'] * 0.5)

    # Training parameters
    epochs = epochs_override if epochs_override is not None else train_config['epochs']
    selfplay_iterations = train_config.get('selfplay_iterations', 1000)
    search_horizon = train_config.get('search_horizon', 10)
    lambda_O = train_config.get('lambda_O', 0.5)  # World model loss weight

    print(f"\nTraining for {epochs} epochs, {selfplay_iterations} iterations per epoch...")
    print(f"World Model will be UPDATED during training (lambda_O={lambda_O})")

    best_advantage = 0

    for epoch in range(epochs):
        policy.train()
        world_model.train()

        total_advantage = 0
        total_improvements = 0
        total_policy_loss = 0
        total_world_loss = 0
        total_steps = 0

        # Self-play iterations
        for iteration in range(selfplay_iterations):
            # Sample random trajectory (with stratified sampling)
            sample_idx = np.random.randint(len(stratified_indices))
            traj_idx = stratified_indices[sample_idx]
            traj = trajectories[traj_idx]

            states = torch.FloatTensor(traj['dem_observations']).to(device)
            # Keep actions as discrete indices
            actions_discrete = traj['actions']
            actions = torch.LongTensor(actions_discrete).to(device)

            # Sample random position in trajectory
            t = np.random.randint(0, len(states) - 1)

            state = states[t:t+1]
            next_state = states[t+1:t+2]
            doctor_action = actions[t:t+1]

            # Search for better action
            better_action, advantage, search_info = search_better_action(
                policy, world_model, state, doctor_action,
                search_horizon=search_horizon,
                n_candidates=5,
                is_transformer=is_transformer
            )

            # If improvement found, train on this experience
            if advantage > 0:
                policy_opt.zero_grad()
                world_opt.zero_grad()

                algo = get_policy_type(policy)

                # Policy loss: learn to produce the better action
                if is_transformer:
                    policy_pred = policy(state, actions=doctor_action)
                    if policy_pred.dim() == 3:
                        policy_pred = policy_pred[:, -1, :]
                    target_action = better_action
                else:
                    policy_pred = policy(state)
                    target_action = better_action

                if policy_pred.dim() == 3:
                    policy_pred = policy_pred.squeeze(1)
                if target_action.dim() == 3:
                    target_action = target_action.squeeze(1)

                policy_loss = ((policy_pred - target_action.detach()) ** 2).mean()

                # Algorithm-specific extra losses
                extra_loss = torch.tensor(0.0, device=device)
                state_2d = state if state.dim() == 2 else state[:, -1, :]

                if algo == 'IQL':
                    # IQL: update value function with expectile regression
                    with torch.no_grad():
                        next_state_2d = next_state if next_state.dim() == 2 else next_state.squeeze(0)
                        v_next = policy.get_value(next_state_2d).squeeze(-1)
                        # reward = -saps2_delta (advantage from search as proxy)
                        q_target = advantage + 0.99 * v_next
                    v_pred = policy.get_value(state_2d).squeeze(-1)
                    v_loss = _expectile_loss(v_pred, q_target)
                    extra_loss = extra_loss + v_loss

                elif algo == 'BCQ':
                    # BCQ: update Q-networks and perturbation
                    q1, q2 = policy.get_q_values(state_2d, better_action)
                    q_loss = (q1 ** 2).mean() + (q2 ** 2).mean()  # Q-regularization
                    extra_loss = extra_loss + 0.1 * q_loss

                elif algo == 'CQL':
                    # CQL: update Q-networks with conservative penalty
                    cql_penalty = policy.compute_cql_penalty(state_2d, better_action)
                    extra_loss = extra_loss + cql_penalty
                    policy.soft_update_targets()

                elif algo == 'DQN':
                    # DQN: Q-learning update with target network
                    with torch.no_grad():
                        next_state_2d = next_state if next_state.dim() == 2 else next_state.squeeze(0)
                        target_q = policy.get_target_q_values(next_state_2d)
                        max_target_q = target_q.max(dim=-1)[0]
                        # advantage as reward proxy
                        q_target = advantage + 0.99 * max_target_q
                    q_values = policy.get_q_values(state_2d)
                    q_pred = (q_values * better_action).sum(dim=-1)
                    extra_loss = extra_loss + ((q_pred - q_target) ** 2).mean()
                    policy.soft_update_target()

                elif algo == 'TD3BC':
                    # TD3+BC: critic + actor update
                    with torch.no_grad():
                        next_state_2d = next_state if next_state.dim() == 2 else next_state.squeeze(0)
                        next_action = torch.softmax(policy.target_policy(next_state_2d), dim=-1)
                        target_q = policy.get_target_q(next_state_2d, next_action).squeeze(-1)
                        q_target = advantage + 0.99 * target_q
                    q1, q2 = policy.get_q_values(state_2d, better_action)
                    critic_loss = ((q1.squeeze(-1) - q_target) ** 2).mean() + \
                                  ((q2.squeeze(-1) - q_target) ** 2).mean()
                    extra_loss = extra_loss + critic_loss
                    policy.soft_update_targets()

                # World Model loss: predict next state given the better action
                # Both state and action should be 2D for world model
                state_2d = state if state.dim() == 2 else state[:, -1, :]
                action_2d = better_action if better_action.dim() == 2 else better_action.squeeze(0)

                world_out = world_model(state_2d, action_2d)

                # Next state prediction loss
                if 'mu' in world_out:
                    next_state_pred = world_out['mu']
                    # Handle different shapes
                    if next_state_pred.dim() == 3:
                        next_state_pred = next_state_pred.squeeze(1)

                    # Ensure next_state has same shape
                    next_state_2d = next_state if next_state.dim() == 2 else next_state.squeeze(0)

                    world_loss = ((next_state_pred - next_state_2d) ** 2).mean()
                else:
                    world_loss = torch.tensor(0.0, device=device)

                # Combined loss (joint update of Policy and World Model)
                total_loss = policy_loss + lambda_O * world_loss + extra_loss

                total_loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)

                policy_opt.step()
                world_opt.step()

                total_policy_loss += policy_loss.item()
                total_world_loss += world_loss.item()
                total_improvements += 1

            total_advantage += advantage if isinstance(advantage, float) else advantage.item()
            total_steps += 1

        # Print epoch stats
        avg_advantage = total_advantage / total_steps
        improvement_rate = total_improvements / total_steps
        avg_policy_loss = total_policy_loss / max(total_improvements, 1)
        avg_world_loss = total_world_loss / max(total_improvements, 1)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Avg Advantage={avg_advantage:.4f}, "
              f"Improvement Rate={improvement_rate:.2%}, "
              f"Policy Loss={avg_policy_loss:.4f}, "
              f"World Loss={avg_world_loss:.4f}")

        # Save checkpoint every epoch
        is_best = avg_advantage > best_advantage
        if is_best:
            best_advantage = avg_advantage

        if is_best:
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'world_model_state_dict': world_model.state_dict(),
                'advantage': avg_advantage,
                'policy_loss': avg_policy_loss,
                'world_loss': avg_world_loss,
            }, output_dir / 'best_checkpoint.pt')
            print(f"  -> Saved checkpoint (NEW BEST advantage={best_advantage:.4f})")
        else:
            print(f"  -> advantage={avg_advantage:.4f}, best={best_advantage:.4f}")

    print(f"\n{'='*50}")
    print(f"Stage 2 Training Complete!")
    print(f"Best Advantage: {best_advantage:.4f}")
    print(f"Checkpoint saved to: {output_dir / 'best_checkpoint.pt'}")
    print(f"{'='*50}\n")

    return output_dir / 'best_checkpoint.pt'


def train_stage2_with_semantic(config_path, checkpoint_path, output_dir=None, epochs_override=None):
    """
    Stage 2 Training with Confidence-Gated Semi-Bootstrap.

    Key improvements over vanilla train_stage2:
    1. Confidence filtering: MC Dropout estimates prediction uncertainty;
       only high-confidence samples undergo semi-bootstrap.
    2. Semi-bootstrap WM loss: target = alpha * WM prediction + (1-alpha) * real next_state.
       Solves causal confusion: better_action was never actually executed.
    3. Dynamic alpha: early training relies more on real data; later training
       shifts toward bootstrap to learn unseen action outcomes.
    """

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']
    print(f"\n{'='*60}")
    print(f"Stage 2 Training (Confidence-Gated Semi-Bootstrap): {model_name}")
    print(f"{'='*60}\n")

    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / model_name / "stage2"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    train_config = config['training']['stage2']
    data_path = config['training']['stage1']['data_path']
    print(f"Loading data from: {data_path}")

    with open(data_path, 'rb') as f:
        trajectories = pickle.load(f)

    # Normalize states (must match Stage1 training)
    norm_path = os.path.join(os.path.dirname(data_path), 'normalization_params.pkl')
    if os.path.exists(norm_path):
        with open(norm_path, 'rb') as f:
            norm_params = pickle.load(f)
        state_mean = norm_params['state_mean']
        state_std = norm_params['state_std'] + 1e-8
        print(f"Loaded normalization params from {norm_path}")
        for traj in trajectories:
            traj['dem_observations'] = (traj['dem_observations'] - state_mean) / state_std
        print(f"Normalized {len(trajectories)} trajectories")
    else:
        print(f"Warning: normalization params not found at {norm_path}, using raw states")

    # --- Hyperparameters for confidence-gated semi-bootstrap ---
    sigma_threshold = train_config.get('sigma_threshold', 2.0)
    mc_samples = train_config.get('mc_samples', 10)
    alpha_min = train_config.get('alpha_min', 0.3)
    alpha_decay_steps = train_config.get('alpha_decay_steps', 50000)
    lambda_O = train_config.get('lambda_O', 0.5)
    confidence_gate = train_config.get('confidence_gate', 0.5)

    print(f"Hyperparameters:")
    print(f"  sigma_threshold   = {sigma_threshold}")
    print(f"  mc_samples        = {mc_samples}")
    print(f"  alpha_min         = {alpha_min}")
    print(f"  alpha_decay_steps = {alpha_decay_steps}")
    print(f"  confidence_gate   = {confidence_gate}")
    print(f"  lambda_O          = {lambda_O}")

    # --- Semantic embedding keys ---
    use_v6 = 'v6' in data_path.lower()
    semantic_task_key = "v6_task_embeddings" if use_v6 else "task_embeddings"
    semantic_hindsight_key = "v6_hindsight_embeddings" if use_v6 else "hindsight_embeddings"
    semantic_foresight_key = "v6_foresight_embeddings" if use_v6 else "foresight_embeddings"

    # --- Stratified sampling ---
    low_indices, mid_indices, high_indices = [], [], []
    for i, traj in enumerate(trajectories):
        init_saps2 = float(traj['acuities'][0, 2])
        if init_saps2 < 50:
            low_indices.append(i)
        elif init_saps2 < 57:
            mid_indices.append(i)
        else:
            high_indices.append(i)

    print(f"\nLoaded {len(trajectories)} trajectories")
    print(f"Stratified sampling (NO oversampling):")
    print(f"  Low (<50):   {len(low_indices):5d}")
    print(f"  Mid (50-57): {len(mid_indices):5d}")
    print(f"  High (>=57): {len(high_indices):5d}")

    stratified_indices = low_indices + mid_indices + high_indices
    print(f"Effective training size: {len(stratified_indices)} trajectories")

    # --- Initialize models ---
    policy_config = config['model']['policy']
    world_config = config['model']['world_model']

    print(f"\nInitializing models...")
    policy = get_policy(policy_config).to(device)
    world_model = get_world_model(world_config).to(device)

    is_transformer = is_transformer_policy(policy)
    print(f"Policy type: {'Transformer' if is_transformer else 'MLP'}")

    print(f"\nLoading Stage 1 checkpoint from: {checkpoint_path}")
    policy, world_model = load_checkpoint(checkpoint_path, policy, world_model)

    policy_opt = optim.Adam(policy.parameters(), lr=train_config['learning_rate'])
    world_opt = optim.Adam(world_model.parameters(), lr=train_config['learning_rate'] * 0.5)

    epochs = epochs_override if epochs_override is not None else train_config['epochs']
    selfplay_iterations = train_config.get('selfplay_iterations', 1000)

    print(f"\nTraining for {epochs} epochs, {selfplay_iterations} iterations per epoch...")
    print(f"{'Epoch':>5} | {'Adv':>8} | {'ImprRate':>8} | {'PiLoss':>8} | "
          f"{'WLoss':>8} | {'Alpha':>6} | {'Conf':>6} | {'Boot%':>6}")
    print("-" * 80)

    best_advantage = 0
    global_step = 0

    for epoch in range(epochs):
        policy.train()
        world_model.train()

        total_advantage = 0.0
        total_improvements = 0
        total_policy_loss = 0.0
        total_world_loss = 0.0
        total_steps = 0
        total_confidence = 0.0
        total_bootstrap_used = 0
        total_alpha = 0.0

        for iteration in range(selfplay_iterations):
            # Sample random trajectory (with stratified sampling)
            sample_idx = np.random.randint(len(stratified_indices))
            traj_idx = stratified_indices[sample_idx]
            traj = trajectories[traj_idx]

            states = torch.FloatTensor(traj['dem_observations']).to(device)
            # Keep actions as discrete indices
            actions_discrete = traj['actions']
            actions = torch.LongTensor(actions_discrete).to(device)

            # Sample random position in trajectory
            t = np.random.randint(0, len(states) - 1)

            state = states[t:t+1]
            next_state_real = states[t+1:t+2]
            doctor_action = actions[t:t+1]

            # --- Prepare semantic context ---
            sem_parts = []
            if semantic_task_key in traj:
                sem_parts.append(torch.FloatTensor(traj[semantic_task_key][t:t+1]).to(device))
            if semantic_hindsight_key in traj:
                sem_parts.append(torch.FloatTensor(traj[semantic_hindsight_key][t:t+1]).to(device))
            if semantic_foresight_key in traj:
                sem_parts.append(torch.FloatTensor(traj[semantic_foresight_key][t:t+1]).to(device))

            semantic_context_tensor = torch.cat(sem_parts, dim=-1) if sem_parts else None
            semantic_context_dict = {'semantic': semantic_context_tensor} if semantic_context_tensor is not None else None

            # --- Search for better action (with confidence gating) ---
            better_action, advantage, search_info = search_better_action(
                policy, world_model, state, doctor_action,
                search_horizon=train_config.get('search_horizon', 10),
                n_candidates=5,
                is_transformer=is_transformer,
                semantic_context=semantic_context_dict,
                semantic_tensor=semantic_context_tensor,
                sigma_threshold=sigma_threshold,
                mc_samples=mc_samples,
            )

            if advantage <= 0:
                total_advantage += advantage
                total_steps += 1
                global_step += 1
                continue

            # --- Compute confidence-gated alpha ---
            confidence = search_info['confidence']

            # Base alpha: decays from 1.0 to alpha_min over alpha_decay_steps
            base_alpha = max(alpha_min, 1.0 - global_step / alpha_decay_steps)

            # Confidence gating: only use bootstrap when confidence >= gate
            if confidence >= confidence_gate:
                alpha = base_alpha * confidence
                use_bootstrap = True
            else:
                alpha = 0.0
                use_bootstrap = False

            # --- Policy loss ---
            policy_opt.zero_grad()
            world_opt.zero_grad()

            algo = get_policy_type(policy)

            if is_transformer:
                policy_pred = policy(state, actions=doctor_action,
                                     context=semantic_context_dict)
                if policy_pred.dim() == 3:
                    policy_pred = policy_pred[:, -1, :]
            else:
                policy_pred = policy(state, context=semantic_context_dict)

            target_action = better_action
            if policy_pred.dim() == 3:
                policy_pred = policy_pred.squeeze(1)
            if target_action.dim() == 3:
                target_action = target_action.squeeze(1)

            policy_loss = ((policy_pred - target_action.detach()) ** 2).mean()

            # Algorithm-specific extra losses
            extra_loss = torch.tensor(0.0, device=device)
            state_2d_alg = state if state.dim() == 2 else state[:, -1, :]

            if algo == 'IQL':
                with torch.no_grad():
                    next_state_2d = next_state_real if next_state_real.dim() == 2 else next_state_real.squeeze(0)
                    v_next = policy.get_value(next_state_2d, context=semantic_context_dict).squeeze(-1)
                    q_target = advantage + 0.99 * v_next
                v_pred = policy.get_value(state_2d_alg, context=semantic_context_dict).squeeze(-1)
                extra_loss = extra_loss + _expectile_loss(v_pred, q_target)

            elif algo == 'BCQ':
                q1, q2 = policy.get_q_values(state_2d_alg, better_action, context=semantic_context_dict)
                extra_loss = extra_loss + 0.1 * ((q1 ** 2).mean() + (q2 ** 2).mean())

            elif algo == 'CQL':
                cql_penalty = policy.compute_cql_penalty(state_2d_alg, better_action)
                extra_loss = extra_loss + cql_penalty
                policy.soft_update_targets()

            elif algo == 'DQN':
                with torch.no_grad():
                    ns2d = next_state_real if next_state_real.dim() == 2 else next_state_real.squeeze(0)
                    target_q = policy.get_target_q_values(ns2d, context=semantic_context_dict)
                    max_target_q = target_q.max(dim=-1)[0]
                    q_target = advantage + 0.99 * max_target_q
                q_values = policy.get_q_values(state_2d_alg, context=semantic_context_dict)
                q_pred = (q_values * better_action).sum(dim=-1)
                extra_loss = extra_loss + ((q_pred - q_target) ** 2).mean()
                policy.soft_update_target()

            elif algo == 'TD3BC':
                with torch.no_grad():
                    ns2d = next_state_real if next_state_real.dim() == 2 else next_state_real.squeeze(0)
                    next_action = torch.softmax(policy.target_policy(
                        policy._get_input(ns2d, semantic_context_dict)), dim=-1)
                    target_q = policy.get_target_q(ns2d, next_action, context=semantic_context_dict).squeeze(-1)
                    q_target = advantage + 0.99 * target_q
                q1, q2 = policy.get_q_values(state_2d_alg, better_action, context=semantic_context_dict)
                critic_loss = ((q1.squeeze(-1) - q_target) ** 2).mean() + \
                              ((q2.squeeze(-1) - q_target) ** 2).mean()
                extra_loss = extra_loss + critic_loss
                policy.soft_update_targets()

            # --- World Model loss: semi-bootstrap ---
            state_2d = state if state.dim() == 2 else state[:, -1, :]
            action_2d = better_action if better_action.dim() == 2 else better_action.squeeze(0)
            next_state_2d = next_state_real if next_state_real.dim() == 2 else next_state_real.squeeze(0)

            world_out = world_model(state_2d, action_2d, semantic_context=semantic_context_tensor)

            if 'mu' in world_out:
                mu_pred = world_out['mu']
                if mu_pred.dim() == 3:
                    mu_pred = mu_pred.squeeze(1)

                # Real data loss (always computed, acts as regularizer)
                real_loss = ((mu_pred - next_state_2d) ** 2).mean()

                if use_bootstrap:
                    # Bootstrap target: use the MC Dropout mean from search phase
                    mu_target = search_info['mu_pred'].detach()
                    if mu_target.dim() == 1:
                        mu_target = mu_target.unsqueeze(0)
                    bootstrap_loss = ((mu_pred - mu_target) ** 2).mean()

                    world_loss = alpha * bootstrap_loss + (1 - alpha) * real_loss
                    total_bootstrap_used += 1
                else:
                    # Insufficient confidence: use purely real data
                    world_loss = real_loss
            else:
                world_loss = torch.tensor(0.0, device=device)

            # Combined loss (joint update of Policy and World Model)
            total_loss = policy_loss + lambda_O * world_loss + extra_loss

            total_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)

            policy_opt.step()
            world_opt.step()

            total_policy_loss += policy_loss.item()
            total_world_loss += world_loss.item()
            total_confidence += confidence
            total_alpha += alpha
            total_improvements += 1
            total_advantage += advantage
            total_steps += 1
            global_step += 1

        # --- Epoch stats ---
        avg_advantage = total_advantage / total_steps
        improvement_rate = total_improvements / total_steps
        avg_policy_loss = total_policy_loss / max(total_improvements, 1)
        avg_world_loss = total_world_loss / max(total_improvements, 1)
        avg_confidence = total_confidence / max(total_improvements, 1)
        avg_alpha = total_alpha / max(total_improvements, 1)
        bootstrap_pct = total_bootstrap_used / max(total_improvements, 1)

        print(f"{epoch+1:5d} | {avg_advantage:8.4f} | {improvement_rate:7.1%} | "
              f"{avg_policy_loss:8.4f} | {avg_world_loss:8.4f} | "
              f"{avg_alpha:6.3f} | {avg_confidence:6.3f} | {bootstrap_pct:5.1%}")

        # Save best checkpoint
        is_best = avg_advantage > best_advantage
        if is_best:
            best_advantage = avg_advantage

        if is_best:
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'world_model_state_dict': world_model.state_dict(),
                'advantage': avg_advantage,
                'policy_loss': avg_policy_loss,
                'world_loss': avg_world_loss,
                'avg_confidence': avg_confidence,
                'avg_alpha': avg_alpha,
                'bootstrap_pct': bootstrap_pct,
            }, output_dir / 'best_checkpoint.pt')
            print(f"  -> Saved checkpoint (NEW BEST advantage={best_advantage:.4f})")

    print(f"\n{'='*60}")
    print(f"Stage 2 Training Complete! (Confidence-Gated Semi-Bootstrap)")
    print(f"Best Advantage: {best_advantage:.4f}")
    print(f"Checkpoint saved to: {output_dir / 'best_checkpoint.pt'}")
    print(f"{'='*60}\n")

    return output_dir / 'best_checkpoint.pt'


if __name__ == '__main__':
    import random
    parser = argparse.ArgumentParser(description='Stage 2 Training')
    parser.add_argument('--config', type=str, required=True, help='Path to model config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to Stage 1 checkpoint')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--semantic', action='store_true', help='Use confidence-gated semi-bootstrap training')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=0,
                        help='Override epoch count from config (0=use config)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    epochs_ov = args.epochs if args.epochs > 0 else None
    if args.semantic:
        train_stage2_with_semantic(args.config, args.checkpoint, args.output_dir,
                                   epochs_override=epochs_ov)
    else:
        train_stage2(args.config, args.checkpoint, args.output_dir,
                     epochs_override=epochs_ov)
