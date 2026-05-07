"""
Policy Model Implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BasePolicy, SemanticEmbedding


class MLP(nn.Module):
    """Simple MLP"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalSelfAttention(nn.Module):
    """Causal self-attention"""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(y)

        return y


class Block(nn.Module):
    """Transformer block"""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT-style Transformer with optional RTG conditioning for Decision Transformer."""

    def __init__(self, state_dim, action_dim, n_layer, n_head, n_embd,
                 context_length, dropout=0.1, use_rtg=True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.context_length = context_length
        self.use_rtg = use_rtg

        # Input embeddings
        self.state_embed = nn.Linear(state_dim, n_embd)
        # Use Embedding for discrete actions instead of Linear
        self.action_embed = nn.Embedding(action_dim, n_embd)
        if use_rtg:
            self.rtg_embed = nn.Linear(1, n_embd)
            # Sequence: [RTG_1, s_1, a_1, RTG_2, s_2, a_2, ...] = 3*T tokens
            seq_len = context_length * 3
        else:
            self.rtg_embed = None
            # Sequence: [s_1, a_1, s_2, a_2, ...] = 2*T tokens
            seq_len = context_length * 2
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, n_embd))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, seq_len, dropout)
            for _ in range(n_layer)
        ])

        # Output head
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, action_dim)

        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, states, actions=None, rtg=None):
        """
        states: (B, T, state_dim) or (B, state_dim)
        actions: (B, T, action_dim) or None
        rtg: (B, T) or None - returns-to-go for DT conditioning
        """
        if states.dim() == 2:
            states = states.unsqueeze(1)

        B, T, _ = states.shape

        state_emb = self.state_embed(states)

        if self.use_rtg and rtg is not None:
            if rtg.dim() == 2:
                rtg = rtg.unsqueeze(-1)  # (B, T) -> (B, T, 1)
            rtg_emb = self.rtg_embed(rtg)  # (B, T, n_embd)

            if actions is not None:
                # Actions are discrete indices (B, T), embedding expects (B, T)
                if actions.dim() == 1:
                    actions = actions.unsqueeze(0)  # (T,) -> (B, T)
                action_emb = self.action_embed(actions)  # (B, T, n_embd)

                # Interleave: [RTG_1, s_1, a_1, RTG_2, s_2, a_2, ...]
                tokens = []
                for t in range(T):
                    tokens.extend([rtg_emb[:, t:t+1], state_emb[:, t:t+1]])
                    if t < actions.shape[1]:
                        tokens.append(action_emb[:, t:t+1])
                    else:
                        tokens.append(action_emb[:, -1:])
                x = torch.cat(tokens, dim=1)
            else:
                # Inference: [RTG_T, s_T]
                x = torch.cat([rtg_emb[:, -1:], state_emb[:, -1:]], dim=1)
        elif actions is not None:
            # Actions are discrete indices (B, T), embedding expects (B, T)
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)  # (T,) -> (B, T)
            action_emb = self.action_embed(actions)  # (B, T, n_embd)
            x = torch.cat([state_emb, action_emb], dim=1)
        else:
            x = state_emb

        # Add positional embeddings
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)

        # Transform
        for block in self.blocks:
            x = block(x)

        # Output
        x = self.ln_f(x)

        if self.use_rtg and rtg is not None and actions is not None:
            # Predict action at state positions (indices 1, 4, 7, ... in [rtg,s,a] sequence)
            state_positions = list(range(1, min(T, actions.shape[1]) * 3, 3))
            if state_positions:
                action_pred = self.head(x[:, state_positions])
                return action_pred
            return self.head(x[:, -1:])
        elif actions is not None:
            action_pred = self.head(x[:, ::2])
            return action_pred
        else:
            return self.head(x[:, -1:])


class BCPolicy(BasePolicy):
    """Behavior Cloning Policy"""

    def __init__(self, config):
        super().__init__(config)
        hidden_dim = config.get("hidden_dim", 256)
        num_layers = config.get("num_layers", 3)

        # Semantic embedding
        self.semantic_embed_layer = None
        if self.semantic_embed:
            self.semantic_embed_layer = SemanticEmbedding(
                self.semantic_embed, embed_dim=hidden_dim
            )

        # Input dimension
        input_dim = self.state_dim
        if self.semantic_embed_layer:
            input_dim += hidden_dim

        self.network = MLP(input_dim, hidden_dim, self.action_dim, num_layers)

    def forward(self, states, actions=None, context=None):
        if context is not None and "semantic" in context and self.semantic_embed_layer is not None:
            semantic_emb = self.semantic_embed_layer(context["semantic"])
            # Broadcast semantic to all timesteps if needed
            if states.dim() == 3 and semantic_emb.dim() == 2:
                B, T = states.shape[:2]
                semantic_emb = semantic_emb.unsqueeze(1).expand(B, T, -1)
            x = torch.cat([states, semantic_emb], dim=-1)
        else:
            x = states
        return self.network(x)

    def get_action(self, states, actions=None, context=None):
        with torch.no_grad():
            return self.forward(states, actions, context)


class DTPolicy(BasePolicy):
    """Decision Transformer Policy with RTG conditioning."""

    def __init__(self, config):
        super().__init__(config)
        self.n_layer = config.get("n_layer", 4)
        self.n_head = config.get("n_head", 8)
        self.n_embd = config.get("n_embd", 128)
        self.context_length = config.get("context_length", 20)
        self.dropout = config.get("dropout", 0.1)
        self.use_rtg = config.get("use_rtg", True)

        self.semantic_embed_layer = None
        if self.semantic_embed:
            self.semantic_embed_layer = SemanticEmbedding(
                self.semantic_embed, embed_dim=self.n_embd
            )

        state_dim = self.state_dim
        if self.semantic_embed_layer:
            state_dim += self.n_embd

        self.transformer = GPT(
            state_dim=state_dim,
            action_dim=self.action_dim,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            context_length=self.context_length,
            dropout=self.dropout,
            use_rtg=self.use_rtg,
        )

    def forward(self, states, actions=None, context=None, rtg=None):
        if context is not None and "semantic" in context and self.semantic_embed_layer is not None:
            semantic_emb = self.semantic_embed_layer(context["semantic"])
            if states.dim() == 3:
                B, T = states.shape[:2]
                semantic_emb = semantic_emb.unsqueeze(1).expand(B, T, -1)
            states = torch.cat([states, semantic_emb], dim=-1)

        return self.transformer(states, actions, rtg=rtg)

    def get_action(self, states, actions=None, context=None, rtg=None):
        with torch.no_grad():
            return self.forward(states, actions, context, rtg=rtg)


class MeDTPolicy(DTPolicy):
    """Medical DT Policy (same as DT, different training)"""

    pass


class SeMDTPolicy(DTPolicy):
    """Semantic DT Policy"""

    def __init__(self, config):
        # Use semantic embed from config, don't force V6
        # config["semantic_embed"] = "V6"
        super().__init__(config)


class CSP_DTPolicy(DTPolicy):
    """CSP-DT Policy"""

    def __init__(self, config):
        super().__init__(config)


class IQLPolicy(BasePolicy):
    """IQL Policy with Value function and advantage-weighted policy.

    IQL learns a state-value function V(s) via expectile regression on
    Q(s,a) = r + gamma * V(s'), and trains the policy with advantage weighting:
    pi(a|s) proportional to exp(beta * (Q(s,a) - V(s))).
    """

    def __init__(self, config):
        super().__init__(config)
        hidden_dim = config.get("hidden_dim", 256)
        num_layers = config.get("num_layers", 3)

        self.semantic_embed_layer = None
        if self.semantic_embed:
            self.semantic_embed_layer = SemanticEmbedding(
                self.semantic_embed, embed_dim=hidden_dim
            )

        input_dim = self.state_dim
        if self.semantic_embed_layer:
            input_dim += hidden_dim

        # Policy network (same as BC)
        self.network = MLP(input_dim, hidden_dim, self.action_dim, num_layers)

        # Value function V(s)
        self.value_net = MLP(input_dim, hidden_dim, 1, num_layers)

    def forward(self, states, actions=None, context=None):
        if context is not None and "semantic" in context and self.semantic_embed_layer is not None:
            semantic_emb = self.semantic_embed_layer(context["semantic"])
            if states.dim() == 3 and semantic_emb.dim() == 2:
                B, T = states.shape[:2]
                semantic_emb = semantic_emb.unsqueeze(1).expand(B, T, -1)
            x = torch.cat([states, semantic_emb], dim=-1)
        else:
            x = states
        return self.network(x)

    def get_value(self, states, context=None):
        """Get V(s) estimate."""
        if context is not None and "semantic" in context and self.semantic_embed_layer is not None:
            semantic_emb = self.semantic_embed_layer(context["semantic"])
            if states.dim() == 3 and semantic_emb.dim() == 2:
                B, T = states.shape[:2]
                semantic_emb = semantic_emb.unsqueeze(1).expand(B, T, -1)
            x = torch.cat([states, semantic_emb], dim=-1)
        else:
            x = states
        return self.value_net(x)

    def get_action(self, states, actions=None, context=None):
        with torch.no_grad():
            return self.forward(states, actions, context)


class BCQPolicy(BasePolicy):
    """BCQ Policy with twin Q-networks and perturbation network.

    BCQ uses: 1) A generative model (BC policy) for candidate actions,
    2) Twin Q-networks to evaluate actions, 3) A perturbation network
    to refine actions. Final action = argmax_a { min(Q1, Q2)(s, a + perturb(s,a)) }.
    """

    def __init__(self, config):
        super().__init__(config)
        hidden_dim = config.get("hidden_dim", 256)
        num_layers = config.get("num_layers", 3)
        self.n_candidates = config.get("n_candidates", 10)
        self.perturb_clip = config.get("perturb_clip", 0.05)

        self.semantic_embed_layer = None
        if self.semantic_embed:
            self.semantic_embed_layer = SemanticEmbedding(
                self.semantic_embed, embed_dim=hidden_dim
            )

        input_dim = self.state_dim
        if self.semantic_embed_layer:
            input_dim += hidden_dim

        # BC policy (behavior cloning backbone)
        self.network = MLP(input_dim, hidden_dim, self.action_dim, num_layers)

        # Twin Q-networks: Q(s, a)
        q_input_dim = self.state_dim + self.action_dim
        if self.semantic_embed_layer:
            q_input_dim += hidden_dim
        self.q1_net = MLP(q_input_dim, hidden_dim, 1, num_layers)
        self.q2_net = MLP(q_input_dim, hidden_dim, 1, num_layers)

        # Perturbation network: outputs delta_a given (s, a)
        self.perturb_net = MLP(q_input_dim, hidden_dim, self.action_dim, num_layers)

    def _get_q_input(self, states, actions, context=None):
        """Concatenate state+action (and optionally semantic) for Q-networks."""
        # Convert discrete actions to one-hot encoding for Q-network input
        if actions.dtype == torch.long:
            actions_onehot = F.one_hot(actions, num_classes=self.action_dim).float()
        else:
            actions_onehot = actions

        if context is not None and "semantic" in context and self.semantic_embed_layer is not None:
            semantic_emb = self.semantic_embed_layer(context["semantic"])
            if states.dim() == 3 and semantic_emb.dim() == 2:
                B, T = states.shape[:2]
                semantic_emb = semantic_emb.unsqueeze(1).expand(B, T, -1)
            return torch.cat([states, actions_onehot, semantic_emb], dim=-1)
        return torch.cat([states, actions_onehot], dim=-1)

    def forward(self, states, actions=None, context=None):
        if context is not None and "semantic" in context and self.semantic_embed_layer is not None:
            semantic_emb = self.semantic_embed_layer(context["semantic"])
            if states.dim() == 3 and semantic_emb.dim() == 2:
                B, T = states.shape[:2]
                semantic_emb = semantic_emb.unsqueeze(1).expand(B, T, -1)
            x = torch.cat([states, semantic_emb], dim=-1)
        else:
            x = states
        return self.network(x)

    def get_q_values(self, states, actions, context=None):
        """Get twin Q-values for (s, a) pairs."""
        q_input = self._get_q_input(states, actions, context)
        return self.q1_net(q_input), self.q2_net(q_input)

    def perturb_action(self, states, actions, context=None):
        """Apply perturbation to actions, clipped to small range.
        For discrete actions, convert to one-hot first."""
        # Convert discrete actions to one-hot for perturbation
        if actions.dtype == torch.long:
            actions_onehot = F.one_hot(actions, num_classes=self.action_dim).float()
        else:
            actions_onehot = actions

        q_input = self._get_q_input(states, actions, context)
        delta = self.perturb_net(q_input)
        delta = delta.clamp(-self.perturb_clip, self.perturb_clip)
        # Return perturbed one-hot representation
        return actions_onehot + delta

    def get_action(self, states, actions=None, context=None):
        """Get action via BCQ: generate candidates, perturb, pick best Q."""
        with torch.no_grad():
            # BC suggestion
            bc_action = self.forward(states, actions, context)
            if bc_action.dim() == 3:
                bc_action = bc_action.squeeze(1)

            # Generate candidate actions around BC suggestion
            if states.dim() == 3:
                state_2d = states[:, -1, :]
            else:
                state_2d = states

            best_action = bc_action
            best_q = float('-inf')

            for _ in range(self.n_candidates):
                noise = torch.randn_like(bc_action) * 0.1
                candidate = (bc_action + noise).softmax(dim=-1)
                perturbed = self.perturb_action(state_2d, candidate, context)
                q1, q2 = self.get_q_values(state_2d, perturbed, context)
                q = torch.min(q1, q2).squeeze(-1)
                if q.item() > best_q:
                    best_q = q.item()
                    best_action = perturbed

            return best_action


class CQLPolicy(BasePolicy):
    """CQL Policy with twin Q-networks and target networks.

    CQL adds a conservative penalty to standard Q-learning:
    L_CQL = alpha * (logsumexp_Q(s, a_random) - Q(s, a_data))
    This penalizes Q-values for OOD actions, making the policy conservative.
    """

    def __init__(self, config):
        super().__init__(config)
        hidden_dim = config.get("hidden_dim", 256)
        num_layers = config.get("num_layers", 3)
        self.cql_alpha = config.get("cql_alpha", 1.0)
        self.cql_n_random_actions = config.get("cql_n_random_actions", 10)
        self.tau = config.get("tau", 0.005)  # target network soft update rate
        self.gamma = config.get("gamma", 0.99)

        self.semantic_embed_layer = None
        if self.semantic_embed:
            self.semantic_embed_layer = SemanticEmbedding(
                self.semantic_embed, embed_dim=hidden_dim
            )

        input_dim = self.state_dim
        if self.semantic_embed_layer:
            input_dim += hidden_dim

        # Policy network
        self.network = MLP(input_dim, hidden_dim, self.action_dim, num_layers)

        # Twin Q-networks + target networks
        q_input_dim = self.state_dim + self.action_dim
        if self.semantic_embed_layer:
            q_input_dim += hidden_dim
        self.q1_net = MLP(q_input_dim, hidden_dim, 1, num_layers)
        self.q2_net = MLP(q_input_dim, hidden_dim, 1, num_layers)
        self.q1_target = MLP(q_input_dim, hidden_dim, 1, num_layers)
        self.q2_target = MLP(q_input_dim, hidden_dim, 1, num_layers)

        # Initialize targets to match online networks
        self.q1_target.load_state_dict(self.q1_net.state_dict())
        self.q2_target.load_state_dict(self.q2_net.state_dict())

    def _get_q_input(self, states, actions, context=None):
        # Convert discrete actions to one-hot encoding for Q-network input
        if actions.dtype == torch.long:
            # Actions are discrete indices, convert to one-hot
            actions_onehot = F.one_hot(actions, num_classes=self.action_dim).float()
            # Ensure actions_onehot matches states dimensions
            # If states is 2D (B, state_dim), actions_onehot should be 2D (B, action_dim)
            # If states is 3D (B, T, state_dim), actions_onehot should be 3D (B, T, action_dim)
        else:
            actions_onehot = actions

        if context is not None and "semantic" in context and self.semantic_embed_layer is not None:
            semantic_emb = self.semantic_embed_layer(context["semantic"])
            if states.dim() == 3 and semantic_emb.dim() == 2:
                B, T = states.shape[:2]
                semantic_emb = semantic_emb.unsqueeze(1).expand(B, T, -1)
            return torch.cat([states, actions_onehot, semantic_emb], dim=-1)
        return torch.cat([states, actions_onehot], dim=-1)

    def forward(self, states, actions=None, context=None):
        if context is not None and "semantic" in context and self.semantic_embed_layer is not None:
            semantic_emb = self.semantic_embed_layer(context["semantic"])
            if states.dim() == 3 and semantic_emb.dim() == 2:
                B, T = states.shape[:2]
                semantic_emb = semantic_emb.unsqueeze(1).expand(B, T, -1)
            x = torch.cat([states, semantic_emb], dim=-1)
        else:
            x = states
        return self.network(x)

    def get_q_values(self, states, actions, context=None):
        q_input = self._get_q_input(states, actions, context)
        return self.q1_net(q_input), self.q2_net(q_input)

    def get_target_q_values(self, states, actions, context=None):
        q_input = self._get_q_input(states, actions, context)
        return self.q1_target(q_input), self.q2_target(q_input)

    def compute_cql_penalty(self, states, actions, context=None):
        """Compute CQL conservative penalty: logsumexp_Q(random_a) - Q(data_a)."""
        q1_data, q2_data = self.get_q_values(states, actions, context)
        q_data = torch.min(q1_data, q2_data)

        # Sample random actions
        batch_size = states.shape[0]
        random_actions = torch.softmax(torch.randn(batch_size, self.action_dim, device=states.device), dim=-1)
        q1_rand, q2_rand = self.get_q_values(states, random_actions, context)
        q_rand = torch.logsumexp(torch.cat([q1_rand, q2_rand], dim=-1), dim=-1, keepdim=False)

        # CQL penalty: push up Q for random actions, push down Q for data actions
        penalty = (q_rand.mean() - q_data.mean()) * self.cql_alpha
        return penalty

    def soft_update_targets(self):
        """Soft update target networks: target = tau * online + (1-tau) * target."""
        for online, target in [(self.q1_net, self.q1_target), (self.q2_net, self.q2_target)]:
            for o_param, t_param in zip(online.parameters(), target.parameters()):
                t_param.data.copy_(self.tau * o_param.data + (1 - self.tau) * t_param.data)

    def get_action(self, states, actions=None, context=None):
        with torch.no_grad():
            return self.forward(states, actions, context)


class DQNPolicy(BasePolicy):
    """DQN Policy with Q-network and target network.

    For discrete action spaces: Q(s) outputs Q-values for all actions,
    policy selects argmax Q(s). Uses target network + replay for stability.
    """

    def __init__(self, config):
        super().__init__(config)
        hidden_dim = config.get("hidden_dim", 256)
        num_layers = config.get("num_layers", 3)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.eps_greedy = config.get("eps_greedy", 0.1)

        self.semantic_embed_layer = None
        if self.semantic_embed:
            self.semantic_embed_layer = SemanticEmbedding(
                self.semantic_embed, embed_dim=hidden_dim
            )

        input_dim = self.state_dim
        if self.semantic_embed_layer:
            input_dim += hidden_dim

        # Q-network: state -> Q-values for all actions
        self.network = MLP(input_dim, hidden_dim, self.action_dim, num_layers)

        # Target Q-network
        self.target_network = MLP(input_dim, hidden_dim, self.action_dim, num_layers)
        self.target_network.load_state_dict(self.network.state_dict())

    def forward(self, states, actions=None, context=None):
        if context is not None and "semantic" in context and self.semantic_embed_layer is not None:
            semantic_emb = self.semantic_embed_layer(context["semantic"])
            if states.dim() == 3 and semantic_emb.dim() == 2:
                B, T = states.shape[:2]
                semantic_emb = semantic_emb.unsqueeze(1).expand(B, T, -1)
            x = torch.cat([states, semantic_emb], dim=-1)
        else:
            x = states
        return self.network(x)

    def get_q_values(self, states, context=None):
        """Get Q-values for all actions given state."""
        return self.forward(states, context=context)

    def get_target_q_values(self, states, context=None):
        """Get target Q-values for all actions given state."""
        if context is not None and "semantic" in context and self.semantic_embed_layer is not None:
            semantic_emb = self.semantic_embed_layer(context["semantic"])
            if states.dim() == 3 and semantic_emb.dim() == 2:
                B, T = states.shape[:2]
                semantic_emb = semantic_emb.unsqueeze(1).expand(B, T, -1)
            x = torch.cat([states, semantic_emb], dim=-1)
        else:
            x = states
        return self.target_network(x)

    def soft_update_target(self):
        for p, tp in zip(self.network.parameters(), self.target_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def get_action(self, states, actions=None, context=None):
        with torch.no_grad():
            q_values = self.forward(states, context=context)
            if q_values.dim() == 3:
                q_values = q_values[:, -1, :]
            # Epsilon-greedy: during training use eps, during eval use greedy
            if self.training and torch.rand(1).item() < self.eps_greedy:
                action_idx = torch.randint(self.action_dim, (q_values.shape[0],), device=q_values.device)
            else:
                action_idx = q_values.argmax(dim=-1)
            # Convert to one-hot
            one_hot = torch.zeros_like(q_values)
            one_hot.scatter_(1, action_idx.unsqueeze(-1), 1.0)
            return one_hot


class TD3BCPolicy(BasePolicy):
    """TD3+BC Policy: Twin Delayed DDPG with Behavior Cloning regularization.

    Combines TD3 (twin Q-networks, delayed policy update, target smoothing)
    with BC regularization: policy_loss = alpha * BC_loss - (1-alpha) * Q(s, pi(s)).
    Designed for offline RL.
    """

    def __init__(self, config):
        super().__init__(config)
        hidden_dim = config.get("hidden_dim", 256)
        num_layers = config.get("num_layers", 3)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.policy_noise = config.get("policy_noise", 0.2)
        self.noise_clip = config.get("noise_clip", 0.5)
        self.policy_delay = config.get("policy_delay", 2)
        self.alpha = config.get("alpha", 2.5)

        self.semantic_embed_layer = None
        if self.semantic_embed:
            self.semantic_embed_layer = SemanticEmbedding(
                self.semantic_embed, embed_dim=hidden_dim
            )

        input_dim = self.state_dim
        if self.semantic_embed_layer:
            input_dim += hidden_dim

        q_input_dim = self.state_dim + self.action_dim
        if self.semantic_embed_layer:
            q_input_dim += hidden_dim

        # Policy network (actor)
        self.network = MLP(input_dim, hidden_dim, self.action_dim, num_layers)

        # Twin Q-networks (critics)
        self.q1_net = MLP(q_input_dim, hidden_dim, 1, num_layers)
        self.q2_net = MLP(q_input_dim, hidden_dim, 1, num_layers)

        # Target networks
        self.target_policy = MLP(input_dim, hidden_dim, self.action_dim, num_layers)
        self.target_q1 = MLP(q_input_dim, hidden_dim, 1, num_layers)
        self.target_q2 = MLP(q_input_dim, hidden_dim, 1, num_layers)

        # Initialize targets
        self.target_policy.load_state_dict(self.network.state_dict())
        self.target_q1.load_state_dict(self.q1_net.state_dict())
        self.target_q2.load_state_dict(self.q2_net.state_dict())

    def _get_input(self, states, context=None):
        if context is not None and "semantic" in context and self.semantic_embed_layer is not None:
            semantic_emb = self.semantic_embed_layer(context["semantic"])
            if states.dim() == 3 and semantic_emb.dim() == 2:
                B, T = states.shape[:2]
                semantic_emb = semantic_emb.unsqueeze(1).expand(B, T, -1)
            return torch.cat([states, semantic_emb], dim=-1)
        return states

    def _get_q_input(self, states, actions, context=None):
        # Convert discrete actions to one-hot encoding for Q-network input
        if actions.dtype == torch.long:
            actions_onehot = F.one_hot(actions, num_classes=self.action_dim).float()
        else:
            actions_onehot = actions

        if context is not None and "semantic" in context and self.semantic_embed_layer is not None:
            semantic_emb = self.semantic_embed_layer(context["semantic"])
            if states.dim() == 3 and semantic_emb.dim() == 2:
                B, T = states.shape[:2]
                semantic_emb = semantic_emb.unsqueeze(1).expand(B, T, -1)
            return torch.cat([states, actions_onehot, semantic_emb], dim=-1)
        return torch.cat([states, actions_onehot], dim=-1)

    def forward(self, states, actions=None, context=None):
        x = self._get_input(states, context)
        return torch.softmax(self.network(x), dim=-1)

    def get_q_values(self, states, actions, context=None):
        q_input = self._get_q_input(states, actions, context)
        return self.q1_net(q_input), self.q2_net(q_input)

    def get_target_q(self, states, actions, context=None):
        # Target policy smoothing: add clipped noise to target actions
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            smoothed_actions = (actions + noise).clamp(0, 1)

            q_input = self._get_q_input(states, smoothed_actions, context)
            q1 = self.target_q1(q_input)
            q2 = self.target_q2(q_input)
            return torch.min(q1, q2)

    def soft_update_targets(self):
        for src, tgt in [(self.network, self.target_policy),
                         (self.q1_net, self.target_q1),
                         (self.q2_net, self.target_q2)]:
            for s, t in zip(src.parameters(), tgt.parameters()):
                t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def compute_actor_loss(self, states, actions_bc, context=None):
        """TD3+BC actor loss: alpha * BC_loss - Q(s, pi(s))."""
        pi_action = self.forward(states, context=context)
        if pi_action.dim() == 3:
            pi_action = pi_action[:, -1, :]

        state_input = states
        if states.dim() == 3:
            state_input = states[:, -1, :]

        q1, _ = self.get_q_values(state_input, pi_action, context)
        q_loss = -q1.mean()

        bc_loss = ((pi_action - actions_bc) ** 2).mean()

        # Adaptive alpha: alpha / mean(|Q|)
        with torch.no_grad():
            q1_val = q1.abs().mean().clamp(min=1e-8)

        return self.alpha / q1_val * bc_loss + q_loss

    def get_action(self, states, actions=None, context=None):
        with torch.no_grad():
            return self.forward(states, actions, context)


# Policy factory
def get_policy(config):
    """Get policy by type"""
    policy_type = config.get("type", "BC")
    config_dict = {
        "state_dim": config["state_dim"],
        "action_dim": config["action_dim"],
        "semantic_embed": config.get("semantic_embed"),
        **{k: v for k, v in config.items() if k not in ["type", "state_dim", "action_dim", "semantic_embed"]}
    }

    policies = {
        "BC": BCPolicy,
        "DT": DTPolicy,
        "MeDT": MeDTPolicy,
        "SeMDT": SeMDTPolicy,
        "CSP_DT": CSP_DTPolicy,
        "IQL": IQLPolicy,
        "BCQ": BCQPolicy,
        "CQL": CQLPolicy,
        "DQN": DQNPolicy,
        "TD3BC": TD3BCPolicy,
    }

    return policies[policy_type](config_dict)
