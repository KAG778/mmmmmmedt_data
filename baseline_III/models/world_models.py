"""
World Model Implementations
"""

import torch
import torch.nn as nn
import math
from .base import BaseWorldModel, SemanticEmbedding


class MLP_World(nn.Module):
    """MLP for World Model"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.network(x)


class BC_World_Model(BaseWorldModel):
    """BC World Model - Simple MLP"""

    def __init__(self, config):
        super().__init__(config)
        hidden_dim = config.get("hidden_dim", 256)
        num_layers = config.get("num_layers", 3)
        dropout = config.get("dropout", 0.1)

        # Input dimension
        input_dim = self.state_dim + self.action_dim
        if self.semantic_embed:
            self.semantic_embed_layer = SemanticEmbedding(
                self.semantic_embed, embed_dim=hidden_dim
            )
            input_dim += hidden_dim
        else:
            self.semantic_embed_layer = None

        # Shared network
        self.network = MLP_World(input_dim, hidden_dim, hidden_dim, num_layers, dropout)

        # Output heads
        self.mu_head = nn.Linear(hidden_dim, self.state_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, self.state_dim)
        self.saps2_head = nn.Linear(hidden_dim, 1)

    def forward(self, states, actions, semantic_context=None):
        # Convert discrete action indices to one-hot if needed
        if actions.dtype in (torch.long, torch.int64, torch.int32):
            actions = torch.nn.functional.one_hot(actions, num_classes=self.action_dim).float()

        x = torch.cat([states, actions], dim=-1)

        if self.semantic_embed_layer and semantic_context is not None:
            semantic_emb = self.semantic_embed_layer(semantic_context)
            # Broadcast semantic to all timesteps
            if x.dim() == 3:
                B, T = x.shape[:2]
                if semantic_emb.dim() == 2:
                    semantic_emb = semantic_emb.unsqueeze(1).expand(B, T, -1)
                elif semantic_emb.dim() == 1:
                    semantic_emb = semantic_emb.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            x = torch.cat([x, semantic_emb], dim=-1)

        h = self.network(x)

        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)
        saps2_delta = self.saps2_head(h).squeeze(-1)

        return {
            "mu": mu,
            "log_sigma": log_sigma,
            "saps2_delta": saps2_delta
        }

    def predict_saps2_delta(self, states, actions, semantic_context=None):
        out = self.forward(states, actions, semantic_context)
        return out["saps2_delta"]


class DT_World_Model(BaseWorldModel):
    """DT World Model - Transformer"""

    def __init__(self, config):
        super().__init__(config)
        self.n_layer = config.get("n_layer", 3)
        self.n_head = config.get("n_head", 4)
        self.n_embd = config.get("n_embd", 128)
        self.dropout = config.get("dropout", 0.1)

        # Semantic embedding
        self.semantic_embed_layer = None
        if self.semantic_embed:
            self.semantic_embed_layer = SemanticEmbedding(
                self.semantic_embed, embed_dim=self.n_embd
            )

        # Input dimension
        state_dim = self.state_dim
        if self.semantic_embed_layer:
            state_dim += self.n_embd

        # Simple transformer (could be more complex)
        self.network = nn.Sequential(
            nn.Linear(state_dim + self.action_dim, self.n_embd),
            nn.LayerNorm(self.n_embd),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_embd, self.n_embd),
            nn.LayerNorm(self.n_embd),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # Output heads
        self.mu_head = nn.Linear(self.n_embd, self.state_dim)
        self.log_sigma_head = nn.Linear(self.n_embd, self.state_dim)
        self.saps2_head = nn.Linear(self.n_embd, 1)

    def forward(self, states, actions, semantic_context=None):
        # Convert discrete action indices to one-hot if needed
        if actions.dtype in (torch.long, torch.int64, torch.int32):
            actions = torch.nn.functional.one_hot(actions, num_classes=self.action_dim).float()

        x = torch.cat([states, actions], dim=-1)

        if self.semantic_embed_layer and semantic_context is not None:
            semantic_emb = self.semantic_embed_layer(semantic_context)
            # Broadcast semantic to all timesteps
            if x.dim() == 3:
                B, T = x.shape[:2]
                if semantic_emb.dim() == 2:
                    semantic_emb = semantic_emb.unsqueeze(1).expand(B, T, -1)
                elif semantic_emb.dim() == 1:
                    semantic_emb = semantic_emb.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            x = torch.cat([x, semantic_emb], dim=-1)

        h = self.network(x)

        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)
        saps2_delta = self.saps2_head(h).squeeze(-1)

        return {
            "mu": mu,
            "log_sigma": log_sigma,
            "saps2_delta": saps2_delta
        }

    def predict_saps2_delta(self, states, actions, semantic_context=None):
        out = self.forward(states, actions, semantic_context)
        return out["saps2_delta"]


class IQL_World_Model(BC_World_Model):
    """IQL World Model - Conservative prediction with uncertainty"""

    def __init__(self, config):
        super().__init__(config)
        hidden_dim = config.get("hidden_dim", 256)

        # Additional uncertainty head
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(self, states, actions, semantic_context=None):
        out = super().forward(states, actions, semantic_context)

        # Get hidden representation from BC_World_Model output
        # We need to extract the hidden representation from the network
        # For now, just return the output without uncertainty head
        # The uncertainty can be estimated using MC dropout separately

        # Note: super().forward() already handles semantic embeddings correctly
        return out


class BCQ_World_Model(BC_World_Model):
    """BCQ World Model - Generative with multiple candidates"""

    def __init__(self, config):
        super().__init__(config)
        self.n_candidates = config.get("n_candidates", 10)

    def forward(self, states, actions, semantic_context=None):
        # For BCQ, we sample multiple predictions and aggregate
        predictions = []
        for _ in range(self.n_candidates):
            # Add noise during training for diversity
            if self.training:
                # Convert actions to float for noise addition, then back to original dtype
                actions_float = actions.float()
                noisy_actions = actions_float + torch.randn_like(actions_float) * 0.1
                noisy_actions = noisy_actions.to(actions.dtype)
                pred = super().forward(states, noisy_actions, semantic_context)
            else:
                pred = super().forward(states, actions, semantic_context)
            predictions.append(pred)

        # Aggregate predictions
        mu = torch.stack([p["mu"] for p in predictions]).mean(dim=0)
        log_sigma = torch.stack([p["log_sigma"] for p in predictions]).mean(dim=0)
        saps2_delta = torch.stack([p["saps2_delta"] for p in predictions]).mean(dim=0)

        return {
            "mu": mu,
            "log_sigma": log_sigma,
            "saps2_delta": saps2_delta
        }


class CQL_World_Model(BC_World_Model):
    """CQL World Model - Same architecture as BC, CQL-specific logic is in policy/training."""

    def __init__(self, config):
        super().__init__(config)


class DQN_World_Model(BC_World_Model):
    """DQN World Model - Same architecture as BC."""

    def __init__(self, config):
        super().__init__(config)


class TD3BC_World_Model(BC_World_Model):
    """TD3+BC World Model - Same architecture as BC."""

    def __init__(self, config):
        super().__init__(config)


class CSP_DT_World_Model(BC_World_Model):
    """CSP-DT World Model - With MC Dropout support"""

    def __init__(self, config):
        super().__init__(config)

    def predict_saps2_delta_with_uncertainty(self, states, actions, semantic_context=None, n_samples=10):
        """Predict with MC Dropout uncertainty estimation"""
        self.enable_dropout()
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.predict_saps2_delta(states, actions, semantic_context)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        return mean, std


class MeDT_World_Model(DT_World_Model):
    """MeDT World Model - Same as DT"""

    pass


class SeMDT_World_Model(DT_World_Model):
    """SeMDT World Model - With V6 semantic"""

    def __init__(self, config):
        config["semantic_embed"] = "V6"
        super().__init__(config)


# World Model factory
def get_world_model(config):
    """Get world model by type"""
    model_type = config.get("type", "BC_World")
    config_dict = {
        "state_dim": config["state_dim"],
        "action_dim": config["action_dim"],
        "semantic_embed": config.get("semantic_embed"),
        **{k: v for k, v in config.items() if k not in ["type", "state_dim", "action_dim", "semantic_embed"]}
    }

    models = {
        "BC_World": BC_World_Model,
        "DT_World": DT_World_Model,
        "MeDT_World": MeDT_World_Model,
        "SeMDT_World": SeMDT_World_Model,
        "IQL_World": IQL_World_Model,
        "BCQ_World": BCQ_World_Model,
        "CQL_World": CQL_World_Model,
        "DQN_World": DQN_World_Model,
        "TD3BC_World": TD3BC_World_Model,
        "CSP_DT_World": CSP_DT_World_Model,
    }

    return models[model_type](config_dict)
