"""
World Model for scheme3_cspdt_v2 series.

Architecture matches exp4.3's BC_World_Model:
  - SemanticEmbedding: compresses raw 2688-dim semantic to hidden_dim (256)
  - 3-layer MLP with dropout
  - Output heads: mu, log_sigma, saps2_delta
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class SemanticEmbedding(nn.Module):
    """Compress semantic embeddings (task + hindsight + foresight) to embed_dim."""

    def __init__(self, input_dim=896 * 3, embed_dim=256):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x):
        return self.embedding(x)


class MLP_World(nn.Module):
    """Multi-layer MLP backbone."""

    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
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


class WorldModel(nn.Module):
    """
    O: (state, action_onehot, semantic) -> (mu, log_sigma, saps2_delta)

    Architecture mirrors exp4.3's BC_World_Model:
    - SemanticEmbedding compresses 2688 -> 256
    - 3-layer MLP backbone with MC Dropout
    - Separate output heads for mu, log_sigma, saps2_delta
    """

    def __init__(self, state_dim=45, action_dim=25, hidden_dim=256, dropout=0.2,
                 semantic_dim=896 * 3, num_layers=3, use_semantic=True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.semantic_dim = semantic_dim
        self.use_semantic = use_semantic

        # Semantic embedding: 2688 -> hidden_dim
        if use_semantic:
            self.semantic_embed_layer = SemanticEmbedding(semantic_dim, hidden_dim)
            input_dim = state_dim + action_dim + hidden_dim
        else:
            self.semantic_embed_layer = None
            input_dim = state_dim + action_dim

        # Shared MLP backbone
        self.network = MLP_World(input_dim, hidden_dim, num_layers, dropout)

        # Output heads
        self.mu_head = nn.Linear(hidden_dim, state_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, state_dim)
        self.saps2_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, action_onehot, semantic=None):
        """
        state: (B, state_dim)
        action_onehot: (B, action_dim)
        semantic: (B, semantic_dim) - optional, raw concatenated embeddings
        Returns: mu (B, state_dim), log_sigma (B, state_dim), saps2_delta (B, 1)
        """
        x = torch.cat([state, action_onehot], dim=-1)

        if self.semantic_embed_layer is not None and semantic is not None:
            sem_emb = self.semantic_embed_layer(semantic)  # (B, hidden_dim)
            # Broadcast if 3D input
            if x.dim() == 3:
                B, T = x.shape[:2]
                if sem_emb.dim() == 2:
                    sem_emb = sem_emb.unsqueeze(1).expand(B, T, -1)
                elif sem_emb.dim() == 1:
                    sem_emb = sem_emb.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            x = torch.cat([x, sem_emb], dim=-1)

        h = self.network(x)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(-4.0, 4.0)
        saps2_delta = self.saps2_head(h)
        return mu, log_sigma, saps2_delta

    def predict(self, state, action, semantic=None, mc_samples=10):
        """
        MC Dropout inference.
        state: (B, state_dim)
        action: (B,) long tensor of action indices
        semantic: (B, semantic_dim) - optional semantic context
        Returns: mu (B, state_dim), sigma (B, state_dim), saps2_delta (B, 1)
        """
        self.train()  # keep dropout active for MC sampling
        action_onehot = F.one_hot(action.long(), num_classes=self.action_dim).float()

        mus, log_sigmas, saps2_deltas = [], [], []
        with torch.no_grad():
            for _ in range(mc_samples):
                mu, log_sigma, saps2_delta = self.forward(state, action_onehot, semantic)
                mus.append(mu)
                log_sigmas.append(log_sigma)
                saps2_deltas.append(saps2_delta)

        mu_mean = torch.stack(mus, dim=0).mean(dim=0)
        mu_std = torch.stack(mus, dim=0).std(dim=0)
        aleatoric = torch.stack(log_sigmas, dim=0).exp().mean(dim=0)
        sigma = mu_std + aleatoric
        saps2_delta_mean = torch.stack(saps2_deltas, dim=0).mean(dim=0)
        return mu_mean, sigma, saps2_delta_mean

    def nll_loss(self, state, action, s_next, semantic=None):
        """NLL loss for state prediction."""
        action_onehot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        mu, log_sigma, _ = self.forward(state, action_onehot, semantic)
        sigma2 = (2 * log_sigma).exp()
        loss = ((s_next - mu) ** 2 / (2 * sigma2 + 1e-8) + log_sigma).mean()
        return loss

    def saps2_loss(self, state, action, delta_saps2_target, semantic=None):
        """MSE loss for SAPS2 delta prediction."""
        action_onehot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        _, _, saps2_pred = self.forward(state, action_onehot, semantic)
        return F.mse_loss(saps2_pred, delta_saps2_target)
