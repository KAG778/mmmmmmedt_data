"""
Base classes for Policy and World Model
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple


class BasePolicy(nn.Module, ABC):
    """Base Policy Model"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.semantic_embed = config.get("semantic_embed")

    @abstractmethod
    def forward(self, states, actions=None, context=None):
        """Forward pass"""
        pass

    @abstractmethod
    def get_action(self, states, actions=None, context=None):
        """Get action (inference)"""
        pass

    def get_parameters(self):
        """Get model parameters"""
        return [p for p in self.parameters() if p.requires_grad]


class BaseWorldModel(nn.Module, ABC):
    """Base World Model"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.semantic_embed = config.get("semantic_embed")
        self.output_heads = config.get("output_heads", ["mu", "sigma"])

    @abstractmethod
    def forward(self, states, actions):
        """Forward pass - predict next state"""
        pass

    @abstractmethod
    def predict_saps2_delta(self, states, actions):
        """Predict SAPS2 delta"""
        pass

    def enable_dropout(self):
        """Enable dropout for MC sampling"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(self, states, actions, semantic_context=None, n_samples=10):
        """
        MC Dropout uncertainty estimation.

        Returns:
            mu_mean: (B, state_dim) mean prediction
            sigma: (B, state_dim) total uncertainty (epistemic + aleatoric)
            saps2_delta_mean: (B,) mean SAPS2 delta prediction
        """
        self.train()  # keep dropout active
        mus, log_sigmas, saps2_deltas = [], [], []

        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(states, actions, semantic_context)
                mus.append(out["mu"])
                log_sigmas.append(out["log_sigma"])
                saps2_deltas.append(out["saps2_delta"])

        mus = torch.stack(mus, dim=0)           # (n_samples, B, D)
        log_sigmas = torch.stack(log_sigmas, dim=0)
        saps2_deltas = torch.stack(saps2_deltas, dim=0)

        mu_mean = mus.mean(dim=0)
        epistemic = mus.std(dim=0)               # model uncertainty
        aleatoric = log_sigmas.exp().mean(dim=0)  # data uncertainty
        sigma = epistemic + aleatoric
        saps2_delta_mean = saps2_deltas.mean(dim=0)

        return mu_mean, sigma, saps2_delta_mean

    def get_parameters(self):
        """Get model parameters"""
        return [p for p in self.parameters() if p.requires_grad]


class SemanticEmbedding(nn.Module):
    """Semantic embedding layer for V6/W1/Qwen templates"""

    def __init__(self, template_type: str, embed_dim: int = 128):
        super().__init__()
        self.template_type = template_type
        self.embed_dim = embed_dim

        if template_type == "V6":
            # V6 template embedding
            self.embedding = nn.Sequential(
                nn.Linear(512, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        elif template_type == "W1":
            # W1 template embedding (quantitative physiological)
            self.embedding = nn.Sequential(
                nn.Linear(256, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        elif template_type == "Qwen":
            # Qwen LLM embedding (896 dim from text-phys45 data)
            self.embedding = nn.Sequential(
                nn.Linear(896, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        elif template_type == "Qwen-TaskForesight":
            # Combined task + foresight (896 + 896 = 1792)
            self.embedding = nn.Sequential(
                nn.Linear(1792, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        elif template_type == "Qwen-Combined":
            # Combined Qwen embeddings (task + hindsight + foresight, 3 * 896 = 2688)
            self.embedding = nn.Sequential(
                nn.Linear(2688, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        else:
            self.embedding = None

    def forward(self, semantic_input):
        if self.embedding is None:
            return None
        return self.embedding(semantic_input)
