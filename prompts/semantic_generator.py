"""
Runtime semantic embedding generator for Stage2 counterfactual training.

This module provides functions to generate hindsight and foresight embeddings
for better_action during Stage2 training, fixing the semantic mismatch issue
where doctor_action semantics were incorrectly used for better_action training.
"""
import torch
import numpy as np
from typing import Tuple
from .saps2_qualitative_prompts import (
    _build_history_text,
    _build_foresight_text,
    _to_numpy,
)
from .text_encoder import PromptTextEncoder


class SemanticGenerator:
    """Generate semantic embeddings for actions at runtime."""

    def __init__(self, encoder: PromptTextEncoder, device: str = "cuda"):
        self.encoder = encoder
        self.device = device

    def generate_for_action(
        self,
        prev_state: np.ndarray,
        curr_state: np.ndarray,
        next_acuities: np.ndarray,
        action_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate hindsight and foresight embeddings for a specific action.

        Args:
            prev_state: (45,) normalized state at t-1
            curr_state: (45,) normalized state at t
            next_acuities: (10,) acuities at t+1 (for foresight)
            action_idx: discrete action ID

        Returns:
            hindsight_emb: (1, 896) tensor
            foresight_emb: (1, 896) tensor
        """
        prev_state_np = _to_numpy(prev_state).reshape(-1)
        curr_state_np = _to_numpy(curr_state).reshape(-1)
        next_acuities_np = _to_numpy(next_acuities).reshape(-1)

        # Generate hindsight text (action + state transitions)
        hindsight_text = _build_history_text(
            prev_state_np, curr_state_np, action_idx
        )

        # Generate foresight text (risk assessment + organ protection)
        foresight_text = _build_foresight_text(
            curr_state_np, next_acuities_np
        )

        # Encode to embeddings
        hindsight_emb_np = self.encoder.encode_texts(
            [hindsight_text], batch_size=1, output_dtype="float16"
        )
        foresight_emb_np = self.encoder.encode_texts(
            [foresight_text], batch_size=1, output_dtype="float16"
        )

        # Convert to torch tensors
        hindsight_emb = torch.from_numpy(hindsight_emb_np).to(self.device)
        foresight_emb = torch.from_numpy(foresight_emb_np).to(self.device)

        return hindsight_emb, foresight_emb
