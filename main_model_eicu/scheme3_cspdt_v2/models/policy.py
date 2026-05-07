import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'medical_semdt/scripts'))

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.GPT import GPTConfig
from models.MeDT import MeDT


class CSPDTPolicy(MeDT):
    """
    SeMDT-based policy for CSP-DT.
    Inherits MeDT directly; adds temperature sampling and candidate generation.
    Supports 6-token (SeMDT) and 7-token (SeMDT_ATG_A/B) variants.
    """

    def get_action_logits(self, states, actions, rtgs, timesteps,
                          task_embeddings, hindsight_embeddings, foresight_embeddings,
                          delta_saps2=None):
        """
        Run a forward pass and return logits for the last timestep.
        Returns: (B, VOCAB_SIZE) logits
        """
        logits, _, _ = self.forward(
            states=states,
            actions=actions,
            targets=None,
            rtgs=rtgs,
            timesteps=timesteps,
            task_embeddings=task_embeddings,
            hindsight_embeddings=hindsight_embeddings,
            foresight_embeddings=foresight_embeddings,
            delta_saps2=delta_saps2,
        )
        # logits shape: (B, T, vocab_size) — take last valid timestep
        return logits[:, -1, :]  # (B, vocab_size)

    def sample_candidates(self, states, actions, rtgs, timesteps,
                          task_embeddings, hindsight_embeddings, foresight_embeddings,
                          doc_action, n_cf=3, temperature=1.5, delta_saps2=None):
        """
        Generate candidate actions for counterfactual self-play.

        Returns list of tensors, each shape (B,):
            [greedy_action, doc_action, cf_1, ..., cf_n_cf]
        """
        self.eval()
        with torch.no_grad():
            logits = self.get_action_logits(
                states, actions, rtgs, timesteps,
                task_embeddings, hindsight_embeddings, foresight_embeddings,
                delta_saps2=delta_saps2,
            )  # (B, vocab_size)

            # greedy
            greedy_action = logits.argmax(dim=-1)  # (B,)

            # temperature-sampled counterfactuals
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            cf_actions = [
                torch.multinomial(probs, num_samples=1).squeeze(-1)
                for _ in range(n_cf)
            ]

        candidates = [greedy_action, doc_action] + cf_actions
        return candidates  # list of (B,) tensors

    def log_prob(self, states, actions, rtgs, timesteps,
                 task_embeddings, hindsight_embeddings, foresight_embeddings,
                 candidate_action, delta_saps2=None):
        """
        Compute log-probability of candidate_action under current policy.
        candidate_action: (B,) long tensor
        Returns: (B,) log-prob tensor
        """
        logits = self.get_action_logits(
            states, actions, rtgs, timesteps,
            task_embeddings, hindsight_embeddings, foresight_embeddings,
            delta_saps2=delta_saps2,
        )  # (B, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, vocab_size)
        return log_probs.gather(1, candidate_action.unsqueeze(-1)).squeeze(-1)  # (B,)


def build_policy(vocab_size, block_size, n_layer, n_head, n_embd,
                 language_emb_dim, max_timestep, model_type='SeMDT'):
    mconf = GPTConfig(
        vocab_size, block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        model_type=model_type,
        max_timestep=max_timestep,
        language_emb_dim=language_emb_dim,
    )
    return CSPDTPolicy(mconf)
