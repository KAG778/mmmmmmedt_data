"""
Channel masking utilities for semantic channel ablation experiments.

Each variant masks specific semantic channels by zeroing out the corresponding
embeddings while keeping model architecture unchanged.

Channel layout:
  - task_embeddings:      896-dim  (patient profile + treatment target)
  - hindsight_embeddings: 896-dim  (previous action effect + state trend)
  - foresight_embeddings: 896-dim  (risk prediction + organ protection)

Variants:
  A1: no_task        — mask task
  A2: no_hindsight   — mask hindsight
  A3: no_foresight   — mask foresight
  A4: only_task      — mask hindsight + foresight
  A5: only_hindsight — mask task + foresight
  A6: only_foresight — mask task + hindsight
"""
import torch

VARIANT_CONFIGS = {
    'A1_no_task':        {'task': False, 'hindsight': True,  'foresight': True},
    'A2_no_hindsight':   {'task': True,  'hindsight': False, 'foresight': True},
    'A3_no_foresight':   {'task': True,  'hindsight': True,  'foresight': False},
    'A4_only_task':      {'task': True,  'hindsight': False, 'foresight': False},
    'A5_only_hindsight': {'task': False, 'hindsight': True,  'foresight': False},
    'A6_only_foresight': {'task': False, 'hindsight': False, 'foresight': True},
}


def get_active_channels(variant):
    """Return dict of which channels are active for a given variant."""
    if variant not in VARIANT_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(VARIANT_CONFIGS.keys())}")
    return VARIANT_CONFIGS[variant]


def mask_embeddings(task_emb, h_emb, f_emb, variant):
    """
    Zero out inactive channels based on variant config.
    All embeddings should be tensors of shape (..., 896).
    Returns (task_emb, h_emb, f_emb) with masked channels zeroed.
    """
    cfg = get_active_channels(variant)
    device = task_emb.device

    task_emb = task_emb if cfg['task'] else torch.zeros_like(task_emb)
    h_emb = h_emb if cfg['hindsight'] else torch.zeros_like(h_emb)
    f_emb = f_emb if cfg['foresight'] else torch.zeros_like(f_emb)

    return task_emb, h_emb, f_emb


def mask_semantic_concat(sem_concat, variant):
    """
    Zero out inactive channels in concatenated semantic embedding.
    sem_concat: (B, 2688) = [task(896) | hindsight(896) | foresight(896)]
    """
    cfg = get_active_channels(variant)
    dim = 896
    sem_concat = sem_concat.clone()
    if not cfg['task']:
        sem_concat[:, :dim] = 0.0
    if not cfg['hindsight']:
        sem_concat[:, dim:2*dim] = 0.0
    if not cfg['foresight']:
        sem_concat[:, 2*dim:] = 0.0
    return sem_concat


def variant_tag(variant):
    """Short tag for log messages."""
    cfg = get_active_channels(variant)
    active = [k[0].upper() for k, v in cfg.items() if v]
    return f"[{variant}]({'+' if active else 'none'})"
