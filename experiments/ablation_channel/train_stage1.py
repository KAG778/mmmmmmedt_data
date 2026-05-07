"""
Stage 1 (Channel Ablation): Cold-start training with masked semantic channels.
  - Policy: SeMDT with channel-masked semantic embeddings
  - WorldModel: WITH semantic but masked channels zeroed in concat
  - Training: epoch-based (100 epochs)

Usage:
  python train_stage1.py --variant A1_no_task
  python train_stage1.py --variant A4_only_task
"""
import argparse
import os
import sys

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, '..', '..', 'main_model', 'scheme3_cspdt_v2')
sys.path.insert(0, script_dir)
sys.path.insert(0, model_dir)

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from channel_mask import mask_embeddings, mask_semantic_concat, get_active_channels, variant_tag
from models.policy import build_policy
from models.world_model import WorldModel
from datasets.v3_semantic_dataset import V3SemanticDataset


def _unpack_batch(batch, device):
    states, actions, rtgs, timesteps, saps, div_saps, traj_mask, traj_len, \
        task_embs, h_embs, f_embs, delta_saps2 = batch
    return (
        states.to(device).float(),
        actions.to(device).long().unsqueeze(-1),
        rtgs.to(device).float().unsqueeze(-1),
        timesteps.to(device).unsqueeze(-1),
        traj_mask.to(device).float(),
        task_embs.to(device).float(),
        h_embs.to(device).float(),
        f_embs.to(device).float(),
        delta_saps2.to(device).float().unsqueeze(-1),
    )


def train_stage1(args):
    # Load variant-specific config
    config_name = f'config_{args.variant}'
    C = __import__(config_name)
    variant = C.VARIANT

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logdir = args.logdir or os.path.join(C.CHECKPOINT_DIR, 'stage1')
    os.makedirs(logdir, exist_ok=True)

    active = get_active_channels(variant)
    tag = variant_tag(variant)
    print(f"{tag} Stage 1 Channel Ablation Training")
    print(f"{tag} Active channels: {active}")

    print(f"{tag} Loading v3 semantic datasets...")
    train_dataset = V3SemanticDataset(
        os.path.join(C.DATA_DIR, 'train_Phys45_v3.pickle'),
        C.CONTEXT_LENGTH, C.RTG_SCALE, gamma=C.RTG_GAMMA,
        language_emb_dim=C.LANGUAGE_EMB_DIM,
    )
    print(f"{tag} Train: {len(train_dataset)} trajectories")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    lang_dim = train_dataset.language_emb_dim or C.LANGUAGE_EMB_DIM

    # Policy WITH channel-masked semantic embeddings
    policy = build_policy(
        vocab_size=C.VOCAB_SIZE, block_size=C.BLOCK_SIZE,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        language_emb_dim=lang_dim, max_timestep=C.CONTEXT_LENGTH,
        model_type=C.MODEL_TYPE,
    ).to(device)

    # WorldModel WITH semantic (but channels will be masked)
    world_model = WorldModel(
        state_dim=C.STATE_DIM, action_dim=C.VOCAB_SIZE,
        hidden_dim=C.O_HIDDEN, dropout=C.MC_DROPOUT,
        use_semantic=True,
    ).to(device)

    opt_pi = torch.optim.AdamW(policy.parameters(), lr=args.lr_pi, weight_decay=C.WEIGHT_DECAY)
    opt_O = torch.optim.AdamW(world_model.parameters(), lr=args.lr_O)

    epochs = args.epochs
    save_interval_epochs = args.save_interval_epochs
    log_interval_steps = args.log_interval_steps

    n_params_pi = sum(p.numel() for p in policy.parameters())
    n_params_O = sum(p.numel() for p in world_model.parameters())
    print(f"{tag} pi params={n_params_pi:,}, O params={n_params_O:,}")
    print(f"{tag} Starting: {epochs} epochs, save every {save_interval_epochs} epochs")

    global_step = 0
    for epoch in range(1, epochs + 1):
        policy.train()
        world_model.train()

        running_pi = running_O = 0.0
        log_count = 0

        for batch_idx, batch in enumerate(train_loader, 1):
            states, actions, rtgs, timesteps, traj_mask, task_embs, h_embs, f_embs, delta_saps2 = \
                _unpack_batch(batch, device)

            # --- Channel masking ---
            task_embs, h_embs, f_embs = mask_embeddings(task_embs, h_embs, f_embs, variant)

            # --- pi: behaviour cloning (with masked semantic) ---
            targets = actions.squeeze(-1)
            forward_kwargs = dict(
                states=states, actions=actions, targets=targets,
                rtgs=rtgs, timesteps=timesteps,
                task_embeddings=task_embs, hindsight_embeddings=h_embs,
                foresight_embeddings=f_embs,
                traj_mask=traj_mask.unsqueeze(-1),
            )
            if 'ATG' in C.MODEL_TYPE:
                forward_kwargs['delta_saps2'] = delta_saps2
            logits, action_loss, _ = policy(**forward_kwargs)
            opt_pi.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt_pi.step()

            # --- O: NLL + SAPS2 delta prediction (with masked semantic concat) ---
            s_t = states[:, :-1, :].reshape(-1, C.STATE_DIM)
            a_t = actions[:, :-1, 0].reshape(-1)
            s_next = states[:, 1:, :].reshape(-1, C.STATE_DIM)
            # Build semantic concat and mask channels
            task_flat = task_embs[:, :-1, :].reshape(-1, 896)
            h_flat = h_embs[:, :-1, :].reshape(-1, 896)
            f_flat = f_embs[:, :-1, :].reshape(-1, 896)
            sem_concat = torch.cat([task_flat, h_flat, f_flat], dim=-1)  # (N, 2688)
            sem_concat = mask_semantic_concat(sem_concat, variant)

            mask_flat = traj_mask[:, 1:].reshape(-1).bool()
            if mask_flat.sum() > 0:
                s_t_m = s_t[mask_flat]
                a_t_m = a_t[mask_flat]
                s_next_m = s_next[mask_flat]
                sem_m = sem_concat[mask_flat]

                nll = world_model.nll_loss(s_t_m, a_t_m, s_next_m, semantic=sem_m)

                saps2_target = delta_saps2[:, :-1, :].reshape(-1, 1)[mask_flat]
                saps2_loss = world_model.saps2_loss(s_t_m, a_t_m, saps2_target, semantic=sem_m)

                opt_O.zero_grad()
                (nll + saps2_loss).backward()
                torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
                opt_O.step()
                running_O += (nll + saps2_loss).item()

            running_pi += action_loss.item()
            log_count += 1
            global_step += 1

            if batch_idx % log_interval_steps == 0:
                print(f"{tag} Epoch {epoch}/{epochs}, Batch {batch_idx}/{len(train_loader)}  "
                      f"pi={running_pi/log_count:.4f}  O={running_O/log_count:.4f}")
                running_pi = running_O = 0.0
                log_count = 0

        # End of epoch
        print(f"{tag} Epoch {epoch}/{epochs} complete (global_step={global_step})")

        # Save checkpoint at intervals
        if epoch % save_interval_epochs == 0 or epoch == epochs:
            ckpt_dir = os.path.join(logdir, f"epoch_{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(policy.state_dict(), os.path.join(ckpt_dir, "policy.pt"))
            torch.save(world_model.state_dict(), os.path.join(ckpt_dir, "world_model.pt"))
            print(f"{tag} Saved checkpoint at epoch {epoch} -> {ckpt_dir}")

    print(f"{tag} Training complete. Final checkpoint in {logdir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True,
                        choices=['A1_no_task', 'A2_no_hindsight', 'A3_no_foresight',
                                 'A4_only_task', 'A5_only_hindsight', 'A6_only_foresight'])
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_interval_epochs', type=int, default=10)
    parser.add_argument('--log_interval_steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_pi', type=float, default=6e-4)
    parser.add_argument('--lr_O', type=float, default=1e-3)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    train_stage1(args)
