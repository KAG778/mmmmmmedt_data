"""
Stage 1 (No WM Semantic): Cold-start training with v3 semantic embeddings.
  - pi: behaviour cloning (CE loss on actions) with continuous SAPS-II RTG — UNCHANGED
  - O: NLL state loss + SAPS2 delta MSE loss — WITHOUT semantic embeddings
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config_no_sem as C
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
        delta_saps2.to(device).float().unsqueeze(-1),  # (B, T, 1)
    )


def train_stage1(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    os.makedirs(args.logdir, exist_ok=True)

    print(f"[Stage1-NoSem] Loading v3 semantic datasets (gamma={C.RTG_GAMMA}, scale={C.RTG_SCALE})...")
    train_dataset = V3SemanticDataset(
        os.path.join(args.datadir, 'train_Phys45_v3.pickle'),
        C.CONTEXT_LENGTH, C.RTG_SCALE, gamma=C.RTG_GAMMA,
        language_emb_dim=C.LANGUAGE_EMB_DIM,
    )
    val_dataset = V3SemanticDataset(
        os.path.join(args.datadir, 'test_Phys45_v3.pickle'),
        C.CONTEXT_LENGTH, C.RTG_SCALE, gamma=C.RTG_GAMMA,
        language_emb_dim=C.LANGUAGE_EMB_DIM,
    )
    print(f"[Stage1-NoSem] Train: {len(train_dataset)} trajectories, Val: {len(val_dataset)} trajectories")

    rtg_vals = train_dataset.trajectories[0]['returns_to_go']
    print(f"[Stage1-NoSem] Sample RTG (traj 0): {rtg_vals[:5]}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    lang_dim = train_dataset.language_emb_dim or C.LANGUAGE_EMB_DIM

    policy = build_policy(
        vocab_size=C.VOCAB_SIZE, block_size=C.BLOCK_SIZE,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        language_emb_dim=lang_dim, max_timestep=C.CONTEXT_LENGTH,
        model_type=C.MODEL_TYPE,
    ).to(device)

    # WorldModel WITHOUT semantic — use_semantic=False
    world_model = WorldModel(
        state_dim=C.STATE_DIM, action_dim=C.VOCAB_SIZE,
        hidden_dim=C.O_HIDDEN, dropout=C.MC_DROPOUT,
        use_semantic=False,
    ).to(device)

    opt_pi = torch.optim.AdamW(policy.parameters(), lr=args.lr_pi, weight_decay=C.WEIGHT_DECAY)
    opt_O = torch.optim.AdamW(world_model.parameters(), lr=args.lr_O)

    max_steps = args.max_steps
    save_interval = args.save_interval
    log_interval = args.log_interval

    step = 0
    train_iter = iter(train_loader)
    running_pi = running_O = 0.0
    log_count = 0
    n_params_pi = sum(p.numel() for p in policy.parameters())
    n_params_O = sum(p.numel() for p in world_model.parameters())
    print(f"[Stage1-NoSem] pi params={n_params_pi:,}, O params={n_params_O:,}")
    print(f"[Stage1-NoSem] WorldModel use_semantic=False")
    print(f"[Stage1-NoSem] Starting: max_steps={max_steps}, save_interval={save_interval}")

    while step < max_steps:
        policy.train()
        world_model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        states, actions, rtgs, timesteps, traj_mask, task_embs, h_embs, f_embs, delta_saps2 = \
            _unpack_batch(batch, device)

        # --- pi: behaviour cloning (UNCHANGED, still uses semantic embeddings) ---
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

        # --- O: NLL + SAPS2 delta prediction (NO semantic) ---
        s_t = states[:, :-1, :].reshape(-1, C.STATE_DIM)
        a_t = actions[:, :-1, 0].reshape(-1)
        s_next = states[:, 1:, :].reshape(-1, C.STATE_DIM)
        mask_flat = traj_mask[:, 1:].reshape(-1).bool()
        if mask_flat.sum() > 0:
            s_t_m = s_t[mask_flat]
            a_t_m = a_t[mask_flat]
            s_next_m = s_next[mask_flat]

            # NLL loss — no semantic passed
            nll = world_model.nll_loss(s_t_m, a_t_m, s_next_m, semantic=None)

            # SAPS2 delta loss — no semantic passed
            saps2_target = delta_saps2[:, :-1, :].reshape(-1, 1)[mask_flat]
            saps2_loss = world_model.saps2_loss(s_t_m, a_t_m, saps2_target, semantic=None)

            opt_O.zero_grad()
            (nll + saps2_loss).backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
            opt_O.step()
            running_O += (nll + saps2_loss).item()

        running_pi += action_loss.item()
        log_count += 1
        step += 1

        if step % log_interval == 0:
            print(f"[Stage1-NoSem] step {step}/{max_steps}  pi={running_pi/log_count:.4f}  O={running_O/log_count:.4f}")
            running_pi = running_O = 0.0
            log_count = 0

        if step % save_interval == 0 or step == max_steps:
            ckpt_dir = os.path.join(args.logdir, f"step_{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(policy.state_dict(), os.path.join(ckpt_dir, "policy.pt"))
            torch.save(world_model.state_dict(), os.path.join(ckpt_dir, "world_model.pt"))
            print(f"[Stage1-NoSem] Saved checkpoint at step {step} -> {ckpt_dir}")

    print(f"[Stage1-NoSem] Training complete. Checkpoints in {args.logdir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=C.DATA_DIR)
    parser.add_argument('--logdir', default='./checkpoints/stage1_no_sem')
    parser.add_argument('--max_steps', type=int, default=C.STAGE1_MAX_STEPS)
    parser.add_argument('--save_interval', type=int, default=C.SAVE_INTERVAL)
    parser.add_argument('--log_interval', type=int, default=C.LOG_INTERVAL)
    parser.add_argument('--batch_size', type=int, default=C.BATCH_SIZE)
    parser.add_argument('--lr_pi', type=float, default=C.LR_PI)
    parser.add_argument('--lr_O', type=float, default=C.LR_O)
    parser.add_argument('--n_layer', type=int, default=C.N_LAYER)
    parser.add_argument('--n_head', type=int, default=C.N_HEAD)
    parser.add_argument('--n_embd', type=int, default=C.N_EMBD)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    train_stage1(args)
