"""
Stage 1: Policy WITHOUT semantic (DT model_type), WM WITH semantic.
  - pi: behaviour cloning (CE loss on actions) with continuous SAPS-II RTG
  - O: NLL state loss + SAPS2 delta MSE loss (WITH semantic embeddings)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config_no_sem_policy as C
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

    print(f"[Stage1-NoSemPolicy] Loading v3 semantic datasets (gamma={C.RTG_GAMMA}, scale={C.RTG_SCALE})...")
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
    print(f"[Stage1-NoSemPolicy] Train: {len(train_dataset)} trajectories, Val: {len(val_dataset)} trajectories")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Build models
    lang_dim = train_dataset.language_emb_dim or C.LANGUAGE_EMB_DIM
    policy = build_policy(
        vocab_size=C.VOCAB_SIZE, block_size=C.BLOCK_SIZE,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        language_emb_dim=lang_dim, max_timestep=C.CONTEXT_LENGTH,
        model_type=C.MODEL_TYPE,
    ).to(device)

    world_model = WorldModel(
        state_dim=C.STATE_DIM, action_dim=C.VOCAB_SIZE,
        hidden_dim=C.O_HIDDEN, dropout=C.MC_DROPOUT,
    ).to(device)

    opt_pi = torch.optim.AdamW(policy.parameters(), lr=args.lr_pi, weight_decay=C.WEIGHT_DECAY)
    opt_O = torch.optim.AdamW(world_model.parameters(), lr=args.lr_O)

    epochs = args.epochs
    save_interval_epochs = args.save_interval_epochs
    log_interval_steps = args.log_interval_steps

    n_params_pi = sum(p.numel() for p in policy.parameters())
    n_params_O = sum(p.numel() for p in world_model.parameters())
    print(f"[Stage1-NoSemPolicy] pi params={n_params_pi:,}, O params={n_params_O:,}")
    print(f"[Stage1-NoSemPolicy] model_type={C.MODEL_TYPE}, block_size={C.BLOCK_SIZE}")
    print(f"[Stage1-NoSemPolicy] Starting: {epochs} epochs, save every {save_interval_epochs} epochs")

    global_step = 0
    best_val_loss = float('inf')
    best_epoch = 0
    start_epoch = 1

    for epoch in range(start_epoch, epochs + 1):
        policy.train()
        world_model.train()

        running_pi = running_O = 0.0
        log_count = 0

        for batch_idx, batch in enumerate(train_loader, 1):
            states, actions, rtgs, timesteps, traj_mask, task_embs, h_embs, f_embs, delta_saps2 = \
                _unpack_batch(batch, device)

            # --- pi: behaviour cloning ---
            targets = actions.squeeze(-1)
            forward_kwargs = dict(
                states=states, actions=actions, targets=targets,
                rtgs=rtgs, timesteps=timesteps,
                task_embeddings=task_embs, hindsight_embeddings=h_embs,
                foresight_embeddings=f_embs,
                traj_mask=traj_mask.unsqueeze(-1),
            )
            logits, action_loss, _ = policy(**forward_kwargs)
            opt_pi.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt_pi.step()

            # --- O: NLL + SAPS2 delta prediction (WITH semantic) ---
            s_t = states[:, :-1, :].reshape(-1, C.STATE_DIM)
            a_t = actions[:, :-1, 0].reshape(-1)
            s_next = states[:, 1:, :].reshape(-1, C.STATE_DIM)
            sem_flat = torch.cat([task_embs[:, :-1, :], h_embs[:, :-1, :], f_embs[:, :-1, :]], dim=-1).reshape(-1, C.LANGUAGE_EMB_DIM * 3)
            mask_flat = traj_mask[:, 1:].reshape(-1).bool()
            if mask_flat.sum() > 0:
                s_t_m = s_t[mask_flat]
                a_t_m = a_t[mask_flat]
                s_next_m = s_next[mask_flat]
                sem_m = sem_flat[mask_flat]

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
                print(f"[Stage1-NoSemPolicy] Epoch {epoch}/{epochs}, Batch {batch_idx}/{len(train_loader)}  "
                      f"pi={running_pi/log_count:.4f}  O={running_O/log_count:.4f}")
                running_pi = running_O = 0.0
                log_count = 0

        # End of epoch - run validation
        policy.eval()
        world_model.eval()
        val_pi_loss = 0.0
        val_O_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                states, actions, rtgs, timesteps, traj_mask, task_embs, h_embs, f_embs, delta_saps2 = \
                    _unpack_batch(batch, device)

                targets = actions.squeeze(-1)
                forward_kwargs = dict(
                    states=states, actions=actions, targets=targets,
                    rtgs=rtgs, timesteps=timesteps,
                    task_embeddings=task_embs, hindsight_embeddings=h_embs,
                    foresight_embeddings=f_embs,
                    traj_mask=traj_mask.unsqueeze(-1),
                )
                _, action_loss, _ = policy(**forward_kwargs)

                s_t = states[:, :-1, :].reshape(-1, C.STATE_DIM)
                a_t = actions[:, :-1, 0].reshape(-1)
                s_next = states[:, 1:, :].reshape(-1, C.STATE_DIM)
                sem_flat = torch.cat([task_embs[:, :-1, :], h_embs[:, :-1, :], f_embs[:, :-1, :]], dim=-1).reshape(-1, C.LANGUAGE_EMB_DIM * 3)
                mask_flat = traj_mask[:, 1:].reshape(-1).bool()

                if mask_flat.sum() > 0:
                    s_t_m = s_t[mask_flat]
                    a_t_m = a_t[mask_flat]
                    s_next_m = s_next[mask_flat]
                    sem_m = sem_flat[mask_flat]

                    nll = world_model.nll_loss(s_t_m, a_t_m, s_next_m, semantic=sem_m)
                    saps2_target = delta_saps2[:, :-1, :].reshape(-1, 1)[mask_flat]
                    saps2_loss = world_model.saps2_loss(s_t_m, a_t_m, saps2_target, semantic=sem_m)

                    val_pi_loss += action_loss.item()
                    val_O_loss += (nll + saps2_loss).item()
                    val_count += 1

        avg_val_pi = val_pi_loss / max(val_count, 1)
        avg_val_O = val_O_loss / max(val_count, 1)
        avg_val_total = avg_val_pi + avg_val_O

        print(f"[Stage1-NoSemPolicy] Epoch {epoch}/{epochs} complete (global_step={global_step})")
        print(f"[Stage1-NoSemPolicy] Validation: pi={avg_val_pi:.4f}, O={avg_val_O:.4f}, total={avg_val_total:.4f}")

        # Save checkpoint at intervals
        if epoch % save_interval_epochs == 0 or epoch == epochs:
            ckpt_dir = os.path.join(args.logdir, f"epoch_{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(policy.state_dict(), os.path.join(ckpt_dir, "policy.pt"))
            torch.save(world_model.state_dict(), os.path.join(ckpt_dir, "world_model.pt"))
            print(f"[Stage1-NoSemPolicy] Saved checkpoint at epoch {epoch} -> {ckpt_dir}")

        # Save best checkpoint
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            best_epoch = epoch
            best_ckpt_dir = os.path.join(args.logdir, "best_checkpoint")
            os.makedirs(best_ckpt_dir, exist_ok=True)
            torch.save(policy.state_dict(), os.path.join(best_ckpt_dir, "policy.pt"))
            torch.save(world_model.state_dict(), os.path.join(best_ckpt_dir, "world_model.pt"))
            import json
            with open(os.path.join(best_ckpt_dir, "metadata.json"), 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'val_loss': avg_val_total,
                    'val_pi_loss': avg_val_pi,
                    'val_O_loss': avg_val_O,
                    'global_step': global_step
                }, f, indent=2)
            print(f"[Stage1-NoSemPolicy] New best checkpoint! Epoch {epoch}, val_loss={avg_val_total:.4f}")

    print(f"[Stage1-NoSemPolicy] Training complete. Best checkpoint at epoch {best_epoch} with val_loss={best_val_loss:.4f}")
    print(f"[Stage1-NoSemPolicy] Best checkpoint saved in {os.path.join(args.logdir, 'best_checkpoint')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=C.DATA_DIR)
    parser.add_argument('--logdir', default='./checkpoints_no_sem_policy/stage1')
    parser.add_argument('--epochs', type=int, default=C.STAGE1_EPOCHS)
    parser.add_argument('--save_interval_epochs', type=int, default=C.SAVE_INTERVAL_EPOCHS)
    parser.add_argument('--log_interval_steps', type=int, default=C.LOG_INTERVAL_STEPS)
    parser.add_argument('--batch_size', type=int, default=C.BATCH_SIZE)
    parser.add_argument('--lr_pi', type=float, default=C.LR_PI)
    parser.add_argument('--lr_O', type=float, default=C.LR_O)
    parser.add_argument('--n_layer', type=int, default=C.N_LAYER)
    parser.add_argument('--n_head', type=int, default=C.N_HEAD)
    parser.add_argument('--n_embd', type=int, default=C.N_EMBD)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume_from', type=int, default=None,
                        help='Resume from a saved epoch checkpoint (e.g. 80)')
    args = parser.parse_args()
    train_stage1(args)
