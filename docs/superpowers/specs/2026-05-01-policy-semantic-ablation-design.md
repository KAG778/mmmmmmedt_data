# Policy Semantic Component Ablation Design

## Goal

Quantify the contribution of each semantic component (Task / Hindsight / Foresight embedding) in the CSP-DT Policy. All experiments use World Model WITHOUT semantic embeddings (`use_semantic=False`), sigma=2.0.

## Ablation Matrix

7 new variants against the existing NoSem-WM baseline:

| # | Variant | Task | Hindsight | Foresight | Category |
|---|---------|:----:|:---------:|:---------:|----------|
| 0 | Full (baseline) | Yes | Yes | Yes | Existing result |
| 1 | `no_task` | Zero | Yes | Yes | Remove one |
| 2 | `no_hindsight` | Yes | Zero | Yes | Remove one |
| 3 | `no_foresight` | Yes | Yes | Zero | Remove one |
| 4 | `only_task` | Yes | Zero | Zero | Keep one |
| 5 | `only_hindsight` | Zero | Yes | Zero | Keep one |
| 6 | `only_foresight` | Zero | Zero | Yes | Keep one |
| 7 | `no_all` | Zero | Zero | Zero | Remove all |

## Implementation: Zero-out

- Policy retains the SeMDT 6-token architecture unchanged
- Ablated components are replaced with `torch.zeros_like(...)` before feeding into the model
- All variants share identical model structure and parameter count
- Only the input embedding content differs

### Zero-out Logic

Applied uniformly in Stage 1, Stage 2, and evaluation:

```python
ABLATE = config.ABLATE_SEMANTIC  # e.g., {'task': True, 'hindsight': False, 'foresight': True}
if ABLATE.get('task', False):
    task_embs = torch.zeros_like(task_embs)
if ABLATE.get('hindsight', False):
    h_embs = torch.zeros_like(h_embs)
if ABLATE.get('foresight', False):
    f_embs = torch.zeros_like(f_embs)
```

## Training Pipeline (per variant)

1. **Stage 1** (100 epochs): Policy BC with ablated embeddings + WM without semantic
2. **Stage 2** (50 epochs x 1000 iterations): Counterfactual self-play with ablated embeddings + WM without semantic
3. **Evaluation**: 10-step rollout, sigma=2.0, stratified by SAPS2 severity (low/mid/high)

All stages train from scratch per variant. No checkpoint reuse across variants.

## File Structure

```
main_model_III_abalation/scheme3_cspdt_v2/
├── config_ablation_semantic.py     # Defines ABLATE_VARIANTS dict with 7 configurations
├── train_stage1_ablation.py        # Stage 1: accepts --variant flag
├── train_stage2_ablation.py        # Stage 2: accepts --variant flag
├── stratified_rollout_ablation.py  # Evaluation: accepts --variant flag
├── run_ablation_all.sh             # Automation: runs all 7 variants sequentially
```

## Config Design

`config_ablation_semantic.py` imports from `config_no_sem_sigma2.py` and adds:

```python
ABLATE_VARIANTS = {
    'no_task':       {'task': True,  'hindsight': False, 'foresight': False},
    'no_hindsight':  {'task': False, 'hindsight': True,  'foresight': False},
    'no_foresight':  {'task': False, 'hindsight': False, 'foresight': True},
    'only_task':     {'task': False, 'hindsight': True,  'foresight': True},
    'only_hindsight':{'task': True,  'hindsight': False, 'foresight': True},
    'only_foresight':{'task': True,  'hindsight': True,  'foresight': False},
    'no_all':        {'task': True,  'hindsight': True,  'foresight': True},
}
```

Each script reads `ABLATE_VARIANTS[args.variant]` and applies zero-out before model forward pass.

## Checkpoint & Results Layout

```
checkpoints_ablation/
├── no_task/stage1/  + stage2/best_checkpoint.pt
├── no_hindsight/stage1/  + stage2/best_checkpoint.pt
├── no_foresight/stage1/  + stage2/best_checkpoint.pt
├── only_task/stage1/  + stage2/best_checkpoint.pt
├── only_hindsight/stage1/  + stage2/best_checkpoint.pt
├── only_foresight/stage1/  + stage2/best_checkpoint.pt
├── no_all/stage1/  + stage2/best_checkpoint.pt

results_ablation/
├── no_task.json
├── no_hindsight.json
├── no_foresight.json
├── only_task.json
├── only_hindsight.json
├── only_foresight.json
├── no_all.json
└── summary.json    # Aggregated comparison table
```

## Hyperparameters

All inherited from `config_no_sem_sigma2.py`:
- CONTEXT_LENGTH=20, VOCAB_SIZE=25, STATE_DIM=45, RTG_SCALE=10
- N_LAYER=4, N_HEAD=8, N_EMBD=128, LANGUAGE_EMB_DIM=896
- SIGMA_THRESHOLD=2.0, SIGMA_THRESHOLD_SEARCH=2.0
- STAGE1_EPOCHS=100, STAGE2_EPOCHS=50
- SELFPLAY_ITERATIONS=1000, SEARCH_RADIUS=2
- MC_SAMPLES=10, UNCERTAINTY_CUTOFF=2.0

## Automation Script

`run_ablation_all.sh` iterates over all 7 variants:
1. Stage 1 training → save to `checkpoints_ablation/<variant>/stage1/`
2. Stage 2 training → load best stage1, save to `checkpoints_ablation/<variant>/stage2/`
3. Evaluation → load best stage2, output to `results_ablation/<variant>.json`
4. Collect all results into `results_ablation/summary.json`

## Success Criteria

- All 7 variants complete Stage 1 + Stage 2 + Evaluation without errors
- Results table shows clear ranking of semantic component importance
- At least one "remove one" variant shows significant performance drop vs baseline (validating semantic utility)
- At least one "keep one" variant shows which single component is most critical
