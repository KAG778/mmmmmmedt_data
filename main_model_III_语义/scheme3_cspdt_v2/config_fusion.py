# Shared config for fusion ablation variants
# Only MODEL_TYPE and BLOCK_SIZE differ per variant
CONTEXT_LENGTH = 20
VOCAB_SIZE = 25
STATE_DIM = 45
RTG_SCALE = 10
RTG_GAMMA = 0.99

N_LAYER = 4
N_HEAD = 8
N_EMBD = 128
LANGUAGE_EMB_DIM = 896

USE_SEMANTIC_WM = False  # WM unchanged (no semantic input)
O_HIDDEN = 256
MC_DROPOUT = 0.2

SIGMA_THRESHOLD = 2.0
LAMBDA_O = 0.5
LR_PI = 6e-4
LR_O = 1e-3
BATCH_SIZE = 64
WEIGHT_DECAY = 0.1

STAGE1_EPOCHS = 100
SAVE_INTERVAL_EPOCHS = 10
LOG_INTERVAL_STEPS = 100
STAGE2_EPOCHS = 50
SELFPLAY_ITERATIONS = 1000
SEARCH_RADIUS = 2

UNCERTAINTY_CUTOFF = 2.0
MC_SAMPLES = 10

SIGMA_THRESHOLD_SEARCH = 2.0
ALPHA_MIN = 0.3
ALPHA_DECAY_STEPS = 50000
CONFIDENCE_GATE = 0.5

DATA_DIR = "/home/wangmeiyi/AuctionNet/medical/last_exp/data/v3"
_BASE_DIR = "/home/wangmeiyi/AuctionNet/medical/last_exp/main_model_III_语义/scheme3_cspdt_v2"

FUSION_VARIANTS = {
    'concat':     {'MODEL_TYPE': 'SeMDT_Concat',    'TOKENS': 3},
    'residual':   {'MODEL_TYPE': 'SeMDT_Residual',  'TOKENS': 3},
    'gated':      {'MODEL_TYPE': 'SeMDT_Gated',     'TOKENS': 3},
    'cross_attn': {'MODEL_TYPE': 'SeMDT_CrossAttn', 'TOKENS': 3},
}

# Default (for the first variant); overwritten per variant at runtime
MODEL_TYPE = 'SeMDT_Concat'
BLOCK_SIZE = CONTEXT_LENGTH * 3  # 60


def configure_variant(variant_name):
    """Set MODEL_TYPE and BLOCK_SIZE for a specific fusion variant.
    Returns (model_type, block_size, tokens)."""
    if variant_name not in FUSION_VARIANTS:
        raise ValueError(f"Unknown fusion variant: {variant_name}. "
                         f"Choose from {list(FUSION_VARIANTS.keys())}")
    info = FUSION_VARIANTS[variant_name]
    global MODEL_TYPE, BLOCK_SIZE
    MODEL_TYPE = info['MODEL_TYPE']
    BLOCK_SIZE = CONTEXT_LENGTH * info['TOKENS']
    return MODEL_TYPE, BLOCK_SIZE, info['TOKENS']


def get_dirs(variant_name):
    """Return (checkpoint_dir, log_dir, result_dir) for a fusion variant."""
    ckpt = os.path.join(_BASE_DIR, 'checkpoints_fusion', variant_name)
    log = os.path.join(_BASE_DIR, 'logs_fusion', variant_name)
    result = os.path.join(_BASE_DIR, 'results_fusion', variant_name)
    return ckpt, log, result


import os
