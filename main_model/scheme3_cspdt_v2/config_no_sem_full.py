# CSP-DT Config: BOTH Policy and WM WITHOUT semantic embeddings
# Diff from config.py: MODEL_TYPE='DT' (3-token), USE_SEMANTIC_WM=False

# Data
CONTEXT_LENGTH = 20
VOCAB_SIZE = 25       # action vocab size
STATE_DIM = 45
RTG_SCALE = 10        # v3: continuous SAPS-II RTG scale
RTG_GAMMA = 0.99      # v3: discounted cumulative sum

# Policy Transformer (π) — DT 3-token (no semantic)
N_LAYER = 4
N_HEAD = 8
N_EMBD = 128
LANGUAGE_EMB_DIM = 896   # Not used by DT model_type, kept for dataset compatibility
BLOCK_SIZE = CONTEXT_LENGTH * 3  # DT: 3 tokens per timestep
MODEL_TYPE = 'DT'     # 3-token variant, no semantic embeddings in policy

# World Model (O) — NO semantic embeddings
USE_SEMANTIC_WM = False   # <-- key difference: WM gets no semantic input
O_HIDDEN = 256
MC_DROPOUT = 0.2

# Sampling
TEMPERATURE = 1.5
N_CF = 3              # number of counterfactual candidates

# Self-play filtering
SIGMA_THRESHOLD = 2.0     # v3: more lenient than v2's 0.3
LAMBDA_O = 0.5

# Optimizers
LR_PI = 6e-4
LR_O = 1e-3
BATCH_SIZE = 64
WEIGHT_DECAY = 0.1

# Training — Stage 1 (epoch-based)
STAGE1_EPOCHS = 100
SAVE_INTERVAL_EPOCHS = 10
LOG_INTERVAL_STEPS = 100

# Training — Stage 2 (epoch-based, per-sample self-play)
STAGE2_EPOCHS = 50
SELFPLAY_ITERATIONS = 1000  # per epoch
SEARCH_RADIUS = 2            # ±2 neighborhood action search

# Evaluation
UNCERTAINTY_CUTOFF = 2.0   # cumulative σ threshold for early stopping
MC_SAMPLES = 10

# Confidence-gated semi-bootstrap (Stage 2)
SIGMA_THRESHOLD_SEARCH = 2.0   # for action search confidence filtering
ALPHA_MIN = 0.3                # minimum base_alpha for semi-bootstrap
ALPHA_DECAY_STEPS = 50000      # steps over which base_alpha decays from 1.0 to ALPHA_MIN
CONFIDENCE_GATE = 0.5          # minimum confidence to apply bootstrap loss

# Paths
DATA_DIR = "/home/wangmeiyi/AuctionNet/medical/last_exp/data/v3"
