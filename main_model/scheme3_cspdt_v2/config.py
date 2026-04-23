# CSP-DT Hyperparameter Configuration (v3 Semantic Embeddings)

# Data
CONTEXT_LENGTH = 20
VOCAB_SIZE = 25       # action vocab size
STATE_DIM = 45
RTG_SCALE = 10        # v3: continuous SAPS-II RTG scale
RTG_GAMMA = 0.99      # v3: discounted cumulative sum

# Policy Transformer (π)
N_LAYER = 4
N_HEAD = 8
N_EMBD = 128
LANGUAGE_EMB_DIM = 896   # SeMDT text embedding dim
BLOCK_SIZE = CONTEXT_LENGTH * 6  # SeMDT: 6 tokens per timestep
MODEL_TYPE = 'SeMDT'     # 6-token variant; see _ATG dir for 7-token

# World Model (O)
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

# Training — Stage 1 (epoch-based, aligned with baselines)
STAGE1_EPOCHS = 100
SAVE_INTERVAL_EPOCHS = 10  # Save checkpoint every 10 epochs
LOG_INTERVAL_STEPS = 100   # Print log every 100 batches

# Training — Stage 2 (epoch-based, per-sample self-play)
STAGE2_EPOCHS = 50
SELFPLAY_ITERATIONS = 1000  # per epoch
SEARCH_RADIUS = 2            # ±2 neighborhood action search

# Evaluation
UNCERTAINTY_CUTOFF = 2.0   # cumulative σ threshold for early stopping
MC_SAMPLES = 10

# Confidence-gated semi-bootstrap (Stage 2)
SIGMA_THRESHOLD_SEARCH = 1.0   # for action search confidence filtering (lowered from 2.0 for higher quality)
ALPHA_MIN = 0.3                # minimum base_alpha for semi-bootstrap
ALPHA_DECAY_STEPS = 50000      # steps over which base_alpha decays from 1.0 to ALPHA_MIN
CONFIDENCE_GATE = 0.5          # minimum confidence to apply bootstrap loss

# Paths
DATA_DIR = "/home/wangmeiyi/AuctionNet/medical/last_exp/data/v3"
