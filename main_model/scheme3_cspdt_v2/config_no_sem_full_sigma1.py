# CSP-DT Config: BOTH Policy and WM WITHOUT semantic — sigma1.0 variant
# Diff from config_no_sem_full.py: SIGMA_THRESHOLD_SEARCH = 1.0

# Data
CONTEXT_LENGTH = 20
VOCAB_SIZE = 25
STATE_DIM = 45
RTG_SCALE = 10
RTG_GAMMA = 0.99

# Policy Transformer (π) — DT 3-token (no semantic)
N_LAYER = 4
N_HEAD = 8
N_EMBD = 128
LANGUAGE_EMB_DIM = 896
BLOCK_SIZE = CONTEXT_LENGTH * 3
MODEL_TYPE = 'DT'

# World Model (O) — NO semantic embeddings
USE_SEMANTIC_WM = False
O_HIDDEN = 256
MC_DROPOUT = 0.2

# Sampling
TEMPERATURE = 1.5
N_CF = 3

# Self-play filtering
SIGMA_THRESHOLD = 2.0
LAMBDA_O = 0.5

# Optimizers
LR_PI = 6e-4
LR_O = 1e-3
BATCH_SIZE = 64
WEIGHT_DECAY = 0.1

# Training — Stage 1
STAGE1_EPOCHS = 100
SAVE_INTERVAL_EPOCHS = 10
LOG_INTERVAL_STEPS = 100

# Training — Stage 2
STAGE2_EPOCHS = 50
SELFPLAY_ITERATIONS = 1000
SEARCH_RADIUS = 2

# Evaluation
UNCERTAINTY_CUTOFF = 2.0
MC_SAMPLES = 10

# Confidence-gated semi-bootstrap (Stage 2)
SIGMA_THRESHOLD_SEARCH = 1.0   # <-- sigma1.0 variant
ALPHA_MIN = 0.3
ALPHA_DECAY_STEPS = 50000
CONFIDENCE_GATE = 0.5

# Paths
DATA_DIR = "/home/wangmeiyi/AuctionNet/medical/last_exp/data/v3"
