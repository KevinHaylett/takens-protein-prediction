"""
config.py — Takens-Based Transformer: Protein Structure Prediction
==================================================================
This is the only file you need to edit before running the system.
Set your folder paths here and all scripts will use them automatically.
"""

# ============================================================
# YOUR PATHS — edit these five lines
# ============================================================

PDB_DIR        = "data/pdb_files"       # raw PDB files from RCSB
CSV_DIR        = "data/csv_files"        # converted CSV files
TRAINING_DIR   = "data/processed_data"  # prepared training tensors
CHECKPOINT_DIR = "data/checkpoints"     # saved model weights
OUTPUT_DIR     = "data/predictions"     # inference outputs

# ============================================================
# MODEL SETTINGS — safe to leave as defaults
# ============================================================

MAX_SEQ_LEN  = 256
EMBED_DIM    = 128
HIDDEN_DIM   = 512
NUM_LAYERS   = 6
MAX_DELAY    = 128
DROPOUT      = 0.1

# ============================================================
# TRAINING SETTINGS — safe to leave as defaults
# ============================================================

BATCH_SIZE    = 4
NUM_EPOCHS    = 250
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 0.01
GRADIENT_CLIP = 1.0
VAL_SPLIT     = 0.1
RANDOM_SEED   = 42
LOG_INTERVAL  = 1

# ============================================================
# SYSTEM SETTINGS
# ============================================================

DEVICE      = "cpu"    # change to "cuda" if you have a GPU
NUM_WORKERS = 0

# ============================================================
# INFERENCE SETTINGS
# ============================================================

SHOW_PLOTS       = True   # display plots interactively
SAVE_PREDICTIONS = True   # save PDB and PNG files to OUTPUT_DIR
VERBOSE          = True   # print detailed evaluation output
