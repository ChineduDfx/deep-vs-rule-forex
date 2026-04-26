import os
from pathlib import Path

# =============================================================================
# PATH SETTINGS
# Controls where data is read from and where all outputs are saved.
# BASE_DIR is derived from config.py's own location so the project is
# portable across Windows, macOS, Linux, and Google Colab with no edits.
# =============================================================================

BASE_DIR = str(Path(__file__).resolve().parent)

DATASET_DIR   = os.path.join(BASE_DIR, "dataset")
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SPLITS_DIR    = os.path.join(BASE_DIR, "data", "splits")
TABLES_DIR    = os.path.join(BASE_DIR, "results", "tables")
FIGURES_DIR   = os.path.join(BASE_DIR, "results", "figures")
MODELS_DIR    = os.path.join(BASE_DIR, "results", "models")

# =============================================================================
# DATA SETTINGS
# Controls which currency pairs and timeframes are used, the date range,
# train/val/test split ratios, sequence length, and random seed.
# =============================================================================

CURRENCY_PAIRS  = ["EURUSD", "GBPUSD"]
TIMEFRAMES      = ["15min", "1h"]

START_DATE      = "2020-01-01"
END_DATE        = "2024-12-31"

TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15

SEQUENCE_LENGTH = 60   # Input window length for LSTM and GRU (timesteps)
RANDOM_SEED     = 42

# =============================================================================
# FEATURE SETTINGS
# Controls which technical indicators are computed and the input dimensionality
# fed into each model.
# =============================================================================

SMA_FAST_PERIOD = 10
SMA_SLOW_PERIOD = 50
RSI_PERIOD      = 14
ATR_PERIOD      = 14
DONCHIAN_PERIOD = 20

INPUT_DIM = 10   # m: number of input features per timestep

FEATURE_NAMES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA10",
    "SMA50",
    "RSI14",
    "ATR14",
    "Donchian_Width",
]

# =============================================================================
# RULE-BASED STRATEGY SETTINGS
# Controls the parameters for each rule-based strategy and the number of
# iterations used when measuring execution latency.
# =============================================================================

# MA Crossover
MA_N_S = 10   # Fast (short) window
MA_N_L = 50   # Slow (long) window

# RSI Threshold
RSI_PERIOD_STRAT = 14
RSI_OVERSOLD     = 30
RSI_OVERBOUGHT   = 70

# Volatility Breakout
VB_K          = 20   # Donchian channel look-back period
VB_ATR_PERIOD = 14

# Number of iterations for latency benchmarking
LATENCY_ITERATIONS = 10_000

# =============================================================================
# DEEP LEARNING SETTINGS
# Controls model architecture, regularisation, optimisation, and training
# stopping criteria shared by both the LSTM and GRU models.
# =============================================================================

HIDDEN_DIM = 64    # d: number of hidden units per recurrent layer
DROPOUT    = 0.2
LR         = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE   = 10    # Early-stopping patience (epochs without val improvement)

LOSS      = "binary_crossentropy"
OPTIMIZER = "Adam"

# =============================================================================
# EVALUATION SETTINGS
# Controls the significance threshold for statistical tests and the analytic
# FLOPs formulas used to estimate computational cost for each model.
# =============================================================================

SIGNIFICANCE_THRESHOLD = 0.05

# Analytic FLOPs expressions (evaluated with the parameter values above)
# MA Crossover:        2*(n_s + n_l) + 3
FLOPS_MA = 2 * (MA_N_S + MA_N_L) + 3

# RSI:                 4*n + 12
FLOPS_RSI = 4 * RSI_PERIOD_STRAT + 12

# Volatility Breakout: 2*k + 5
FLOPS_VB = 2 * VB_K + 5

# LSTM:                8*d^2 + 8*d*m + 12*d
FLOPS_LSTM = 8 * HIDDEN_DIM**2 + 8 * HIDDEN_DIM * INPUT_DIM + 12 * HIDDEN_DIM

# GRU:                 6*d^2 + 6*d*m + 9*d
FLOPS_GRU = 6 * HIDDEN_DIM**2 + 6 * HIDDEN_DIM * INPUT_DIM + 9 * HIDDEN_DIM
