"""
strategies/volatility_breakout.py

Volatility Breakout strategy using Donchian channels.

Signal rule (mean reversion):
    +1  when Close < previous bar's Lower channel  (price below band — expect bounce)
    -1  when Close > previous bar's Upper channel  (price above band — expect reversal)
     0  otherwise                                  (price inside channel — neutral)

The channel boundaries are shifted by one bar to avoid lookahead bias:
the signal at time t is compared against the channel computed up to t-1.

Neutral bars (0) are excluded from accuracy evaluation so that only
bars where the strategy actually commits to a direction are assessed.

Parameters are imported from config.py:
    VB_K                Donchian channel look-back period (20)
    LATENCY_ITERATIONS  number of repetitions for latency benchmarking
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config import VB_K, LATENCY_ITERATIONS


# ---------------------------------------------------------------------------
# 1. SIGNAL GENERATION
# ---------------------------------------------------------------------------

def generate_signals(df):
    """
    Generate Donchian channel breakout signals from OHLCV data.

    Computes the k-period Donchian channel (rolling High max / Low min)
    and shifts both boundaries by one bar so that the signal at time t
    uses only information available at t-1, eliminating lookahead bias.

    Signal rule (mean reversion):
        +1  Close(t) < Lower(t-1)   (price below lower band — expect bounce upward)
        -1  Close(t) > Upper(t-1)   (price above upper band — expect reversal downward)
         0  otherwise               (price inside channel — neutral)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with 'High', 'Low', and 'Close' columns.

    Returns
    -------
    pd.Series
        Integer signal series (+1 / 0 / -1) aligned to df.index.
    """
    upper_shifted = df["High"].rolling(window=VB_K).max().shift(1)
    lower_shifted = df["Low"].rolling(window=VB_K).min().shift(1)

    # Mean reversion mode: signals fade the breakout,
    # anticipating price return toward the channel midpoint
    signals = np.where(
        df["Close"] < lower_shifted,  1,
        np.where(df["Close"] > upper_shifted, -1, 0)
    )
    sig_series = pd.Series(signals, index=df.index, name="signal", dtype=int)

    # Distribution check — print once per call to confirm correct generation
    counts = sig_series.value_counts().sort_index()
    print(f"  VB signal distribution:  "
          f"+1={counts.get(1, 0):,}  "
          f" 0={counts.get(0, 0):,}  "
          f"-1={counts.get(-1, 0):,}")

    return sig_series


# ---------------------------------------------------------------------------
# 2. FLOPS CALCULATION
# ---------------------------------------------------------------------------

def compute_flops():
    """
    Return the analytic FLOPs count for one volatility breakout evaluation.

    Formula (from paper):
        FLOPs = 2 * k + 5

    where k = VB_K (Donchian channel look-back period).

    Returns
    -------
    int
        FLOPs per signal evaluation.
    """
    return 2 * VB_K + 5


# ---------------------------------------------------------------------------
# 3. LATENCY MEASUREMENT
# ---------------------------------------------------------------------------

def measure_latency(df):
    """
    Measure the average single-step signal latency for Volatility Breakout.

    Timing reflects one complete per-bar inference step:
        1. Find max(High) and min(Low) over the pre-computed VB_K window.
        2. Compare the current Close against the pre-computed boundaries.

    The VB_K-length High and Low windows and the current Close scalar are
    extracted once before the timing loop, consistent with the incremental
    deployment model where boundaries are maintained as a rolling buffer
    rather than recomputed from the full series each bar.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame used to extract the most recent VB_K-bar window.

    Returns
    -------
    float
        Mean single-step latency in microseconds (µs).
    """
    high_window = df["High"].values[-(VB_K + 1):-1]   # VB_K bars ending at t-1
    low_window  = df["Low"].values[-(VB_K + 1):-1]
    close_val   = float(df["Close"].iloc[-1])

    start = time.perf_counter()
    for _ in range(LATENCY_ITERATIONS):
        upper = float(high_window.max())
        lower = float(low_window.min())
        _sig  = 1 if close_val > upper else (-1 if close_val < lower else 0)
    elapsed = time.perf_counter() - start
    return (elapsed / LATENCY_ITERATIONS) * 1_000_000   # convert to µs


# ---------------------------------------------------------------------------
# 4. EVALUATE
# ---------------------------------------------------------------------------

def evaluate(df, test_index):
    """
    Evaluate the volatility breakout strategy on the held-out test set.

    Steps:
        1. Generate signals on the full DataFrame.
        2. Filter signals to test_index.
        3. Align signals with the 'Target' column and drop NaN rows.
        4. Exclude neutral signals (0) — only breakout bars are scored.
        5. Compute classification metrics against the Target labels.
        6. Measure execution latency.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature-engineered DataFrame including a 'Target' column
        (+1 / -1 directional labels).
    test_index : pd.DatetimeIndex
        Index of the test-set rows to evaluate on.

    Returns
    -------
    dict
        accuracy       : float     — fraction of correct directional calls
        precision      : float     — weighted precision
        recall         : float     — weighted recall
        f1             : float     — weighted F1 score
        confusion_matrix : np.ndarray — 2×2 confusion matrix
        n_signals      : int       — directional bars evaluated (non-neutral)
        n_neutral      : int       — bars excluded as neutral (inside channel)
        flops          : int       — analytic FLOPs per evaluation
        latency_us     : float     — mean latency per call in µs
    """
    signals = generate_signals(df)

    # Restrict to test period and align with targets
    signals_test = signals.loc[test_index]
    targets_test = df.loc[test_index, "Target"]

    combined = pd.concat([signals_test, targets_test], axis=1).dropna()

    # Separate neutral bars before scoring
    n_neutral = int((combined["signal"] == 0).sum())
    active    = combined[combined["signal"] != 0]

    y_pred = active["signal"].astype(int)
    y_true = active["Target"].astype(int)

    return {
        "accuracy":         accuracy_score(y_true, y_pred),
        "precision":        precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":           recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1":               f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "n_signals":        len(y_pred),
        "n_neutral":        n_neutral,
        "flops":            compute_flops(),
        "latency_us":       measure_latency(df),
    }
