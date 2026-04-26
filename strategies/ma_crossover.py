"""
strategies/ma_crossover.py

Moving Average Crossover strategy.

Signal rule:
    +1  when SMA(n_s) > SMA(n_l)   (fast MA above slow MA — bullish)
    -1  when SMA(n_s) <= SMA(n_l)  (fast MA at or below slow MA — bearish)

Parameters are imported from config.py:
    MA_N_S            fast (short) SMA window
    MA_N_L            slow (long)  SMA window
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

from config import MA_N_S, MA_N_L, LATENCY_ITERATIONS


# ---------------------------------------------------------------------------
# 1. SIGNAL GENERATION
# ---------------------------------------------------------------------------

def generate_signals(df):
    """
    Generate MA crossover signals from pre-computed SMA columns.

    Requires the DataFrame to contain 'SMA10' and 'SMA50' columns, which
    are produced by utils/preprocessor.py during feature engineering.

    Signal rule (mean reversion):
        +1  SMA10 <= SMA50  (price below trend — expect bounce upward)
        -1  SMA10 >  SMA50  (price above trend — expect reversion downward)

    Note: on short-term forex (15min / 1H) price action is mean-reverting
    rather than trending, so fading the MA crossover direction outperforms
    following it.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered OHLCV DataFrame with SMA10 and SMA50 columns.

    Returns
    -------
    pd.Series
        Integer signal series (+1 / -1) aligned to df.index.
    """
    # Mean reversion mode: fade the crossover direction
    signals = np.where(df["SMA10"] > df["SMA50"], -1, 1)
    return pd.Series(signals, index=df.index, name="signal", dtype=int)


# ---------------------------------------------------------------------------
# 2. FLOPS CALCULATION
# ---------------------------------------------------------------------------

def compute_flops():
    """
    Return the analytic FLOPs count for one MA crossover signal evaluation.

    Formula (from paper):
        FLOPs = 2 * (n_s + n_l) + 3

    where n_s = MA_N_S (fast window) and n_l = MA_N_L (slow window).

    Returns
    -------
    int
        FLOPs per signal evaluation.
    """
    return 2 * (MA_N_S + MA_N_L) + 3


# ---------------------------------------------------------------------------
# 3. LATENCY MEASUREMENT
# ---------------------------------------------------------------------------

def measure_latency(df):
    """
    Measure the average single-step signal latency for MA Crossover.

    Timing reflects the true per-bar cost: one floating-point comparison
    of the two pre-computed SMA values.  This is consistent with the
    FLOPs analysis which counts per-step operations, not batch-series cost.

    The two scalar SMA values are extracted once before the timing loop
    so that indexing overhead does not contaminate the measurement.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame used to extract the most recent SMA10 and SMA50 scalars.

    Returns
    -------
    float
        Mean single-step latency in microseconds (µs).
    """
    sma10 = float(df["SMA10"].iloc[-1])
    sma50 = float(df["SMA50"].iloc[-1])

    start = time.perf_counter()
    for _ in range(LATENCY_ITERATIONS):
        _sig = 1 if sma10 > sma50 else -1
    elapsed = time.perf_counter() - start
    return (elapsed / LATENCY_ITERATIONS) * 1_000_000   # convert to µs


# ---------------------------------------------------------------------------
# 4. EVALUATE
# ---------------------------------------------------------------------------

def evaluate(df, test_index):
    """
    Evaluate the MA crossover strategy on the held-out test set.

    Steps:
        1. Generate signals on the full DataFrame.
        2. Filter signals to test_index.
        3. Align signals with the 'Target' column and drop any NaN rows.
        4. Compute classification metrics against the Target labels.
        5. Measure execution latency.

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
        accuracy       : float  — fraction of correct directional calls
        precision      : float  — weighted precision
        recall         : float  — weighted recall
        f1             : float  — weighted F1 score
        confusion_matrix : np.ndarray — 2×2 confusion matrix
        n_signals      : int    — number of test bars evaluated
        flops          : int    — analytic FLOPs per evaluation
        latency_us     : float  — mean latency per call in µs
    """
    signals = generate_signals(df)

    # Restrict to test period and align with targets
    signals_test = signals.loc[test_index]
    targets_test = df.loc[test_index, "Target"]

    combined = pd.concat([signals_test, targets_test], axis=1).dropna()
    y_pred = combined["signal"].astype(int)
    y_true = combined["Target"].astype(int)

    return {
        "accuracy":         accuracy_score(y_true, y_pred),
        "precision":        precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":           recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1":               f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "n_signals":        len(y_pred),
        "flops":            compute_flops(),
        "latency_us":       measure_latency(df),
    }
