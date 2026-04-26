"""
strategies/rsi_threshold.py

RSI Threshold strategy.

Signal rule:
    +1  when RSI14 < RSI_OVERSOLD   (oversold — buy signal)
    -1  when RSI14 > RSI_OVERBOUGHT (overbought — sell signal)
     0  otherwise                   (neutral — no position)

Neutral bars (0) are excluded from accuracy evaluation so that only
bars where the strategy actually commits to a direction are assessed.

Parameters are imported from config.py:
    RSI_PERIOD_STRAT    RSI look-back period (14)
    RSI_OVERSOLD        oversold threshold  (30)
    RSI_OVERBOUGHT      overbought threshold (70)
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

from config import RSI_PERIOD_STRAT, RSI_OVERSOLD, RSI_OVERBOUGHT, LATENCY_ITERATIONS


# ---------------------------------------------------------------------------
# 1. SIGNAL GENERATION
# ---------------------------------------------------------------------------

def generate_signals(df):
    """
    Generate RSI threshold signals from a pre-computed RSI14 column.

    Requires the DataFrame to contain an 'RSI14' column, produced by
    utils/preprocessor.py during feature engineering.

    Signal rule:
        +1  RSI14 < RSI_OVERSOLD   (oversold zone)
        -1  RSI14 > RSI_OVERBOUGHT (overbought zone)
         0  RSI_OVERSOLD <= RSI14 <= RSI_OVERBOUGHT (neutral)

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered OHLCV DataFrame with an RSI14 column.

    Returns
    -------
    pd.Series
        Integer signal series (+1 / 0 / -1) aligned to df.index.
    """
    signals = np.where(
        df["RSI14"] < RSI_OVERSOLD,  1,
        np.where(df["RSI14"] > RSI_OVERBOUGHT, -1, 0)
    )
    return pd.Series(signals, index=df.index, name="signal", dtype=int)


# ---------------------------------------------------------------------------
# 2. FLOPS CALCULATION
# ---------------------------------------------------------------------------

def compute_flops():
    """
    Return the analytic FLOPs count for one RSI signal evaluation.

    Formula (from paper):
        FLOPs = 4 * n + 12

    where n = RSI_PERIOD_STRAT (14).

    Returns
    -------
    int
        FLOPs per signal evaluation.
    """
    return 4 * RSI_PERIOD_STRAT + 12


# ---------------------------------------------------------------------------
# 3. LATENCY MEASUREMENT
# ---------------------------------------------------------------------------

def measure_latency(df):
    """
    Measure the average single-step signal latency for RSI Threshold.

    Timing reflects one complete Wilder RSI update step:
        1. Apply Wilder's EWM formula to update avg_gain and avg_loss.
        2. Compute RS = avg_gain / avg_loss.
        3. Compute RSI = 100 − 100 / (1 + RS).
        4. Apply oversold / overbought threshold comparison.

    Pre-computed scalar inputs (avg_gain, avg_loss, new_gain, new_loss)
    are extracted once before the timing loop to isolate the true
    single-step inference cost.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame used to derive pre-computed RSI state scalars.

    Returns
    -------
    float
        Mean single-step latency in microseconds (µs).
    """
    alpha  = 1.0 / RSI_PERIOD_STRAT
    delta  = df["Close"].diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)

    avg_gain = float(
        gain.ewm(alpha=alpha, min_periods=RSI_PERIOD_STRAT, adjust=False)
            .mean().iloc[-2]
    )
    avg_loss = float(
        loss.ewm(alpha=alpha, min_periods=RSI_PERIOD_STRAT, adjust=False)
            .mean().iloc[-2]
    )
    new_gain = float(gain.iloc[-1])
    new_loss = float(loss.iloc[-1])

    start = time.perf_counter()
    for _ in range(LATENCY_ITERATIONS):
        ag  = (1.0 - alpha) * avg_gain + alpha * new_gain
        al  = (1.0 - alpha) * avg_loss + alpha * new_loss
        rs  = ag / al if al != 0.0 else float("inf")
        rsi = 100.0 - (100.0 / (1.0 + rs))
        _sig = 1 if rsi < RSI_OVERSOLD else (-1 if rsi > RSI_OVERBOUGHT else 0)
    elapsed = time.perf_counter() - start
    return (elapsed / LATENCY_ITERATIONS) * 1_000_000   # convert to µs


# ---------------------------------------------------------------------------
# 4. EVALUATE
# ---------------------------------------------------------------------------

def evaluate(df, test_index):
    """
    Evaluate the RSI threshold strategy on the held-out test set.

    Steps:
        1. Generate signals on the full DataFrame.
        2. Filter signals to test_index.
        3. Align signals with the 'Target' column and drop NaN rows.
        4. Exclude neutral signals (0) — only directional bars are scored.
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
        n_neutral      : int       — bars excluded as neutral
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
