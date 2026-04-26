"""
utils/preprocessor.py

Functions to clean raw merged OHLCV data and resample it from 1-minute bars
to the timeframes required by the project (15min, 1H).

Pipeline order:
    load_raw → remove_weekends → remove_low_liquidity → handle_missing
    → resample_ohlcv → save_processed

Run this module directly to execute the full preprocessing pipeline for all
currency pairs and timeframes defined in config.py.
"""

import os

import pandas as pd

from config import (
    RAW_DIR,
    PROCESSED_DIR,
    START_DATE,
    END_DATE,
    CURRENCY_PAIRS,
    TIMEFRAMES,
)


# ---------------------------------------------------------------------------
# 1. LOAD RAW DATA
# ---------------------------------------------------------------------------

def load_raw(pair):
    """
    Load the merged raw CSV file for a given currency pair from data/raw/.

    The file is expected to be named <PAIR>_raw.csv (e.g. EURUSD_raw.csv)
    and was produced by utils/data_loader.py.  The DateTime index is parsed
    explicitly using the HistData format "%Y%m%d %H%M%S".

    Parameters
    ----------
    pair : str
        Currency pair identifier, e.g. "EURUSD" or "GBPUSD".

    Returns
    -------
    pd.DataFrame
        DataFrame with a DatetimeIndex and OHLCV columns.
    """
    filepath = os.path.join(RAW_DIR, f"{pair}_raw.csv")
    df = pd.read_csv(
        filepath,
        index_col="DateTime",
        parse_dates=True,
    )
    # Re-parse index explicitly to guarantee datetime dtype.
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df[df.index.notna()]
    print(f"[{pair}] Loaded {len(df):,} rows from {filepath}")
    return df


# ---------------------------------------------------------------------------
# 2. REMOVE WEEKENDS
# ---------------------------------------------------------------------------

def remove_weekends(df):
    """
    Remove all rows where the timestamp falls on a Saturday or Sunday.

    Forex markets are closed over the weekend.  Any weekend rows present
    in the raw data are artefacts and should be discarded before further
    processing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with weekend rows removed.
    """
    before = len(df)
    mask = df.index.dayofweek < 5          # 0=Mon … 4=Fri; 5=Sat, 6=Sun
    df = df[mask]
    removed = before - len(df)
    print(f"  remove_weekends: removed {removed:,} rows "
          f"({len(df):,} remaining)")
    return df


# ---------------------------------------------------------------------------
# 3. REMOVE LOW LIQUIDITY PERIODS
# ---------------------------------------------------------------------------

def remove_low_liquidity(df):
    """
    Remove known low-liquidity periods at the edges of each trading week.

    Two windows are excluded:
      - Friday 21:00–23:59 UTC  — market wind-down before weekend close.
      - Monday 00:00–01:00 UTC  — illiquid period just after market open.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex (weekends already removed).

    Returns
    -------
    pd.DataFrame
        DataFrame with low-liquidity rows removed.
    """
    before = len(df)

    is_friday  = df.index.dayofweek == 4   # Friday
    is_monday  = df.index.dayofweek == 0   # Monday

    hour   = df.index.hour
    minute = df.index.minute

    # Friday 21:00–23:59 UTC (>= 21:00)
    friday_wind_down = is_friday & (hour >= 21)

    # Monday 00:00–01:00 UTC (<= 01:00, inclusive of the 01:00 bar)
    monday_open = is_monday & (
        (hour == 0) | ((hour == 1) & (minute == 0))
    )

    keep = ~(friday_wind_down | monday_open)
    df = df[keep]

    removed = before - len(df)
    print(f"  remove_low_liquidity: removed {removed:,} rows "
          f"({len(df):,} remaining)")
    return df


# ---------------------------------------------------------------------------
# 4. HANDLE MISSING VALUES
# ---------------------------------------------------------------------------

def handle_missing(df):
    """
    Fill short gaps in the 1-minute series and drop bars that cannot be filled.

    Steps:
      1. Build a complete 1-minute DatetimeIndex spanning the data range,
         restricted to weekday hours outside the low-liquidity windows that
         were already removed.
      2. Reindex the DataFrame against this complete index so gaps appear
         as NaN rows.
      3. Forward-fill up to 3 consecutive missing bars (limit=3).
      4. Drop any remaining bars where Close is still NaN (gaps > 3 bars).
      5. Report filled and dropped counts.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned 1-minute DataFrame (weekends and low-liquidity rows removed).

    Returns
    -------
    pd.DataFrame
        Gap-filled DataFrame with no NaN values in the Close column.
    """
    # Build a complete 1-minute grid for the same date range.
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="1min",
    )

    # Restrict to weekdays only (Mon–Fri).
    full_index = full_index[full_index.dayofweek < 5]

    # Exclude the same low-liquidity windows removed earlier.
    is_friday      = full_index.dayofweek == 4
    is_monday      = full_index.dayofweek == 0
    hour           = full_index.hour
    minute         = full_index.minute

    friday_wind_down = is_friday & (hour >= 21)
    monday_open      = is_monday & ((hour == 0) | ((hour == 1) & (minute == 0)))

    full_index = full_index[~(friday_wind_down | monday_open)]

    # Reindex to expose gaps as NaN rows.
    df_reindexed = df.reindex(full_index)

    n_gaps = df_reindexed["Close"].isna().sum()

    # Forward-fill gaps of up to 3 consecutive bars.
    df_filled = df_reindexed.ffill(limit=3)

    n_filled  = n_gaps - df_filled["Close"].isna().sum()
    n_dropped = df_filled["Close"].isna().sum()

    # Drop bars that could not be filled (gaps longer than 3 bars).
    df_filled = df_filled.dropna(subset=["Close"])

    print(f"  handle_missing: {n_gaps:,} gap bars found | "
          f"{n_filled:,} forward-filled | "
          f"{n_dropped:,} dropped (gaps > 3 bars) | "
          f"{len(df_filled):,} rows remaining")

    return df_filled


# ---------------------------------------------------------------------------
# 5. RESAMPLE TO TIMEFRAMES
# ---------------------------------------------------------------------------

def resample_ohlcv(df, timeframe):
    """
    Resample a 1-minute OHLCV DataFrame to a coarser timeframe.

    Aggregation rules:
        Open   → first value in the bar
        High   → maximum value in the bar
        Low    → minimum value in the bar
        Close  → last value in the bar
        Volume → sum of all 1-minute volumes in the bar

    Incomplete bars (those containing NaN after resampling) are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned 1-minute DataFrame with OHLCV columns.
    timeframe : str
        Target resampling frequency accepted by pandas, e.g. "15min" or "1H".

    Returns
    -------
    pd.DataFrame
        Resampled OHLCV DataFrame at the requested timeframe.
    """
    resampled = df.resample(timeframe).agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    })

    before = len(resampled)
    resampled = resampled.dropna()
    dropped = before - len(resampled)

    print(f"  resample_ohlcv [{timeframe}]: {len(resampled):,} bars "
          f"({dropped:,} incomplete bars dropped)")

    return resampled


# ---------------------------------------------------------------------------
# 6. SAVE PROCESSED DATA
# ---------------------------------------------------------------------------

def save_processed(df, pair, timeframe):
    """
    Save a resampled OHLCV DataFrame to data/processed/.

    The file is named <PAIR>_<timeframe>.csv, e.g. EURUSD_15min.csv.

    Parameters
    ----------
    df : pd.DataFrame
        Resampled DataFrame to save.
    pair : str
        Currency pair identifier, e.g. "EURUSD".
    timeframe : str
        Timeframe label used in the filename, e.g. "15min" or "1H".
    """
    filename = f"{pair}_{timeframe}.csv"
    out_path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(out_path)
    print(f"  Saved: {out_path}  ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# 7. MAIN EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    summary = []   # Collect (pair, timeframe, n_rows) for final table.

    for pair in CURRENCY_PAIRS:
        print(f"\n{'='*50}")
        print(f"Processing: {pair}")
        print(f"{'='*50}")

        # Load
        df = load_raw(pair)

        # Clean
        df = remove_weekends(df)
        df = remove_low_liquidity(df)
        df = handle_missing(df)

        # Resample and save for each timeframe
        for tf in TIMEFRAMES:
            df_tf = resample_ohlcv(df, tf)
            save_processed(df_tf, pair, tf)
            summary.append((pair, tf, len(df_tf)))

    # Final summary table
    print(f"\n{'='*50}")
    print(f"{'SUMMARY':^50}")
    print(f"{'='*50}")
    print(f"  {'Pair':<10} {'Timeframe':<12} {'Bars':>10}")
    print(f"  {'-'*34}")
    for pair, tf, n_rows in summary:
        print(f"  {pair:<10} {tf:<12} {n_rows:>10,}")
    print(f"{'='*50}")
