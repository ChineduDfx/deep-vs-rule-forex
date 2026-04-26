"""
utils/data_loader.py

Functions to extract, load, and merge the raw HistData CSV dataset.
Run this module directly to perform the full extraction and merge pipeline.
"""

import os
import zipfile
import glob

import pandas as pd

from config import (
    DATASET_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    START_DATE,
    END_DATE,
    CURRENCY_PAIRS,
)


# ---------------------------------------------------------------------------
# 1. EXTRACT ZIP FILES
# ---------------------------------------------------------------------------

def extract_zips(dataset_dir, raw_dir):
    """
    Extract all zip files found in dataset_dir into raw_dir.

    Parameters
    ----------
    dataset_dir : str
        Path to the folder containing the downloaded zip files.
    raw_dir : str
        Destination folder where the contents will be extracted.
    """
    files = os.listdir(dataset_dir)

    for filename in sorted(files):
        if not filename.endswith(".zip"):
            continue

        zip_path = os.path.join(dataset_dir, filename)
        print(f"Extracting: {filename}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)

    print("Extraction complete.\n")


# ---------------------------------------------------------------------------
# 2. LOAD AND PARSE RAW CSV FILES
# ---------------------------------------------------------------------------

def load_raw_csv(filepath):
    """
    Read a single HistData CSV file and return a parsed DataFrame.

    HistData ASCII files:
      - Use a semicolon (;) as the delimiter.
      - Have no header row.
      - DateTime column format: "20200101 170000"  ("%Y%m%d %H%M%S").
      - Columns: DateTime, Open, High, Low, Close, Volume

    After reading, the DateTime column is explicitly converted with
    pd.to_datetime(errors='coerce') so any unparseable values become NaT.
    Rows with NaT index are then dropped and counted.

    Parameters
    ----------
    filepath : str
        Absolute path to the CSV file to load.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by datetime with OHLCV columns.
    """
    # First 5 rows of the first file are printed to confirm the format.
    # (Controlled by the caller in __main__ — see below.)
    df = pd.read_csv(
        filepath,
        sep=";",
        header=None,
        names=["DateTime", "Open", "High", "Low", "Close", "Volume"],
    )

    # Explicitly parse with the known HistData format string.
    df["DateTime"] = pd.to_datetime(
        df["DateTime"], format="%Y%m%d %H%M%S", errors="coerce"
    )

    # Report and drop rows where datetime parsing failed.
    n_nat = df["DateTime"].isna().sum()
    if n_nat > 0:
        print(f"  INFO: {n_nat} row(s) dropped due to failed datetime parsing in {os.path.basename(filepath)}")
        df = df.dropna(subset=["DateTime"])

    df = df.set_index("DateTime")
    return df


# ---------------------------------------------------------------------------
# 3. MERGE YEARLY FILES
# ---------------------------------------------------------------------------

def merge_pair(raw_dir, pair):
    """
    Find, load, and merge all yearly CSV files for a given currency pair.

    Steps performed:
      - Glob all CSVs in raw_dir whose name contains the pair ticker.
      - Load each file with load_raw_csv() and concatenate into one DataFrame.
      - Sort by datetime index.
      - Filter to the [START_DATE, END_DATE] window defined in config.py.
      - Detect and drop duplicate timestamps (reports count before dropping).
      - Print total row count and date range of the merged result.

    Parameters
    ----------
    raw_dir : str
        Path to the folder containing the extracted CSV files.
    pair : str
        Currency pair identifier, e.g. "EURUSD" or "GBPUSD".

    Returns
    -------
    pd.DataFrame
        Merged, cleaned DataFrame for the requested pair.
    """
    # Match only the original HistData files, not the *_raw.csv files we generate.
    pattern = os.path.join(raw_dir, f"DAT_ASCII_{pair}_M1_*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found for pair '{pair}' in {raw_dir}. "
            "Run extract_zips() first."
        )

    frames = []
    for fp in csv_files:
        print(f"  Loading: {os.path.basename(fp)}")
        frames.append(load_raw_csv(fp))

    df = pd.concat(frames)

    # Ensure the full merged index is datetime type before any operations.
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()

    # Filter to configured date range
    df = df.loc[START_DATE:END_DATE]

    # Check and remove duplicate timestamps
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        print(f"  WARNING: {n_dupes} duplicate timestamp(s) found — dropping duplicates.")
        df = df[~df.index.duplicated(keep="first")]

    print(
        f"\n{pair} merged: {len(df):,} rows | "
        f"{df.index.min().date()} to {df.index.max().date()}\n"
    )

    return df


# ---------------------------------------------------------------------------
# 4. SAVE MERGED FILES
# ---------------------------------------------------------------------------

def save_merged(df, raw_dir, pair):
    """
    Save a merged DataFrame to data/raw/ as <PAIR>_raw.csv.

    Parameters
    ----------
    df : pd.DataFrame
        The merged DataFrame to save (datetime index, OHLCV columns).
    raw_dir : str
        Destination directory (data/raw/).
    pair : str
        Currency pair identifier used to name the output file,
        e.g. "EURUSD" → "EURUSD_raw.csv".
    """
    filename = f"{pair}_raw.csv"
    out_path = os.path.join(raw_dir, filename)
    df.to_csv(out_path)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 5. MAIN EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Step 1 — extract all zip archives into data/raw/
    extract_zips(DATASET_DIR, RAW_DIR)

    # Inspect the raw format of the first available CSV so the datetime
    # layout is visible before any parsing takes place.
    import glob as _glob
    _sample_files = sorted(_glob.glob(os.path.join(RAW_DIR, "DAT_ASCII_EURUSD_M1_*.csv")))
    if _sample_files:
        print("--- Raw file preview (first 5 rows, no parsing) ---")
        import csv as _csv
        with open(_sample_files[0], newline="") as _f:
            for i, row in enumerate(_csv.reader(_f, delimiter=";")):
                if i >= 5:
                    break
                print(row)
        print()

    summary = {}

    # Step 2 — merge and save each currency pair
    for pair in CURRENCY_PAIRS:
        print(f"--- {pair} ---")
        df = merge_pair(RAW_DIR, pair)
        save_merged(df, RAW_DIR, pair)
        summary[pair] = len(df)

    # Step 3 — print final summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    for pair, n_rows in summary.items():
        print(f"  {pair}: {n_rows:,} rows")
    print("=" * 40)
