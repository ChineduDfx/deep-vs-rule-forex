# Evaluating Rule-Based Trading Strategies Against Deep Learning Benchmarks for Short-Term Forex Directional Prediction

This is a research project comparing three rule-based trading strategies against LSTM and GRU deep learning models for next-bar directional prediction on EURUSD and GBPUSD forex data.

---

**Author:**  
Ogbogu Chinedu Francis (B00103795)  

**Affiliation:**  
School of Computing, Engineering & Intelligent Systems  
Ulster University  
Belfast, United Kingdom  

**Email:**  
ogbogu-c1@ulster.ac.uk  


---

## Research Question

Do classical rule-based strategies (MA Crossover, RSI Threshold, Volatility Breakout) achieve statistically significant directional accuracy on short-term forex data, and how do they compare to LSTM and GRU models in terms of accuracy and computational efficiency?

---

## Project Structure

```
FX/
├── dataset/                    # Raw zip files from HistData.com (not tracked)
├── data/
│   ├── raw/                    # Extracted CSVs + merged *_raw.csv files
│   ├── processed/              # Cleaned, resampled, featured datasets
│   └── splits/                 # Train / validation / test CSVs
├── strategies/
│   ├── ma_crossover.py         # Moving Average Crossover
│   ├── rsi_threshold.py        # RSI Threshold
│   └── volatility_breakout.py  # Donchian Channel Volatility Breakout
├── models/
│   ├── lstm_model.py           # LSTM model definition
│   └── gru_model.py            # GRU model definition
├── evaluation/
│   ├── accuracy_metrics.py     # Classification metrics
│   └── computational_metrics.py # FLOPs and latency
├── utils/
│   ├── data_loader.py          # Zip extraction and CSV merging
│   └── preprocessor.py         # Cleaning, resampling, feature engineering
├── notebooks/
│   ├── preliminary_analysis.ipynb   # EDA, preprocessing, feature engineering
│   ├── strategy_evaluation.ipynb    # Rule-based strategy evaluation
│   └── deep_learning.ipynb          # LSTM and GRU training and evaluation
├── results/
│   ├── tables/                 # CSV result tables
│   ├── figures/                # All plots and charts
│   └── models/                 # Saved model weights
├── config.py                   # All project settings in one place
├── setup.py                    # Creates folder structure on first run
├── main.py                     # Master script
└── requirements.txt            # Python dependencies
```

---

## Data

| Property | Value |
|---|---|
| Source | [HistData.com](https://www.histdata.com) — ASCII 1-minute bars |
| Pairs | EURUSD, GBPUSD |
| Period | January 2020 – December 2024 |
| Raw resolution | 1-minute OHLCV |
| Resampled to | 15-minute, 1-hour |
| Split | 70% train / 15% validation / 15% test (chronological, no shuffle) |

**Preprocessing steps applied:**
1. Weekend rows removed (markets closed Saturday–Sunday)
2. Low-liquidity periods removed (Friday 21:00–23:59 UTC, Monday 00:00–01:00 UTC)
3. Gaps ≤ 3 consecutive bars forward-filled; longer gaps dropped

---

## Features

Ten features are computed per time step (input dimension m = 10):

| # | Feature | Description |
|---|---|---|
| 1 | Open | Raw open price |
| 2 | High | Raw high price |
| 3 | Low | Raw low price |
| 4 | Close | Raw close price |
| 5 | Volume | Tick volume |
| 6 | SMA10 | 10-period simple moving average |
| 7 | SMA50 | 50-period simple moving average |
| 8 | RSI14 | 14-period RSI (Wilder smoothing) |
| 9 | ATR14 | 14-period Average True Range |
| 10 | Donchian_Width | 20-period channel width (High_max − Low_min) |

**Target variable:** Binary directional label D(t) = +1 if Close(t+1) > Close(t), −1 otherwise.

---

## Strategies

All three strategies operate in **mean-reversion mode**, which outperforms trend-following on short-term forex data.

### Moving Average Crossover
- **Signal:** −1 when SMA10 > SMA50, +1 when SMA10 ≤ SMA50
- **FLOPs:** 2(n_s + n_l) + 3 = **123**
- Always generates a signal (no neutral bars)

### RSI Threshold
- **Signal:** +1 when RSI14 < 30 (oversold), −1 when RSI14 > 70 (overbought), 0 otherwise
- **FLOPs:** 4n + 12 = **68**
- Neutral bars (~87–92% of all bars) excluded from accuracy evaluation

### Volatility Breakout (Donchian Channel)
- **Signal:** +1 when Close < Lower_band(t−1), −1 when Close > Upper_band(t−1), 0 otherwise
- **FLOPs:** 2k + 5 = **45**
- Channel boundaries lagged by 1 bar to eliminate lookahead bias
- Neutral bars (~88% of all bars) excluded from accuracy evaluation

---

## Models

Both recurrent models share the same architecture settings:

| Hyperparameter | Value |
|---|---|
| Hidden units (d) | 64 |
| Dropout | 0.2 |
| Sequence length | 60 bars |
| Learning rate | 0.001 |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping patience | 10 |
| Loss | Binary crossentropy |
| Optimiser | Adam |

**FLOPs per inference step:**
- LSTM: 8d² + 8dm + 12d = **34,624**
- GRU:  6d² + 6dm + 9d = **25,992**

---

## Notebooks

Run in this order:

| Notebook | Purpose |
|---|---|
| `preliminary_analysis.ipynb` | Data extraction, cleaning, EDA, feature engineering, splits |
| `strategy_evaluation.ipynb` | Rule-based strategy evaluation, significance testing |
| `deep_learning.ipynb` | LSTM and GRU training, evaluation, final comparison |

---

## Setup

### Local

```bash
pip install -r requirements.txt
python setup.py          # creates all folders
python utils/data_loader.py    # extracts zips and merges CSVs
python utils/preprocessor.py   # cleans and resamples data
```

Then open the notebooks in order.

### Google Colab

1. Upload the `FX/` folder to Google Drive
2. Open any notebook in Colab
3. Cell 2 detects Colab automatically, mounts Google Drive, and sets all paths
4. Run all cells

---

## Configuration

All settings are centralised in `config.py`. `BASE_DIR` is derived from the file's own location using `pathlib` — no hardcoded paths, works on Windows, macOS, Linux, and Colab without edits.

Key settings to be aware of:

```python
CURRENCY_PAIRS  = ["EURUSD", "GBPUSD"]
TIMEFRAMES      = ["15min", "1h"]
START_DATE      = "2020-01-01"
END_DATE        = "2024-12-31"
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15
SEQUENCE_LENGTH = 60
RANDOM_SEED     = 42
```

---

## Dependencies

```
pandas>=2.2.0
numpy>=1.26.0
scipy>=1.13.0
tensorflow>=2.16.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
tqdm>=4.66.0
openpyxl>=3.1.0
```

---

## Results Location

| Output | Path |
|---|---|
| Accuracy tables | `results/tables/` |
| Figures and charts | `results/figures/` |
| Saved model weights | `results/models/` |
| Train/val/test splits | `data/splits/` |
