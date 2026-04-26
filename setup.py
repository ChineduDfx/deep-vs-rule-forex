import os
from pathlib import Path

from config import (
    DATASET_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    SPLITS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    MODELS_DIR,
)

FOLDERS = [
    DATASET_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    SPLITS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    MODELS_DIR,
]

for folder in FOLDERS:
    path = Path(folder)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {folder}")
    else:
        print(f"Already exists: {folder}")

print("\nProject setup complete.")
