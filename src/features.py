# src/features.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from .config import FEATURES_DIR

def load_all_features():
    """Load and merge all CSV files in processed/features"""
    dfs = []
    for file in os.listdir(FEATURES_DIR):
        if file.endswith(".csv"):
            path = os.path.join(FEATURES_DIR, file)
            print(f"📂 Loading {file} ...")
            df = pd.read_csv(path, low_memory=False)
            # Strip column names to avoid hidden spaces
            df.columns = df.columns.str.strip()
            dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"✅ Combined dataset shape: {combined.shape}")
    return combined

def preprocess_features(df: pd.DataFrame, label_col="Label"):
    """Clean, scale, and split features/labels"""

    # Strip column names just in case
    df.columns = df.columns.str.strip()

    # Ensure label exists
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found! Check your CSVs.")

    # Drop non-numeric columns except Label
    for col in df.columns:
        if col != label_col and not pd.api.types.is_numeric_dtype(df[col]):
            df.drop(columns=[col], inplace=True)

    # Replace infinities with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Debug: print columns containing NaN (original infs included)
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        print(f"⚠️ Columns with NaN or inf values: {nan_cols}")

    # Drop rows with NaN
    df.dropna(inplace=True)

    X = df.drop(columns=[label_col], errors="ignore")
    y = df[label_col]

    # Clip extreme values to a safe range
    X = np.clip(X, -1e10, 1e10)

    # Debug: check for remaining issues
    if np.isinf(X.values).any():
        print("❌ Warning: Infinite values remain in X after cleaning")
    if (np.abs(X.values) > 1e10).any():
        print("❌ Warning: Extremely large values remain in X after clipping")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"✅ Features scaled. Shape: {X_scaled.shape}")

    return X_scaled, y, scaler
