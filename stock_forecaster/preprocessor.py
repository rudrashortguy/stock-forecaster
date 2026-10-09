"""Data preprocessing module for stock forecaster."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_and_normalize(
    filepath: str,
    lookback: int = 60,
    target_column: str = "Close",
) -> tuple:
    """
    Load OHLC data and create normalized sequences.

    Args:
        filepath: Path to CSV file with OHLC data.
        lookback: Number of past days to use for prediction.
        target_column: Column to predict (default: Close).

    Returns:
        Tuple of (X, y, scaler, prices_df)
    """
    df = pd.read_csv(filepath)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in {df.columns.tolist()}"
        )

    # Select relevant columns
    price_cols = ["Open", "High", "Low", "Close", "Volume"]
    available_cols = [c for c in price_cols if c in df.columns]
    data = df[available_cols].values

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback : i])
        y.append(scaled_data[i, available_cols.index(target_column)])

    X, y = np.array(X), np.array(y)
    return X, y, scaler, df[["Date", target_column]]


def create_sequences(data: np.ndarray, lookback: int = 60) -> tuple:
    """
    Create sequences from normalized data.

    Args:
        data: Normalized array of shape (n_samples, n_features).
        lookback: Number of past time steps.

    Returns:
        Tuple of (X, y) arrays.
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i])
        y.append(data[i, -1])  # Predict last feature (Close)

    return np.array(X), np.array(y)