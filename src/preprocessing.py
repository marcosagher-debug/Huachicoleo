"""Utilities for loading data and creating temporal windows for anomaly detection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Features expected in the dataset for the anomaly detection pipeline.
FEATURES: Tuple[str, ...] = (
    "flow",
    "pressure",
    "pump_rpm",
    "tank_level",
    "power",
)


@dataclass(frozen=True)
class WindowConfig:
    """Configuration describing how to slice the time series into windows."""

    size: int
    step: int

    def iter_starts(self, sequence_length: int) -> Iterable[int]:
        """Yield starting indices for the configured sliding window."""

        end = sequence_length - self.size + 1
        for start in range(0, max(end, 0), self.step):
            yield start


def load_dataset(path: Path, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Load a CSV file ordered by timestamp."""

    df = pd.read_csv(path, parse_dates=[timestamp_col])
    return df.sort_values(timestamp_col).reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataframe sequentially into train/validation/test partitions."""

    n_samples = len(df)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def fit_scaler_on_train_normals(
    df: pd.DataFrame, train_len: int, label_col: str = "label"
) -> StandardScaler:
    """Fit a scaler using only the normal observations from the training span."""

    scaler = StandardScaler()
    train_slice = df.iloc[:train_len]
    mask_normals = train_slice[label_col] == 0
    scaler.fit(train_slice.loc[mask_normals, FEATURES])
    return scaler


def generate_windows(
    values: np.ndarray, labels: np.ndarray, timestamps: pd.Series, config: WindowConfig
) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    """Create sliding windows and flag them as anomalous if any timestep is anomalous."""

    X, y, ts = [], [], []
    for start in config.iter_starts(len(values)):
        end = start + config.size
        if end > len(values):
            break
        X.append(values[start:end])
        y.append(int(labels[start:end].max()))
        ts.append(timestamps.iloc[end - 1])
    return np.array(X), np.array(y), pd.Series(ts, name="window_end")


def build_windowed_datasets(
    df: pd.DataFrame,
    scaler: StandardScaler,
    config: WindowConfig,
    train_len: int,
    val_len: int,
    label_col: str = "label",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    pd.Series,
    np.ndarray,
    np.ndarray,
    pd.Series,
    np.ndarray,
    np.ndarray,
    pd.Series,
]:
    """Scale the dataset and create windowed train/val/test arrays."""

    values = scaler.transform(df[list(FEATURES)])
    labels = df[label_col].to_numpy()
    timestamps = df["timestamp"]

    def make_split(start: int, end: int):
        return generate_windows(values[start:end], labels[start:end], timestamps.iloc[start:end], config)

    X_train_all, y_train_all, ts_train = make_split(0, train_len)
    X_val, y_val, ts_val = make_split(train_len, train_len + val_len)
    X_test, y_test, ts_test = make_split(train_len + val_len, len(df))
    return (
        X_train_all,
        y_train_all,
        ts_train,
        X_val,
        y_val,
        ts_val,
        X_test,
        y_test,
        ts_test,
    )


__all__ = [
    "FEATURES",
    "WindowConfig",
    "build_windowed_datasets",
    "fit_scaler_on_train_normals",
    "generate_windows",
    "load_dataset",
    "temporal_split",
]
