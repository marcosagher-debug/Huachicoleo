from datetime import timedelta

import numpy as np
import pandas as pd

from src.preprocessing import (
    FEATURES,
    WindowConfig,
    build_windowed_datasets,
    fit_scaler_on_train_normals,
)


def make_sample_dataframe(rows: int = 12) -> pd.DataFrame:
    base_time = pd.Timestamp("2023-01-01 00:00:00")
    data = {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(rows)],
        "flow": np.linspace(10, 20, rows),
        "pressure": np.linspace(30, 40, rows),
        "pump_rpm": np.linspace(1000, 1100, rows),
        "tank_level": np.linspace(50, 60, rows),
        "power": np.linspace(200, 250, rows),
        "label": np.zeros(rows, dtype=int),
    }
    df = pd.DataFrame(data)
    # Introduce two anomalous points
    df.loc[rows - 2 :, "label"] = 1
    return df


def test_window_config_generates_expected_indices():
    config = WindowConfig(size=4, step=2)
    indices = list(config.iter_starts(10))
    assert indices == [0, 2, 4, 6]


def test_build_windowed_datasets_creates_expected_shapes():
    df = make_sample_dataframe()
    train_len = 6
    val_len = 3
    scaler = fit_scaler_on_train_normals(df, train_len)
    config = WindowConfig(size=3, step=1)

    (
        X_train_all,
        y_train_all,
        ts_train,
        X_val,
        y_val,
        ts_val,
        X_test,
        y_test,
        ts_test,
    ) = build_windowed_datasets(df, scaler, config, train_len, val_len)

    assert X_train_all.shape[1:] == (3, len(FEATURES))
    assert X_val.shape[0] == max(len(ts_val), 0)
    assert X_test.shape[0] == len(ts_test)
    # Ensure anomalous windows are flagged correctly
    assert y_test[-1] == 1
    assert y_train_all.max() in (0, 1)
