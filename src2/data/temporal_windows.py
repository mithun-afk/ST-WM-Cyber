"""src2/data/temporal_windows.py — builds sliding temporal windows for LSTM."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class TemporalWindowBuilder:
    """Converts a flat time-sorted DataFrame into (X, y, timestamps) arrays.

    Parameters
    ----------
    seq_len : int
        Number of time steps in each input window.
    """

    def __init__(self, seq_len: int = 5) -> None:
        self.seq_len = seq_len

    # ------------------------------------------------------------------
    def build(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "binary_label",
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """Create sliding windows from a chronologically-ordered DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must be sorted by time and contain *feature_cols* + *label_col*.
        feature_cols : list[str]
            Column names used as features (shape F per row).
        label_col : str
            Column containing binary labels (0 / 1).

        Returns
        -------
        X : np.ndarray  shape (N, seq_len, F)  dtype float32
        y_risk : np.ndarray  shape (N,)  dtype float32  — label at t+1
        timestamps : list  — timestamp (or index) at t+1 for each window
        """
        n = len(df)
        required = self.seq_len + 1
        if n < required:
            raise ValueError(
                f"Insufficient temporal data.\n\n"
                f"Required:\n{required} observations\n\n"
                f"Available:\n{n} observations\n\n"
                f"Please provide a longer capture."
            )

        feature_arr = df[feature_cols].to_numpy(dtype=np.float32)
        label_arr = df[label_col].to_numpy(dtype=np.float32)

        # Resolve timestamps — use 'Timestamp' or 'timestamp' col, else index
        if "Timestamp" in df.columns:
            ts_arr = df["Timestamp"].tolist()
        elif "timestamp" in df.columns:
            ts_arr = df["timestamp"].tolist()
        else:
            ts_arr = list(df.index)

        X_list: list[np.ndarray] = []
        y_list: list[float] = []
        ts_list: list = []

        # Slide window: window = [i : i+seq_len], target = i+seq_len
        for i in range(n - self.seq_len):
            X_list.append(feature_arr[i : i + self.seq_len])
            y_list.append(label_arr[i + self.seq_len])
            ts_list.append(ts_arr[i + self.seq_len])

        X = np.stack(X_list, axis=0).astype(np.float32)          # (N, seq_len, F)
        y_risk = np.array(y_list, dtype=np.float32)               # (N,)
        return X, y_risk, ts_list
