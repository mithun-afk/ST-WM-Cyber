"""src2/data/feature_engineering.py — derives extra features from canonical cols."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src2.data.schema import CANONICAL_FEATURES

# ---------------------------------------------------------------------------
# Full list of features after engineering (20 canonical + 5 derived = 25)
# ---------------------------------------------------------------------------
ENGINEERED_FEATURE_COLS: list[str] = CANONICAL_FEATURES + [
    "byte_ratio",
    "pkt_ratio",
    "iat_jitter",
    "syn_rate",
    "rst_rate",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 5 derived features to *df* and return the augmented DataFrame.

    Derived features
    ----------------
    byte_ratio  = fwd_bytes  / (bwd_bytes  + 1)
    pkt_ratio   = fwd_pkts   / (bwd_pkts   + 1)
    iat_jitter  = flow_iat_std / (flow_iat_mean + 1)
    syn_rate    = syn_flag_cnt / (fwd_pkts + bwd_pkts + 1)
    rst_rate    = rst_flag_cnt / (fwd_pkts + bwd_pkts + 1)

    All NaN produced by the arithmetic are filled with 0.

    Parameters
    ----------
    df : pd.DataFrame
        Must already contain CANONICAL_FEATURES columns.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with 5 additional columns appended (in-place copy).
    """
    df = df.copy()

    total_pkts = df["fwd_pkts"] + df["bwd_pkts"]

    df["byte_ratio"] = df["fwd_bytes"] / (df["bwd_bytes"] + 1)
    df["pkt_ratio"] = df["fwd_pkts"] / (df["bwd_pkts"] + 1)
    df["iat_jitter"] = df["flow_iat_std"] / (df["flow_iat_mean"] + 1)
    df["syn_rate"] = df["syn_flag_cnt"] / (total_pkts + 1)
    df["rst_rate"] = df["rst_flag_cnt"] / (total_pkts + 1)

    # Replace any inf or NaN produced by division
    for col in ["byte_ratio", "pkt_ratio", "iat_jitter", "syn_rate", "rst_rate"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df
