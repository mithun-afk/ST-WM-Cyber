"""src2/data/csv_loader.py — loads and normalises CIC-IDS-2018 CSV files."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src2.data.schema import CANONICAL_FEATURES
from src2.config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name mapping: raw CIC-IDS-2018 header  →  canonical name
# ---------------------------------------------------------------------------
_CIC_COLUMN_MAP: dict[str, str] = {
    "Flow Duration": "flow_duration",
    "Tot Fwd Pkts": "fwd_pkts",
    "Tot Bwd Pkts": "bwd_pkts",
    "TotLen Fwd Pkts": "fwd_bytes",
    "TotLen Bwd Pkts": "bwd_bytes",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "SYN Flag Cnt": "syn_flag_cnt",
    "RST Flag Cnt": "rst_flag_cnt",
    "PSH Flag Cnt": "psh_flag_cnt",
    "ACK Flag Cnt": "ack_flag_cnt",
    "FIN Flag Cnt": "fin_flag_cnt",
    "Pkt Len Mean": "pkt_len_mean",
    "Pkt Len Std": "pkt_len_std",
    "Flow Byts/s": "flow_bytes_per_sec",
    "Flow Pkts/s": "flow_pkts_per_sec",
    "Init Fwd Win Byts": "init_fwd_win_bytes",
    "Active Mean": "active_mean",
    "Idle Mean": "idle_mean",
    "Down/Up Ratio": "down_up_ratio",
}

# Columns kept in addition to canonical features
_META_COLS = ["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol", "Timestamp"]


def _normalise_label(series: pd.Series, attack_keywords: list[str]) -> pd.Series:
    """Map label strings → 0 (benign), 1 (attack), -1 (unknown/drop)."""
    result = pd.Series(-1, index=series.index, dtype=int)
    lower = series.str.strip().str.lower()
    benign_mask = lower.isin(["benign"])
    result[benign_mask] = 0
    # Any label containing an attack keyword → 1
    attack_mask = series.str.strip().apply(
        lambda lbl: any(kw.lower() in lbl.lower() for kw in attack_keywords)
    )
    result[attack_mask] = 1
    return result


def load_and_normalize_csv(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Load a CIC-IDS-2018 CSV and return (normalised_df, mode).

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with CANONICAL_FEATURES + metadata + ``binary_label``.
    mode : str
        Always ``'FLOW_ONLY_MODE'`` (PCAP extends this).

    Raises
    ------
    ValueError
        If the file does not exist or critical columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Data file not found: {path}")

    cfg = get_config()
    attack_keywords: list[str] = cfg["labels"]["attack_keywords"]

    # ------------------------------------------------------------------
    # 1. Read CSV, strip whitespace from column names
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV '{path}': {exc}") from exc

    df.columns = [c.strip() for c in df.columns]

    # ------------------------------------------------------------------
    # 2. Check Label column exists
    # ------------------------------------------------------------------
    if "Label" not in df.columns:
        raise ValueError(
            f"Critical column 'Label' is missing from '{path}'. "
            f"Available columns: {list(df.columns)}"
        )

    # ------------------------------------------------------------------
    # 3. Check required CIC columns exist before renaming
    # ------------------------------------------------------------------
    missing = [c for c in _CIC_COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(
            f"Critical feature columns missing from '{path}': {missing}"
        )

    # ------------------------------------------------------------------
    # 4. Rename CIC columns → canonical names
    # ------------------------------------------------------------------
    df = df.rename(columns=_CIC_COLUMN_MAP)

    # ------------------------------------------------------------------
    # 5. Normalise label
    # ------------------------------------------------------------------
    df["binary_label"] = _normalise_label(df["Label"], attack_keywords)
    # Drop unknown rows
    df = df[df["binary_label"] != -1].copy()
    df["binary_label"] = df["binary_label"].astype(np.float32)

    # ------------------------------------------------------------------
    # 6. Parse and sort by Timestamp
    # ------------------------------------------------------------------
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
        df = df.sort_values("Timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 7. Select output columns
    # ------------------------------------------------------------------
    keep_meta = [c for c in _META_COLS if c in df.columns]
    output_cols = keep_meta + ["binary_label"] + CANONICAL_FEATURES
    # Only keep columns that actually exist
    output_cols = [c for c in output_cols if c in df.columns]
    df = df[output_cols].copy()

    # ------------------------------------------------------------------
    # 8. Replace inf → NaN
    # ------------------------------------------------------------------
    df[CANONICAL_FEATURES] = df[CANONICAL_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    # ------------------------------------------------------------------
    # 9. Clip numerical columns at 1st and 99th percentile
    # ------------------------------------------------------------------
    for col in CANONICAL_FEATURES:
        s = pd.to_numeric(df[col], errors="coerce")
        lo = s.quantile(0.01)
        hi = s.quantile(0.99)
        df[col] = s.clip(lo, hi)

    logger.info("Loaded %d rows from '%s' (mode=FLOW_ONLY_MODE)", len(df), path)
    return df, "FLOW_ONLY_MODE"
