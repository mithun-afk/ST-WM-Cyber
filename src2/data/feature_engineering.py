"""src2/data/feature_engineering.py — derives extra features from canonical cols."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src2.data.schema import CANONICAL_FEATURES, MODEL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 11 derived features and return the augmented DataFrame (25 -> 31 total).

    Derived features
    ----------------
    byte_ratio       = fwd_bytes / (bwd_bytes + 1)
    pkt_ratio        = fwd_pkts  / (bwd_pkts  + 1)
    iat_jitter       = flow_iat_std / (flow_iat_mean + 1)
    syn_rate         = syn_flag_cnt / (fwd_pkts + bwd_pkts + 1)
    rst_rate         = rst_flag_cnt / (fwd_pkts + bwd_pkts + 1)
    fragment_count   = from PCAP if available, else 0
    retransmit_count = from PCAP if available, else 0
    ttl_mean         = from PCAP if available, else 0
    ttl_std          = from PCAP if available, else 0
    unique_dst_ips   = unique destination IPs per window if available, else 0
    unique_dst_ports = unique destination ports per window if available, else 0
    """
    df = df.copy()

    total_pkts = df["fwd_pkts"] + df["bwd_pkts"]

    df["byte_ratio"] = df["fwd_bytes"] / (df["bwd_bytes"] + 1)
    df["pkt_ratio"] = df["fwd_pkts"] / (df["bwd_pkts"] + 1)
    df["iat_jitter"] = df["flow_iat_std"] / (df["flow_iat_mean"] + 1)
    df["syn_rate"] = df["syn_flag_cnt"] / (total_pkts + 1)
    df["rst_rate"] = df["rst_flag_cnt"] / (total_pkts + 1)

    # PCAP-derived features — use existing values if present, else zero-fill
    for col in ["fragment_count", "retransmit_count", "ttl_mean", "ttl_std",
                "unique_dst_ips", "unique_dst_ports"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # Replace any inf or NaN
    for col in ["byte_ratio", "pkt_ratio", "iat_jitter", "syn_rate", "rst_rate",
                "fragment_count", "retransmit_count", "ttl_mean", "ttl_std",
                "unique_dst_ips", "unique_dst_ports"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df

def prepare_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unified preprocessing function guaranteeing exact parity between 
    training-time and live inference-time feature preparation.
    """
    df = df.copy()
    
    # 1. Engineer derived features
    df = engineer_features(df)
    
    # 2. Ensure all MODEL_FEATURES exist (fill missing with 0.0)
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
            
    # 3. Identical missing-value handling (0-fill)
    df = df.fillna(0.0)
    
    # 4. Identical infinite-value handling (0-fill)
    df = df.replace([np.inf, -np.inf], 0.0)
    
    # 5. Identical clipping policy
    for col in MODEL_FEATURES:
        df[col] = df[col].clip(lower=-1e30, upper=1e30)
        
    # 6. Identical feature order
    return df[MODEL_FEATURES]
