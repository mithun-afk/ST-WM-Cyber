"""src2/models/inference.py — single inference entry point for the UI.

Implements the COMMON JSON CONTRACT defined in Checkpoint 2.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src2.data.temporal_windows import TemporalWindowBuilder

# ---------------------------------------------------------------------------
# MITRE stage names (index matches num_stages=6 in STGWMModel)
# ---------------------------------------------------------------------------
STAGE_NAMES = [
    "Benign",
    "Reconnaissance",
    "InitialAccess",
    "LateralMovement",
    "CommandAndControl",
    "Impact",
]

_RISK_LOW_THRESH    = 0.35
_RISK_MEDIUM_THRESH = 0.65


def _risk_level(p: float) -> str:
    if p < _RISK_LOW_THRESH:
        return "LOW"
    if p < _RISK_MEDIUM_THRESH:
        return "MEDIUM"
    return "HIGH"


def _stage_name(idx: int) -> str:
    idx = int(idx)
    if 0 <= idx < len(STAGE_NAMES):
        return STAGE_NAMES[idx]
    return f"Stage{idx}"


def run_inference(
    model,
    df_features,
    feature_cols: list[str],
    seq_len: int = 5,
    forecast_steps: int = 7,
) -> dict:
    """Run the full inference pipeline and return the COMMON JSON CONTRACT dict.

    Parameters
    ----------
    model        : STGWMModel  (or any object with .status, .predict, .forecast)
    df_features  : pd.DataFrame  containing feature_cols + 'binary_label'
    feature_cols : list[str]  ordered feature column names
    seq_len      : int  temporal window length
    forecast_steps : int  number of autoregressive forecast steps

    Returns
    -------
    dict matching the COMMON JSON CONTRACT.
    """
    # ------------------------------------------------------------------
    # 0. Check for empty traffic (e.g. initial padding or dead air)
    if "fwd_pkts" in df_features.columns and df_features["fwd_pkts"].sum() == 0 and df_features["bwd_pkts"].sum() == 0:
        return {"status": "OK", "mode": None, "risk": 0.0, "risk_level": "Safe", "risk_current": 0.0, "forecast": [0.0] * forecast_steps, "stage": "Benign", "stage_confidence": 1.0, "per_window_risk": [0.0], "per_window_stage": ["Benign"], "n_windows": 1, "n_flagged": 0, "important_features": [], "important_windows": [], "matched_indicators": [], "dynamics_mse": 0.0, "synthetic": False, "error": None}

    # 0. Check model is trained
    # ------------------------------------------------------------------
    if model.status != "TRAINED":
        return {
            "status": "ERROR",
            "mode": None,
            "risk": None,
            "risk_level": None,
            "risk_current": None,
            "forecast": [],
            "stage": None,
            "stage_confidence": None,
            "per_window_risk": [],
            "per_window_stage": [],
            "n_windows": 0,
            "n_flagged": 0,
            "important_features": [],
            "important_windows": [],
            "matched_indicators": [],
            "dynamics_mse": None,
            "synthetic": False,
            "error": "MODEL_NOT_TRAINED",
        }

    # ------------------------------------------------------------------
    # 1. Data quality gate — minimum rows
    # ------------------------------------------------------------------
    n_rows = len(df_features)
    required = seq_len + 1
    if n_rows < required:
        return {
            "status": "ERROR",
            "mode": None,
            "risk": None,
            "risk_level": None,
            "risk_current": None,
            "forecast": [],
            "stage": None,
            "stage_confidence": None,
            "per_window_risk": [],
            "per_window_stage": [],
            "n_windows": 0,
            "n_flagged": 0,
            "important_features": [],
            "important_windows": [],
            "matched_indicators": [],
            "dynamics_mse": None,
            "synthetic": False,
            "error": "INSUFFICIENT_HISTORY",
            "required": required,
            "available": n_rows,
        }

    # ------------------------------------------------------------------
    # 2. Build temporal windows
    # ------------------------------------------------------------------
    # Ensure binary_label col exists (use zeros if missing for inference-only)
    df_work = df_features.copy()
    if "binary_label" not in df_work.columns:
        df_work["binary_label"] = 0.0

    try:
        builder = TemporalWindowBuilder(seq_len=seq_len)
        X, y_risk_gt, timestamps = builder.build(df_work, feature_cols)
    except Exception as e:
        return {
            "status": "ERROR",
            "mode": None,
            "risk": None,
            "risk_level": None,
            "risk_current": None,
            "forecast": [],
            "stage": None,
            "stage_confidence": None,
            "per_window_risk": [],
            "per_window_stage": [],
            "n_windows": 0,
            "n_flagged": 0,
            "important_features": [],
            "important_windows": [],
            "matched_indicators": [],
            "dynamics_mse": None,
            "synthetic": False,
            "error": f"BUILD_ERROR: {str(e)}",
        }

    n_windows = len(X)
    if n_windows < 1:
        return {
            "status": "ERROR",
            "mode": None,
            "risk": None,
            "risk_level": None,
            "risk_current": None,
            "forecast": [],
            "stage": None,
            "stage_confidence": None,
            "per_window_risk": [],
            "per_window_stage": [],
            "n_windows": n_windows,
            "n_flagged": 0,
            "important_features": [],
            "important_windows": [],
            "matched_indicators": [],
            "dynamics_mse": None,
            "synthetic": False,
            "error": "INSUFFICIENT_HISTORY",
            "required": required,
            "available": n_rows,
        }

    # ------------------------------------------------------------------
    # 3. Per-window predictions
    # ------------------------------------------------------------------
    preds = model.predict(X)
    per_window_risk: list[float] = [float(p) for p in preds["risk_prob"]]
    per_window_stage: list[str] = [_stage_name(s) for s in preds["stage_idx"]]

    risk_current = per_window_risk[-1]
    n_flagged = int(np.sum(np.array(per_window_risk) > 0.5))

    # ------------------------------------------------------------------
    # 4. Forecast — use last seq_len windows as context
    # ------------------------------------------------------------------
    # X_context: shape (seq_len, F) — last window's raw (unscaled) feature rows
    # X shape is (N, seq_len, F); take last window
    X_context = X[-1]   # (seq_len, F)
    fc = model.forecast(X_context, steps=forecast_steps)
    forecast_risk: list[float] = [float(v) for v in fc["risk_trajectory"]]
    forecast_stages: list[int] = [int(v) for v in fc["stage_trajectory"]]

    # Peak risk in forecast
    if forecast_risk:
        peak_idx = int(np.argmax(forecast_risk))
        risk_peak = float(forecast_risk[peak_idx])
        peak_stage_idx = forecast_stages[peak_idx]
    else:
        risk_peak = risk_current
        peak_stage_idx = int(preds["stage_idx"][-1])

    # Stage confidence — softmax of stage_head outputs unavailable here,
    # so use a heuristic: we proxy it as the peak risk for now
    # (Checkpoint 3 will wire actual softmax probs)
    stage_confidence = risk_peak

    # ------------------------------------------------------------------
    # ENTERPRISE SOC HEURISTIC OVERRIDE (False Positive Reduction)
    # ------------------------------------------------------------------
    # In real environments, streaming 4K video or downloading huge files 
    # creates a massive bandwidth spike that looks identical to a DoS attack to a naive AI.
    # A true DoS (Impact) relies on high connection rates (syn_rate) and tiny packets.
    # Video streaming uses huge packets (MTU 1500) and low SYN rates.
    
    if len(df_features) > 0:
        latest = df_features.iloc[-1]
        syn_rate = latest.get("syn_rate", 0)
        pkt_len_mean = latest.get("pkt_len_mean", 0)
        
        # If the network is pushing huge packets with minimal new connections, it is a Benign download.
        if pkt_len_mean > 500 and syn_rate < 0.05:
            # Override False Positive
            risk_peak = min(risk_peak, 0.15)
            risk_current = min(risk_current, 0.15)
            peak_stage_idx = 0 # Benign
            stage_confidence = 0.99
            
            # Flatten trajectory to safe levels
            forecast_risk = [min(r, 0.15) for r in forecast_risk]
            forecast_stages = [0] * len(forecast_stages)
            per_window_risk = [min(r, 0.15) for r in per_window_risk]
            per_window_stage = ["Benign"] * len(per_window_stage)

    # ------------------------------------------------------------------
    # 5. Detect mode
    # ------------------------------------------------------------------
    # Heuristic: if feature_cols contains any pcap-only feature, it's FLOW_AND_PACKET
    PCAP_MARKERS = {"ttl_mean", "ttl_std", "fragment_count", "retransmit_count"}
    mode = "FLOW_AND_PACKET_MODE" if PCAP_MARKERS & set(feature_cols) else "FLOW_ONLY_MODE"

    # ------------------------------------------------------------------
    # 6. Detect synthetic data
    # ------------------------------------------------------------------
    synthetic = False
    if "attack_stage" in df_features.columns:
        # Check if it contains demo stage names (heuristic)
        synthetic = True

    # ------------------------------------------------------------------
    # 7. Intelligence (explainability and mitre indicators)
    # ------------------------------------------------------------------
    from src2.intelligence.explainability import get_important_features, get_important_windows
    from src2.intelligence.mitre import match_indicators
    import pandas as pd

    df_last = pd.DataFrame(X[:, -1, :], columns=feature_cols)
    important_features = get_important_features(model, X, feature_cols, top_n=10)
    important_windows = get_important_windows(per_window_risk, timestamps=timestamps)
    matched_indicators = match_indicators(df_last)

    # ------------------------------------------------------------------
    # 8. Assemble contract
    # ------------------------------------------------------------------
    training_log = getattr(model, "_training_log", {})
    dynamics_mse = training_log.get("dynamics_mse", None)

    return {
        "status": "OK",
        "mode": mode,
        "risk": risk_peak,
        "risk_level": _risk_level(risk_peak),
        "risk_current": risk_current,
        "forecast": forecast_risk,
        "stage": _stage_name(peak_stage_idx),
        "stage_confidence": stage_confidence,
        "per_window_risk": per_window_risk,
        "per_window_stage": per_window_stage,
        "n_windows": n_windows,
        "n_flagged": n_flagged,
        "important_features": important_features,
        "important_windows": important_windows,
        "matched_indicators": matched_indicators,
        "dynamics_mse": dynamics_mse,
        "synthetic": synthetic,
        "error": None,
    }
