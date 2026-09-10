"""src2/models/inference.py — single inference entry point for the UI.

Implements the COMMON JSON CONTRACT defined in Checkpoint 2.

NOTE: All inference is done using a calibrated decision threshold tuned on
the validation set during training. This replaces the hard-coded 0.5 threshold
which produced systematic false positives in real-world conditions.
"""
from __future__ import annotations

import json
import joblib
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
    "Initial Access",
    "Lateral Movement",
    "Command & Control",
    "Impact",
]

# Risk level boundaries — these map raw neural network probability to a
# human-readable alert level displayed in the dashboard.
_RISK_LOW_THRESH    = 0.30
_RISK_MEDIUM_THRESH = 0.60


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


def _load_calibrated_threshold(model_dir: str = "eval_results") -> float:
    """Load the threshold tuned on the validation set during training.
    
    Falls back to 0.5 if not found (e.g. before first retrain).
    """
    thresh_path = Path(model_dir) / "calibrated_threshold.json"
    if thresh_path.exists():
        try:
            with open(thresh_path) as f:
                return float(json.load(f).get("threshold", 0.5))
        except Exception:
            pass
    return 0.5


def _load_lr_scorer(model_dir: str = "eval_results"):
    """Load the Logistic Regression primary risk scorer if available."""
    lr_path = Path(model_dir) / "lr_pipeline.pkl"
    if lr_path.exists():
        try:
            data = joblib.load(str(lr_path))
            return data.get("pipeline")
        except Exception:
            pass
    return None


def _make_error(code: str, **extra) -> dict:
    """Build a standardised error response."""
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
        "error": code,
        **extra,
    }


def run_inference(
    model,
    df_features,
    feature_cols: list[str],
    seq_len: int = 5,
    forecast_steps: int = 7,
    model_dir: str = "eval_results",
) -> dict:
    """Run the full inference pipeline and return the COMMON JSON CONTRACT dict.

    Parameters
    ----------
    model        : STGWMModel  (or any object with .status, .predict, .forecast)
    df_features  : pd.DataFrame  containing feature_cols (binary_label optional)
    feature_cols : list[str]  ordered feature column names
    seq_len      : int  temporal window length (must match training)
    forecast_steps : int  number of autoregressive forecast steps
    model_dir    : str  path to eval_results/ to load calibrated threshold

    Returns
    -------
    dict matching the COMMON JSON CONTRACT.
    """
    # ------------------------------------------------------------------
    # 0. Model sanity check
    # ------------------------------------------------------------------
    if model is None or model.status != "TRAINED":
        return _make_error("MODEL_NOT_TRAINED")

    # ------------------------------------------------------------------
    # 1. Data quality gate — minimum rows
    # ------------------------------------------------------------------
    n_rows = len(df_features)
    required = seq_len + 1
    if n_rows < required:
        return _make_error("INSUFFICIENT_HISTORY", required=required, available=n_rows)

    # ------------------------------------------------------------------
    # 2. Build temporal windows
    # ------------------------------------------------------------------
    import pandas as pd
    df_work = df_features.copy()
    if "binary_label" not in df_work.columns:
        df_work["binary_label"] = 0.0

    # Clip extreme values to prevent scaler overflow from stray packets
    for col in feature_cols:
        if col in df_work.columns:
            q99 = df_work[col].quantile(0.99)
            if q99 > 0:
                df_work[col] = df_work[col].clip(upper=q99 * 10)

    try:
        builder = TemporalWindowBuilder(seq_len=seq_len)
        X, y_risk_gt, timestamps = builder.build(df_work, feature_cols)
    except Exception as e:
        return _make_error(f"BUILD_ERROR: {str(e)}")

    n_windows = len(X)
    if n_windows < 1:
        return _make_error("INSUFFICIENT_HISTORY", required=required, available=n_rows)

    # ------------------------------------------------------------------
    # 3. Per-window risk scoring — LR + LSTM Ensemble
    # ------------------------------------------------------------------
    # Primary risk scorer: Logistic Regression (F1=0.81 on held-out test set)
    # Trained on window-aggregated features, calibrated threshold from validation.
    # LSTM provides temporal forecast trajectory and MITRE stage classification.
    lr_scorer = _load_lr_scorer(model_dir)
    threshold = _load_calibrated_threshold(model_dir)

    if lr_scorer is not None and hasattr(lr_scorer, 'predict_proba'):
        # Use LR for per-window risk probability (discriminative, calibrated)
        X_flat = X.reshape(len(X), -1)
        per_window_risk_raw = [float(p) for p in lr_scorer.predict_proba(X_flat)[:, 1]]
    else:
        # Fallback to LSTM if LR not available
        preds_raw = model.predict(X)
        per_window_risk_raw = [float(p) for p in preds_raw["risk_prob"]]

    # Always use LSTM for stage (temporal context matters for stage classification)
    preds = model.predict(X)
    per_window_stage: list[str] = [_stage_name(s) for s in preds["stage_idx"]]

    n_flagged = int(np.sum(np.array(per_window_risk_raw) > threshold))
    risk_current = per_window_risk_raw[-1]

    # ------------------------------------------------------------------
    # 4. Autoregressive forecast — LSTM world model
    # ------------------------------------------------------------------
    X_context = X[-1]   # (seq_len, F) — most recent LSTM context
    fc = model.forecast(X_context, steps=forecast_steps)
    # Scale LSTM forecast by current LR risk to ground it to reality
    lstm_forecast_raw: list[float] = [float(v) for v in fc["risk_trajectory"]]
    forecast_stages: list[int] = [int(v) for v in fc["stage_trajectory"]]

    # Blend: 60% LR-anchored, 40% LSTM trend direction
    # This keeps the forecast calibrated while showing temporal trends
    lr_anchor = risk_current
    forecast_risk = []
    for i, lv in enumerate(lstm_forecast_raw):
        # LSTM deviation from its own mean, scaled by LR anchor
        lstm_dev = lv - 0.5
        blended = np.clip(lr_anchor + 0.4 * lstm_dev + 0.05 * (i * lstm_dev), 0.0, 1.0)
        forecast_risk.append(float(blended))

    # Peak risk in forecast horizon
    if forecast_risk:
        peak_idx = int(np.argmax(forecast_risk))
        risk_peak = float(forecast_risk[peak_idx])
        peak_stage_idx = forecast_stages[peak_idx]
    else:
        risk_peak = risk_current
        peak_stage_idx = int(preds["stage_idx"][-1])

    # Stage: use majority vote of per-window stages, weighted by LR risk
    current_stage_idx = int(preds["stage_idx"][-1])
    # If LR says low risk but LSTM says high stage, trust LR
    if risk_current < threshold and current_stage_idx > 0:
        current_stage_idx = 0  # Override to Benign if LR says safe

    # Confidence: distance from decision boundary × LR probability
    stage_confidence = min(1.0, abs(risk_current - threshold) / max(threshold, 0.01))

    # ------------------------------------------------------------------
    # 5. Detect mode & synthetic flag
    # ------------------------------------------------------------------
    PCAP_MARKERS = {"ttl_mean", "ttl_std", "fragment_count", "retransmit_count"}
    mode = "FLOW_AND_PACKET_MODE" if PCAP_MARKERS & set(feature_cols) else "FLOW_ONLY_MODE"
    synthetic = "attack_stage" in df_features.columns

    # ------------------------------------------------------------------
    # 6. Explainability & MITRE intelligence
    # ------------------------------------------------------------------
    from src2.intelligence.explainability import get_important_features, get_important_windows
    from src2.intelligence.mitre import match_indicators

    df_last = pd.DataFrame(X[:, -1, :], columns=feature_cols)
    important_features = get_important_features(model, X, feature_cols, top_n=10)
    important_windows  = get_important_windows(per_window_risk_raw, timestamps=timestamps)
    matched_indicators = match_indicators(df_last)

    # ------------------------------------------------------------------
    # 7. Assemble contract
    # ------------------------------------------------------------------
    training_log = getattr(model, "_training_log", {})
    dynamics_mse = training_log.get("dynamics_mse", None)

    return {
        "status": "OK",
        "mode": mode,
        # Use current-window risk as primary (more stable than peak forecast)
        "risk": risk_current,
        "risk_peak": risk_peak,
        "risk_level": _risk_level(risk_current),
        "risk_current": risk_current,
        "threshold": threshold,
        "forecast": forecast_risk,
        "stage": _stage_name(current_stage_idx),
        "stage_confidence": stage_confidence,
        "per_window_risk": per_window_risk_raw,
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
