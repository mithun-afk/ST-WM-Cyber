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
    forecast_steps: int = 12,
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
        X, X_next, y_risk_gt, timestamps = builder.build(df_work, feature_cols)
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

    preds = model.predict(X)

    if lr_scorer is not None and hasattr(lr_scorer, 'predict_proba'):
        X_flat = X.reshape(n_windows, -1)
        try:
            lr_risk_raw = [float(p) for p in lr_scorer.predict_proba(X_flat)[:, 1]]
        except Exception as e:
            print(f"LR Scorer error (likely feature dim mismatch): {e}")
            lr_risk_raw = [float(p) for p in preds["risk_prob"]]
    else:
        lr_risk_raw = [float(p) for p in preds["risk_prob"]]

    # PRIMARY DETECTION: Logistic Regression (calibrated, stateless, high precision on protocol flags)
    # LSTM risk output is used only for the FORECAST trajectory (its sigmoid is biased toward 1.0
    # due to focal loss alpha=0.90 needed for 1.6% minority class training — not suitable as a
    # real-time alert trigger on its own).
    per_window_risk_raw = lr_risk_raw

    # Always use LSTM for stage (temporal context matters for stage classification)
    per_window_stage: list[str] = [_stage_name(s) for s in preds["stage_idx"]]

    n_flagged = int(np.sum(np.array(per_window_risk_raw) > threshold))
    risk_current = per_window_risk_raw[-1]

    # ------------------------------------------------------------------
    # 4. Autoregressive forecast — LSTM world model
    # ------------------------------------------------------------------
    X_context = X[-1]   # (seq_len, F) — most recent LSTM context
    fc = model.forecast(X_context, steps=forecast_steps)
    lstm_forecast_raw: list[float] = [float(v) for v in fc["risk_trajectory"]]
    
    # Calculate forecast stages using rules applied to the predicted dynamics trajectory
    from src2.intelligence.mitre import match_indicators
    dyn_traj = fc.get("dynamics_trajectory", [])
    forecast_stages_names = []
    if dyn_traj:
        df_forecast = pd.DataFrame(dyn_traj, columns=feature_cols)
        for i in range(len(df_forecast)):
            row_df = df_forecast.iloc[[i]]
            row_indicators = match_indicators(row_df)
            if float(lstm_forecast_raw[i]) >= 0.5 and row_indicators:
                best_ind = max(row_indicators, key=lambda x: x.get('count', 0))
                forecast_stages_names.append(best_ind['stage'])
            else:
                forecast_stages_names.append("Benign")
    else:
        forecast_stages_names = ["Benign"] * forecast_steps

    # Forecast: scale LSTM trajectory relative to current max-ensemble risk.
    # The LSTM gives relative direction; we anchor its mean to the current risk.
    lr_anchor = risk_current
    lstm_mean = float(np.mean(lstm_forecast_raw)) if lstm_forecast_raw else 0.5
    forecast_risk = []
    for i, lv in enumerate(lstm_forecast_raw):
        # Center LSTM output and scale around LR anchor
        lstm_dev = lv - lstm_mean
        blended = float(np.clip(lr_anchor + lstm_dev, 0.0, 1.0))
        forecast_risk.append(blended)

    # Peak risk in forecast horizon
    if forecast_risk:
        peak_idx = int(np.argmax(forecast_risk))
        risk_peak = float(forecast_risk[peak_idx])
    else:
        risk_peak = risk_current

    from src2.intelligence.mitre import match_indicators
    df_last = pd.DataFrame(X[:, -1, :], columns=feature_cols)
    matched_indicators = match_indicators(df_last)

    # Stage: explicitly rule/knowledge-graph based
    if risk_current >= threshold:
        if matched_indicators:
            # Pick the stage of the most frequently matched indicator
            best_indicator = max(matched_indicators, key=lambda x: x.get('count', 0))
            current_stage_name = best_indicator['stage']
        else:
            current_stage_name = "Unknown Activity"
    else:
        current_stage_name = "Benign"

    # Confidence based on LR distance from threshold and rule match
    base_conf = min(1.0, abs(risk_current - threshold) / max(threshold, 0.01))
    rule_bonus = 0.2 if matched_indicators else 0.0
    stage_confidence = min(1.0, base_conf + rule_bonus)

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

    important_features = get_important_features(model, lr_scorer, X, feature_cols, top_n=10)
    important_windows  = get_important_windows(per_window_risk_raw, timestamps=timestamps)

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
        "forecast_stages": forecast_stages_names,
        "stage": current_stage_name,
        "stage_confidence": stage_confidence,
        "per_window_risk": per_window_risk_raw,
        "per_window_stage": per_window_stage,
        "n_windows": n_windows,
        "n_flagged": n_flagged,
        "present_risk_explanation": important_features["present_risk_explanation"],
        "future_forecast_explanation": important_features["future_forecast_explanation"],
        "important_windows": important_windows,
        "matched_indicators": matched_indicators,
        "dynamics_mse": dynamics_mse,
        "synthetic": synthetic,
        "error": None,
    }
