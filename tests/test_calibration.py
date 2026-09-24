import pytest
import json
import numpy as np
from pathlib import Path
from src2.models.inference import run_inference, _load_calibrated_thresholds

class DummyModel:
    status = "TRAINED"
    def predict(self, X):
        return {"risk_prob": [0.8] * len(X), "stage_idx": [5] * len(X)}
    def forecast(self, X, steps):
        return {"risk_trajectory": [0.8] * steps, "dynamics_trajectory": []}

class DummyLR:
    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        probs[:, 1] = 0.4  # LR risk
        return probs[:, 1]

def test_inference_uses_lr_threshold(tmp_path, monkeypatch):
    # Create dummy eval results
    import joblib
    import os
    
    thresh_path = tmp_path / "calibrated_threshold.json"
    with open(thresh_path, "w") as f:
        json.dump({
            "schema_version": 1,
            "lr_threshold": 0.5,
            "lstm_threshold": 0.9,
            "window_sec": 5,
            "calibration_split": "validation"
        }, f)
        
    lr_path = tmp_path / "lr_pipeline.pkl"
    joblib.dump({"pipeline": DummyLR()}, lr_path)
    
    # Mock data
    import pandas as pd
    from src2.data.schema import MODEL_FEATURES
    
    df = pd.DataFrame(np.random.rand(30, len(MODEL_FEATURES)), columns=MODEL_FEATURES)
    df["binary_label"] = 0.0
    
    res = run_inference(DummyModel(), df, MODEL_FEATURES, seq_len=5, forecast_steps=3, model_dir=str(tmp_path))
    
    print("RES:", res)
    assert res["status"] == "OK"
    assert res["threshold"] == 0.5  # Should use LR threshold
    assert res["risk_current"] == 0.4  # Should use LR risk
    assert res["n_flagged"] == 0  # 0.4 < 0.5

def test_inference_uses_lstm_fallback(tmp_path, monkeypatch):
    # No LR model saved
    thresh_path = tmp_path / "calibrated_threshold.json"
    with open(thresh_path, "w") as f:
        json.dump({
            "schema_version": 1,
            "lr_threshold": 0.5,
            "lstm_threshold": 0.9,
            "window_sec": 5,
            "calibration_split": "validation"
        }, f)
        
    import pandas as pd
    from src2.data.schema import MODEL_FEATURES
    
    df = pd.DataFrame(np.random.rand(30, len(MODEL_FEATURES)), columns=MODEL_FEATURES)
    df["binary_label"] = 0.0
    
    res = run_inference(DummyModel(), df, MODEL_FEATURES, seq_len=5, forecast_steps=3, model_dir=str(tmp_path))
    
    assert res["status"] == "OK"
    assert res["threshold"] == 0.9  # Should fallback to LSTM threshold
    assert res["risk_current"] == 0.8  # Should fallback to LSTM risk
    assert res["n_flagged"] == 0  # 0.8 < 0.9
