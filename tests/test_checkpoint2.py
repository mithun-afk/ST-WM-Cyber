"""tests/test_checkpoint2.py — Pytest suite for Checkpoint 2: LSTM World Model."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make project importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src2.data.demo_generator import generate_demo_data
from src2.data.feature_engineering import engineer_features, ENGINEERED_FEATURE_COLS
from src2.data.temporal_windows import TemporalWindowBuilder
from src2.models.world_model import STGWMModel, _STGWMNet
from src2.models.inference import run_inference

import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def demo_df():
    path = generate_demo_data()
    df = pd.read_csv(path, comment="#")
    df = engineer_features(df)
    return df


@pytest.fixture(scope="module")
def windows(demo_df):
    builder = TemporalWindowBuilder(seq_len=5)
    X, y_risk, ts = builder.build(demo_df, ENGINEERED_FEATURE_COLS)
    return X, y_risk, ts


@pytest.fixture(scope="module")
def trained_model(demo_df):
    model = STGWMModel(input_dim=25, hidden_dim=32, num_layers=2, num_stages=6)
    log = model.fit(
        demo_df[ENGINEERED_FEATURE_COLS].values,
        demo_df["binary_label"].values,
        epochs=5,
    )
    return model, log


# ---------------------------------------------------------------------------
# test_model_forward_shape
# ---------------------------------------------------------------------------
def test_model_forward_shape():
    net = _STGWMNet(input_dim=25, hidden_dim=32, num_layers=2, num_stages=6)
    net.eval()
    B, seq_len, F = 8, 5, 25
    x = torch.randn(B, seq_len, F)
    with torch.no_grad():
        dyn, risk, stage = net(x)
    assert dyn.shape == (B, F),        f"dynamics shape {dyn.shape} != {(B, F)}"
    assert risk.shape == (B, 1),       f"risk shape {risk.shape} != {(B, 1)}"
    assert stage.shape == (B, 6),      f"stage shape {stage.shape} != {(B, 6)}"


# ---------------------------------------------------------------------------
# test_model_fit_demo
# ---------------------------------------------------------------------------
def test_model_fit_demo(trained_model):
    model, log = trained_model
    assert "converged" in log, "Training log missing 'converged' key"
    assert log["epochs_run"] == 5
    assert isinstance(log["best_val_loss"], float)
    assert log["converged"] is True


# ---------------------------------------------------------------------------
# test_model_predict_shape
# ---------------------------------------------------------------------------
def test_model_predict_shape(trained_model, windows):
    model, _ = trained_model
    X, _, _ = windows
    out = model.predict(X)
    assert "risk_prob" in out
    assert len(out["risk_prob"]) == len(X), (
        f"risk_prob length {len(out['risk_prob'])} != n_windows {len(X)}"
    )
    assert out["risk_prob"].ndim == 1
    assert out["stage_idx"].shape == (len(X),)
    assert out["dynamics_pred"].shape == (len(X), 25)


# ---------------------------------------------------------------------------
# test_model_forecast_steps
# ---------------------------------------------------------------------------
def test_model_forecast_steps(trained_model, windows):
    model, _ = trained_model
    X, _, _ = windows
    X_context = X[-1]   # (seq_len=5, F=25)
    fc = model.forecast(X_context, steps=7)
    assert "risk_trajectory" in fc
    assert len(fc["risk_trajectory"]) == 7, (
        f"Expected 7 forecast steps, got {len(fc['risk_trajectory'])}"
    )
    assert len(fc["stage_trajectory"]) == 7
    assert len(fc["dynamics_trajectory"]) == 7


# ---------------------------------------------------------------------------
# test_model_save_load
# ---------------------------------------------------------------------------
def test_model_save_load(trained_model, windows):
    model, _ = trained_model
    X, _, _ = windows
    orig_preds = model.predict(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(tmpdir)
        loaded, tag = STGWMModel.load(tmpdir)

    assert tag == "LOADED", f"Expected 'LOADED', got '{tag}'"
    assert loaded is not None
    assert loaded.status == "TRAINED"

    loaded_preds = loaded.predict(X)
    np.testing.assert_allclose(
        orig_preds["risk_prob"], loaded_preds["risk_prob"], atol=1e-5,
        err_msg="Predictions differ after save/load"
    )


# ---------------------------------------------------------------------------
# test_model_load_missing
# ---------------------------------------------------------------------------
def test_model_load_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        model, tag = STGWMModel.load(tmpdir)
    assert model is None
    assert tag == "MODEL_NOT_TRAINED"


# ---------------------------------------------------------------------------
# test_inference_api_ok
# ---------------------------------------------------------------------------
def test_inference_api_ok(trained_model, demo_df):
    model, _ = trained_model
    result = run_inference(model, demo_df, ENGINEERED_FEATURE_COLS, seq_len=5, forecast_steps=7)
    assert result["status"] == "OK", f"Expected OK, got: {result}"
    assert result["error"] is None
    assert isinstance(result["risk"], float)
    assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert isinstance(result["n_windows"], int) and result["n_windows"] > 0
    assert len(result["forecast"]) == 7


# ---------------------------------------------------------------------------
# test_inference_api_not_trained
# ---------------------------------------------------------------------------
def test_inference_api_not_trained(demo_df):
    untrained = STGWMModel(input_dim=25)
    result = run_inference(untrained, demo_df, ENGINEERED_FEATURE_COLS)
    assert result["error"] == "MODEL_NOT_TRAINED"


# ---------------------------------------------------------------------------
# test_inference_api_insufficient
# ---------------------------------------------------------------------------
def test_inference_api_insufficient(trained_model, demo_df):
    model, _ = trained_model
    tiny_df = demo_df.head(3)   # only 3 rows — far less than seq_len+1=6
    result = run_inference(model, tiny_df, ENGINEERED_FEATURE_COLS, seq_len=5)
    assert result["error"] == "INSUFFICIENT_HISTORY", (
        f"Expected INSUFFICIENT_HISTORY, got: {result['error']}"
    )


# ---------------------------------------------------------------------------
# test_forecast_is_list_of_floats
# ---------------------------------------------------------------------------
def test_forecast_is_list_of_floats(trained_model, windows):
    model, _ = trained_model
    X, _, _ = windows
    fc = model.forecast(X[-1], steps=7)
    for val in fc["risk_trajectory"]:
        assert isinstance(val, float), f"Expected float, got {type(val)}"
        assert 0.0 <= val <= 1.0, f"Risk probability {val} out of [0,1]"
