import pytest
import pandas as pd
from src2.live.live_pipeline import LivePipeline
from src2.models.inference import run_inference
from src2.data.schema import MODEL_FEATURES

class DummyModel:
    status = "TRAINED"
    def predict(self, X):
        return {"risk_prob": [0.99] * len(X), "stage_idx": [5] * len(X)}
    def forecast(self, X, steps):
        return {"risk_trajectory": [0.99] * steps, "dynamics_trajectory": []}

def test_live_does_not_infer_before_min_history():
    pipeline = LivePipeline(model=DummyModel())
    
    for i in range(5):
        pipeline._process_window([])
        res = pipeline.latest_result
        assert res["status"] == "COLLECTING"
        assert res.get("n_windows", 0) == i
        assert res.get("risk", 0.0) == 0.0
        assert res.get("risk_current", 0.0) == 0.0
        
    pipeline._process_window([])
    res = pipeline.latest_result
    assert res["status"] != "COLLECTING"

def test_no_zero_window_is_passed_to_model():
    pipeline = LivePipeline(model=DummyModel())
    # The history starts empty
    assert len(pipeline.history) == 0
    pipeline._process_window([])
    # It adds a window, but doesn't pass zeroed windows to the model
    assert len(pipeline.history) == 1
    assert pipeline.latest_result["status"] == "COLLECTING"

def test_startup_cannot_generate_false_alert():
    pipeline = LivePipeline(model=DummyModel())
    pipeline._process_window([])
    assert pipeline.latest_result.get("stage") != "Impact"
    assert pipeline.latest_result.get("risk", 0.0) == 0.0

def test_live_requires_real_windows():
    pipeline = LivePipeline(model=DummyModel())
    pipeline._process_window([])
    res = pipeline.latest_result
    assert res.get("risk", 0.0) == 0.0
