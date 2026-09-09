import pytest
import numpy as np
import pandas as pd
import torch
from src2.intelligence.mitre import MITRE_INDICATORS, match_indicators
from src2.intelligence.explainability import compute_gradient_saliency, get_important_features, get_important_windows

def test_mitre_indicators_defined():
    assert len(MITRE_INDICATORS) >= 8
    
    # Check required keys
    for ind in MITRE_INDICATORS:
        assert 'id' in ind
        assert 'name' in ind
        assert 'stage' in ind
        assert 'condition' in ind
        assert 'description' in ind

def test_match_indicators_demo():
    # Empty match
    df_empty = pd.DataFrame([{'syn_rate': 0, 'fwd_pkts': 100}])
    matches = match_indicators(df_empty)
    assert len(matches) == 0

def test_match_indicators_synthetic_recon():
    # Trigger T1046 Network Service Scanning
    df_recon = pd.DataFrame([{'syn_rate': 0.9, 'fwd_pkts': 2}])
    matches = match_indicators(df_recon)
    assert len(matches) == 1
    assert matches[0]['id'] == 'T1046'
    assert matches[0]['window_indices'] == [0]

class DummyNet(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.fc = torch.nn.Linear(input_dim, 1)
        
    def forward(self, x):
        # x is (B, seq_len, F)
        # Returns dyn, risk_logit, stage
        # We need risk_logit to be derived from x so grad exists
        dyn = torch.zeros(x.size(0), x.size(2))
        stage = torch.zeros(x.size(0), 6)
        
        # simple reduction
        risk_logit = self.fc(x.mean(dim=1))
        return dyn, risk_logit, stage

class DummyModel:
    def __init__(self, input_dim):
        self._net = DummyNet(input_dim)
        self.scaler = None

def test_gradient_saliency_shape():
    model = DummyModel(input_dim=25)
    X_tensor = torch.randn(1, 5, 25)
    saliency = compute_gradient_saliency(model, X_tensor)
    assert saliency.shape == (25,)

def test_gradient_saliency_no_crash():
    model = DummyModel(input_dim=25)
    X_tensor = torch.randn(1, 5, 25)
    saliency = compute_gradient_saliency(model, X_tensor)
    assert not np.isnan(saliency).any()

def test_important_features_top10():
    model = DummyModel(input_dim=25)
    X = np.random.randn(10, 5, 25)
    feature_cols = [f"feat_{i}" for i in range(25)]
    important = get_important_features(model, X, feature_cols, top_n=10)
    assert len(important) == 10
    assert 'feature' in important[0]
    assert 'importance' in important[0]
    assert 'rank' in important[0]
    assert important[0]['rank'] == 1

def test_important_windows_threshold():
    per_window_risk = [0.1, 0.6, 0.9, 0.4]
    timestamps = ["t1", "t2", "t3", "t4"]
    flagged = get_important_windows(per_window_risk, timestamps, threshold=0.5)
    assert len(flagged) == 2
    
    assert flagged[0]['window_idx'] == 1
    assert flagged[0]['severity'] == 'MEDIUM'
    
    assert flagged[1]['window_idx'] == 2
    assert flagged[1]['severity'] == 'HIGH'
    assert flagged[1]['timestamp'] == "t3"

def test_inference_api_populates_features():
    # Will be tested in full integration
    pass

def test_inference_api_populates_windows():
    # Will be tested in full integration
    pass

def test_full_contract_schema():
    # Will be tested in full integration
    pass
