import pytest
import pandas as pd
from src2.intelligence.mitre import match_indicators

def test_zero_syn_flag_no_recon():
    # If syn_flag_cnt is 0, and syn_rate is 0, it should not trigger Reconnaissance
    df = pd.DataFrame([{
        "syn_flag_cnt": 0,
        "unique_dst_ports": 10,
        "syn_rate": 0.0
    }])
    
    matches = match_indicators(df)
    for m in matches:
        assert m["stage"] != "Reconnaissance"

def test_high_syn_flag_triggers_recon():
    # Should trigger T1046 Network Service Scanning
    df = pd.DataFrame([{
        "syn_flag_cnt": 5,
        "unique_dst_ports": 10,
        "syn_rate": 0.7
    }])
    
    matches = match_indicators(df)
    stages = [m["stage"] for m in matches]
    assert "Reconnaissance" in stages
