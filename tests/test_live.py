import pytest
from src2.live.interface import get_interfaces
from src2.live.live_aggregator import aggregate_packets

def test_interface_discovery():
    # It might return [] on permissions/no npcap, but shouldn't crash
    ifaces = get_interfaces()
    assert isinstance(ifaces, list)

def test_live_aggregator_empty():
    features = aggregate_packets([], 5.0)
    assert features["flow_duration"] == 0.0
    assert features["fwd_pkts"] == 0.0
    # Must have 24 features (Wait, Canonical=20, Pcap_extras=4 -> 24)
    # The 25th is added by feature engineering
    assert len(features) == 24
