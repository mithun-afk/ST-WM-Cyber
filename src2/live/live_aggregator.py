import numpy as np
from typing import List, Dict

def aggregate_packets(packets, duration_sec: float) -> Dict[str, float]:
    """
    Aggregates a list of Scapy packets into the 20 Canonical + 4 PCAP features.
    Treats the entire window of traffic as an aggregated network state.
    """
    # If no packets, return zeros
    if not packets:
        return _empty_features()
        
    fwd_pkts = 0
    bwd_pkts = 0
    fwd_bytes = 0
    bwd_bytes = 0
    
    syn_cnt, rst_cnt, psh_cnt, ack_cnt, fin_cnt = 0, 0, 0, 0, 0
    pkt_lens = []
    ttls = []
    iats = []
    
    last_time = None
    
    # We will just split forward/backward by treating the first seen IP as "local/fwd"
    local_ip = None
    
    for p in packets:
        # Pkt Length
        length = len(p)
        pkt_lens.append(length)
        
        # Timing (IAT)
        pt = float(p.time)
        if last_time is not None:
            iats.append(pt - last_time)
        last_time = pt
        
        # IP layer
        if p.haslayer("IP"):
            if local_ip is None:
                local_ip = p["IP"].src
                
            is_fwd = (p["IP"].src == local_ip)
            ttls.append(p["IP"].ttl)
            
            if is_fwd:
                fwd_pkts += 1
                fwd_bytes += length
            else:
                bwd_pkts += 1
                bwd_bytes += length
                
        # TCP Flags
        if p.haslayer("TCP"):
            flags = p["TCP"].flags
            if 'S' in flags: syn_cnt += 1
            if 'R' in flags: rst_cnt += 1
            if 'P' in flags: psh_cnt += 1
            if 'A' in flags: ack_cnt += 1
            if 'F' in flags: fin_cnt += 1
            
    # Calculate stats safely
    pkt_lens = np.array(pkt_lens)
    iats = np.array(iats) if iats else np.array([0.0])
    ttls = np.array(ttls) if ttls else np.array([0.0])
    
    total_pkts = fwd_pkts + bwd_pkts
    total_bytes = fwd_bytes + bwd_bytes
    
    features = {
        "flow_duration": duration_sec * 1e6, # microsecs
        "fwd_pkts": float(fwd_pkts),
        "bwd_pkts": float(bwd_pkts),
        "fwd_bytes": float(fwd_bytes),
        "bwd_bytes": float(bwd_bytes),
        "flow_iat_mean": float(np.mean(iats) * 1e6) if len(iats) > 0 else 0.0,
        "flow_iat_std": float(np.std(iats) * 1e6) if len(iats) > 0 else 0.0,
        "syn_flag_cnt": float(syn_cnt),
        "rst_flag_cnt": float(rst_cnt),
        "psh_flag_cnt": float(psh_cnt),
        "ack_flag_cnt": float(ack_cnt),
        "fin_flag_cnt": float(fin_cnt),
        "pkt_len_mean": float(np.mean(pkt_lens)) if len(pkt_lens) > 0 else 0.0,
        "pkt_len_std": float(np.std(pkt_lens)) if len(pkt_lens) > 0 else 0.0,
        "flow_bytes_per_sec": float(total_bytes / duration_sec) if duration_sec > 0 else 0.0,
        "flow_pkts_per_sec": float(total_pkts / duration_sec) if duration_sec > 0 else 0.0,
        "init_fwd_win_bytes": 0.0, # Approximate / not easily available without flow tracking
        "active_mean": 0.0,
        "idle_mean": 0.0,
        "down_up_ratio": float(bwd_pkts / fwd_pkts) if fwd_pkts > 0 else 0.0,
        
        # PCAP extras
        "ttl_mean": float(np.mean(ttls)) if len(ttls) > 0 else 0.0,
        "ttl_std": float(np.std(ttls)) if len(ttls) > 0 else 0.0,
        "fragment_count": 0.0,
        "retransmit_count": 0.0
    }
    return features

def _empty_features() -> Dict[str, float]:
    return {
        "flow_duration": 0.0,
        "fwd_pkts": 0.0, "bwd_pkts": 0.0, "fwd_bytes": 0.0, "bwd_bytes": 0.0,
        "flow_iat_mean": 0.0, "flow_iat_std": 0.0,
        "syn_flag_cnt": 0.0, "rst_flag_cnt": 0.0, "psh_flag_cnt": 0.0,
        "ack_flag_cnt": 0.0, "fin_flag_cnt": 0.0,
        "pkt_len_mean": 0.0, "pkt_len_std": 0.0,
        "flow_bytes_per_sec": 0.0, "flow_pkts_per_sec": 0.0,
        "init_fwd_win_bytes": 0.0, "active_mean": 0.0, "idle_mean": 0.0,
        "down_up_ratio": 0.0,
        "ttl_mean": 0.0, "ttl_std": 0.0, "fragment_count": 0.0, "retransmit_count": 0.0
    }
