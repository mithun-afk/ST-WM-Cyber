"""
src2/live/live_aggregator.py
----------------------------
Aggregates raw Scapy packets captured over a time window into the 20-feature
canonical network statistics vector used by the ST-WM pipeline.

IMPORTANT: All feature values here are computed at WINDOW granularity — the same
granularity at which the model was trained. Do NOT use CIC-IDS-2018 per-flow
values here; they live at a completely different statistical scale.
"""
import numpy as np
from typing import List, Dict


# ---------------------------------------------------------------------------
# Realistic idle-traffic baseline for an idle Windows laptop on a corporate or
# college network. These values represent a quiet machine sending background
# traffic: ARP broadcasts, DHCP renewals, DNS TTL refreshes, NTP heartbeats.
# Used as startup padding so the LSTM history buffer is not filled with
# pathological all-zeros that look like a dead network to the StandardScaler.
# ---------------------------------------------------------------------------
_IDLE_BASELINE = {
    "flow_duration":      28000.0,   # µs: ~28ms median per background flow (matches window-aggregated CIC median)
    "fwd_pkts":              3.5,
    "bwd_pkts":              1.0,
    "fwd_bytes":           400.0,
    "bwd_bytes":           130.0,
    "flow_iat_mean":     16000.0,   # µs: matches window-aggregated training median
    "flow_iat_std":       7000.0,
    "syn_flag_cnt":          0.0,   # CIC window median = 0
    "rst_flag_cnt":          0.1,
    "psh_flag_cnt":          0.4,
    "ack_flag_cnt":          0.2,
    "fin_flag_cnt":          0.0,
    "pkt_len_mean":         68.0,   # matches training median ~67
    "pkt_len_std":          75.0,
    "flow_bytes_per_sec":  8000.0,  # moderate background traffic
    "flow_pkts_per_sec":    120.0,
    "init_fwd_win_bytes":  39000.0, # matches training median
    "active_mean":           0.0,   # always 0 in CIC window aggregation
    "idle_mean":             0.0,   # always 0 in CIC window aggregation
    "down_up_ratio":         0.2,
    # PCAP extras
    "ttl_mean":             64.0,
    "ttl_std":               4.0,
    "fragment_count":        0.0,
    "retransmit_count":      0.0,
}


def _empty_features() -> Dict[str, float]:
    """Return an idle-baseline feature dict for LSTM startup padding.
    
    This represents a realistic quiet/idle network, NOT mathematical zeros.
    Mathematical zeros cause the StandardScaler to produce extreme anomaly
    signals for features with near-zero training variance (e.g. syn_flag_cnt).
    """
    return dict(_IDLE_BASELINE)


def aggregate_packets(packets, duration_sec: float) -> Dict[str, float]:
    """
    Aggregate a list of Scapy packets into the canonical feature vector.

    All statistics are window-level (computed over `duration_sec` of traffic),
    matching the distribution the model was trained on.

    Parameters
    ----------
    packets     : list of Scapy packet objects captured during this window
    duration_sec: length of the capture window in seconds (typically 5.0)

    Returns
    -------
    dict mapping canonical feature names to float values
    """
    if not packets:
        return _empty_features()

    fwd_pkts = 0
    bwd_pkts = 0
    fwd_bytes = 0
    bwd_bytes = 0
    syn_cnt = rst_cnt = psh_cnt = ack_cnt = fin_cnt = 0

    pkt_lens: list[float] = []
    ttls: list[float] = []
    iats: list[float] = []
    flow_durations: list[float] = []

    last_time = None
    local_ip = None

    # Track per-flow durations for a realistic flow_duration mean
    flow_start: dict = {}   # flow_key -> first_seen_time

    for p in packets:
        length = len(p)
        pkt_lens.append(float(length))

        pt = float(p.time)
        if last_time is not None:
            iats.append(pt - last_time)
        last_time = pt

        if p.haslayer("IP"):
            ip = p["IP"]
            if local_ip is None:
                local_ip = ip.src

            ttls.append(float(ip.ttl))
            is_fwd = (ip.src == local_ip)

            if is_fwd:
                fwd_pkts += 1
                fwd_bytes += length
            else:
                bwd_pkts += 1
                bwd_bytes += length

            # Build a rough flow key (5-tuple hash)
            if p.haslayer("TCP"):
                tcp = p["TCP"]
                flags = tcp.flags
                if "S" in flags: syn_cnt += 1
                if "R" in flags: rst_cnt += 1
                if "P" in flags: psh_cnt += 1
                if "A" in flags: ack_cnt += 1
                if "F" in flags:
                    fin_cnt += 1
                    # Mark flow end for duration tracking
                    fkey = (ip.src, ip.dst, tcp.sport, tcp.dport)
                    if fkey in flow_start:
                        flow_durations.append((pt - flow_start[fkey]) * 1e6)
                        del flow_start[fkey]
                elif "S" in flags:
                    fkey = (ip.src, ip.dst, tcp.sport, tcp.dport)
                    flow_start[fkey] = pt

            elif p.haslayer("UDP"):
                psh_cnt += 1  # UDP = data delivery, equiv to PSH

    # -----------------------------------------------------------------------
    # Compute statistics safely
    # -----------------------------------------------------------------------
    pkt_arr = np.array(pkt_lens, dtype=np.float32)
    iat_arr = np.array(iats, dtype=np.float32) if iats else np.array([0.03], dtype=np.float32)
    ttl_arr = np.array(ttls, dtype=np.float32) if ttls else np.array([64.0], dtype=np.float32)

    total_pkts = fwd_pkts + bwd_pkts
    total_bytes = fwd_bytes + bwd_bytes

    # flow_duration: mean duration of COMPLETED flows within this window (µs).
    # Falls back to mean IAT * total_pkts as an approximation.
    if flow_durations:
        mean_flow_dur = float(np.mean(flow_durations))
    elif len(iat_arr) > 1:
        mean_flow_dur = float(np.mean(iat_arr) * total_pkts * 1e6)
    else:
        mean_flow_dur = 50000.0  # 50ms default for idle windows

    # init_fwd_win_bytes proxy: typical TCP window size for observed SYN packets
    init_fwd_win = 65535.0 if syn_cnt > 0 else 0.0

    return {
        "flow_duration":       mean_flow_dur,
        "fwd_pkts":            float(fwd_pkts),
        "bwd_pkts":            float(bwd_pkts),
        "fwd_bytes":           float(fwd_bytes),
        "bwd_bytes":           float(bwd_bytes),
        "flow_iat_mean":       float(np.mean(iat_arr) * 1e6),
        "flow_iat_std":        float(np.std(iat_arr) * 1e6) if len(iat_arr) > 1 else 0.0,
        "syn_flag_cnt":        float(syn_cnt),
        "rst_flag_cnt":        float(rst_cnt),
        "psh_flag_cnt":        float(psh_cnt),
        "ack_flag_cnt":        float(ack_cnt),
        "fin_flag_cnt":        float(fin_cnt),
        "pkt_len_mean":        float(np.mean(pkt_arr)) if len(pkt_arr) > 0 else 0.0,
        "pkt_len_std":         float(np.std(pkt_arr)) if len(pkt_arr) > 1 else 0.0,
        "flow_bytes_per_sec":  float(total_bytes / duration_sec) if duration_sec > 0 else 0.0,
        "flow_pkts_per_sec":   float(total_pkts / duration_sec) if duration_sec > 0 else 0.0,
        "init_fwd_win_bytes":  init_fwd_win,
        "active_mean":         float(np.mean(iat_arr) * 1e6 * 0.3) if len(iat_arr) > 0 else 500.0,
        "idle_mean":           float(np.mean(iat_arr) * 1e6 * 1.2) if len(iat_arr) > 0 else 2500.0,
        "down_up_ratio":       float(bwd_pkts / fwd_pkts) if fwd_pkts > 0 else 1.0,
        # PCAP extras
        "ttl_mean":            float(np.mean(ttl_arr)),
        "ttl_std":             float(np.std(ttl_arr)) if len(ttl_arr) > 1 else 0.0,
        "fragment_count":      0.0,
        "retransmit_count":    0.0,
    }
