"""src2/data/pcap_parser.py — extracts per-flow PCAP features using scapy."""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MIN_IP_PCT = 0.50  # at least 50 % of packets must have an IP layer


def _flow_key(pkt) -> str:
    """Return a canonical 5-tuple flow key string."""
    try:
        ip = pkt["IP"]
        proto = ip.proto
        src = ip.src
        dst = ip.dst
        sport = getattr(pkt, "sport", 0) or 0
        dport = getattr(pkt, "dport", 0) or 0
        # Normalise direction (lower IP first)
        if (src, sport) > (dst, dport):
            src, dst = dst, src
            sport, dport = dport, sport
        return f"{src}:{sport}->{dst}:{dport}/{proto}"
    except Exception:
        return "unknown"


def parse_pcap(path: str | Path) -> tuple[pd.DataFrame | None, str]:
    """Parse a PCAP file and return per-flow feature statistics.

    Returns
    -------
    (df, mode) where mode is one of:
      - ``'FLOW_AND_PACKET_MODE'``  — success
      - ``'PACKET_DATA_INSUFFICIENT'`` — scapy unavailable or not enough IP pkts
    """
    path = Path(path)

    # ------------------------------------------------------------------
    # 1. Import scapy (optional dependency)
    # ------------------------------------------------------------------
    try:
        from scapy.all import PcapReader  # type: ignore
    except ImportError:
        logger.warning(
            "scapy is not installed. Returning PACKET_DATA_INSUFFICIENT."
        )
        return None, "PACKET_DATA_INSUFFICIENT"

    if not path.exists():
        logger.warning("PCAP file not found: %s", path)
        return None, "PACKET_DATA_INSUFFICIENT"

    # ------------------------------------------------------------------
    # 2. Stream packets and accumulate per-flow stats
    # ------------------------------------------------------------------
    flow_ttls: dict[str, list[int]] = defaultdict(list)
    flow_frags: dict[str, int] = defaultdict(int)
    flow_retx: dict[str, int] = defaultdict(int)
    seen_seq: dict[str, set] = defaultdict(set)
    flow_tcp_wins: dict[str, list[int]] = defaultdict(list)
    flow_urgs: dict[str, int] = defaultdict(int)
    flow_payload_lens: dict[str, list[int]] = defaultdict(list)
    flow_entropy: dict[str, list[float]] = defaultdict(list)

    total_pkts = 0
    ip_pkts = 0

    try:
        with PcapReader(str(path)) as reader:
            for pkt in reader:
                total_pkts += 1
                if "IP" not in pkt:
                    continue
                ip_pkts += 1
                key = _flow_key(pkt)
                ip_layer = pkt["IP"]
                ttl = ip_layer.ttl
                flow_ttls[key].append(ttl)
                # Fragment flag (MF bit set or fragment offset > 0)
                flags = ip_layer.flags
                frag_off = ip_layer.frag
                if int(flags) & 0x1 or frag_off > 0:
                    flow_frags[key] += 1
                
                # Payload length
                payload_len = len(pkt["IP"].payload)
                flow_payload_lens[key].append(payload_len)
                
                # Scan entropy (payload entropy approximation per packet)
                if pkt.haslayer("Raw"):
                    raw_data = bytes(pkt["Raw"])
                    if len(raw_data) > 0:
                        counts = np.bincount(np.frombuffer(raw_data, dtype=np.uint8), minlength=256)
                        probs = counts[counts > 0] / len(raw_data)
                        entropy = -np.sum(probs * np.log2(probs))
                        flow_entropy[key].append(entropy)
                
                if "TCP" in pkt:
                    tcp_layer = pkt["TCP"]
                    seq = tcp_layer.seq
                    if seq in seen_seq[key]:
                        flow_retx[key] += 1
                    else:
                        seen_seq[key].add(seq)
                        
                    flow_tcp_wins[key].append(tcp_layer.window)
                    
                    if tcp_layer.flags & 0x20:  # URG flag
                        flow_urgs[key] += 1
                        
    except Exception as exc:
        logger.warning("Failed to parse PCAP '%s': %s", path, exc)
        return None, "PACKET_DATA_INSUFFICIENT"

    if total_pkts == 0 or (ip_pkts / total_pkts) < _MIN_IP_PCT:
        logger.warning(
            "Less than 50%% of packets have an IP layer (%d/%d). "
            "Returning PACKET_DATA_INSUFFICIENT.",
            ip_pkts,
            total_pkts,
        )
        return None, "PACKET_DATA_INSUFFICIENT"

    # ------------------------------------------------------------------
    # 3. Build output DataFrame
    # ------------------------------------------------------------------
    rows = []
    all_keys = set(flow_ttls)
    for key in all_keys:
        ttls = flow_ttls[key]
        tcp_wins = flow_tcp_wins[key]
        payloads = flow_payload_lens[key]
        entropies = flow_entropy[key]
        
        rows.append(
            {
                "flow_key": key,
                "ttl_mean": float(np.mean(ttls)),
                "ttl_std": float(np.std(ttls)) if len(ttls) > 1 else 0.0,
                "fragment_count": flow_frags.get(key, 0),
                "retransmit_count": flow_retx.get(key, 0),
                "tcp_window_mean": float(np.mean(tcp_wins)) if tcp_wins else 0.0,
                "urg_flag_cnt": flow_urgs.get(key, 0),
                "payload_dist_mean": float(np.mean(payloads)) if payloads else 0.0,
                "scan_entropy": float(np.mean(entropies)) if entropies else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    logger.info(
        "PCAP parsed: %d flows from %d IP packets (total %d pkts)",
        len(df),
        ip_pkts,
        total_pkts,
    )
    return df, "FLOW_AND_PACKET_MODE"
