"""src2/data/demo_generator.py — creates clearly-labelled synthetic traffic data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Attack stage boundaries (row indices)
# ---------------------------------------------------------------------------
_STAGES = [
    (0,   200, "Benign",          0, dict(syn_flag_cnt=0.1,  flow_bytes_per_sec=1e3)),
    (200, 250, "Reconnaissance",  1, dict(syn_flag_cnt=8.0,  flow_bytes_per_sec=5e2)),
    (250, 350, "InitialAccess",   1, dict(syn_flag_cnt=6.0,  flow_bytes_per_sec=8e4, fwd_bytes=3000)),
    (350, 450, "LateralMovement", 1, dict(ack_flag_cnt=10.0, rst_flag_cnt=0.2, flow_bytes_per_sec=2e4)),
    (450, 500, "Impact",          1, dict(fwd_bytes=9000,    bwd_bytes=9000, flow_bytes_per_sec=1e6)),
]

_TOTAL = 500
_SEED = 42

_FEATURE_DEFAULTS = {
    "flow_duration":      50000,
    "fwd_pkts":           10,
    "bwd_pkts":           8,
    "fwd_bytes":          400,
    "bwd_bytes":          350,
    "flow_iat_mean":      2000,
    "flow_iat_std":       500,
    "syn_flag_cnt":       0,
    "rst_flag_cnt":       0,
    "psh_flag_cnt":       2,
    "ack_flag_cnt":       5,
    "fin_flag_cnt":       1,
    "pkt_len_mean":       60,
    "pkt_len_std":        10,
    "flow_bytes_per_sec": 5000,
    "flow_pkts_per_sec":  200,
    "init_fwd_win_bytes": 8192,
    "active_mean":        1000,
    "idle_mean":          3000,
    "down_up_ratio":      1.0,
}


def generate_demo_data(output_dir: str | Path = "data/demo") -> Path:
    """Generate a 500-row synthetic traffic dataset.

    Parameters
    ----------
    output_dir : str | Path
        Directory where ``demo_traffic.csv`` will be saved.

    Returns
    -------
    Path
        Absolute path to the saved CSV file.
    """
    rng = np.random.default_rng(_SEED)
    rows = []

    # Build timestamps starting at a round hour
    base_ts = pd.Timestamp("2024-01-01 09:00:00")
    timestamps = [base_ts + pd.Timedelta(seconds=i * 2) for i in range(_TOTAL)]

    for start, end, stage_name, label, overrides in _STAGES:
        n = end - start
        for idx in range(n):
            row = {k: v + rng.normal(0, v * 0.05 + 1) for k, v in _FEATURE_DEFAULTS.items()}
            row.update({k: v + rng.normal(0, abs(v) * 0.05 + 0.1) for k, v in overrides.items()})
            # Clamp negatives for counts / sizes
            for col in ["syn_flag_cnt", "rst_flag_cnt", "psh_flag_cnt", "ack_flag_cnt",
                        "fin_flag_cnt", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
                        "flow_bytes_per_sec", "flow_pkts_per_sec", "init_fwd_win_bytes"]:
                row[col] = max(0.0, row[col])
            row["binary_label"] = float(label)
            row["attack_stage"] = stage_name
            row["Timestamp"] = timestamps[start + idx]
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "demo_traffic.csv"

    # Prepend header comment
    header_comment = "# SYNTHETIC DATA — NOT FOR BENCHMARK\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(header_comment)
        df.to_csv(fh, index=False)

    return out_path.resolve()
