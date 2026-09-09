"""src2/data/schema.py — canonical feature list and DataQualityGate."""
import sys
from pathlib import Path

# Allow importing get_config even without sys.path tricks
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src2.config import get_config

# ---------------------------------------------------------------------------
# Canonical 20 numerical features (CIC-IDS-2018 derived, normalised names)
# ---------------------------------------------------------------------------
CANONICAL_FEATURES: list[str] = [
    "flow_duration",
    "fwd_pkts",
    "bwd_pkts",
    "fwd_bytes",
    "bwd_bytes",
    "flow_iat_mean",
    "flow_iat_std",
    "syn_flag_cnt",
    "rst_flag_cnt",
    "psh_flag_cnt",
    "ack_flag_cnt",
    "fin_flag_cnt",
    "pkt_len_mean",
    "pkt_len_std",
    "flow_bytes_per_sec",
    "flow_pkts_per_sec",
    "init_fwd_win_bytes",
    "active_mean",
    "idle_mean",
    "down_up_ratio",
]

# Extra features available only when a PCAP is also provided
PCAP_EXTRA_FEATURES: list[str] = [
    "ttl_mean",
    "ttl_std",
    "fragment_count",
    "retransmit_count",
]


class DataQualityGate:
    """Validates a DataFrame before it enters the ML pipeline."""

    def __init__(self) -> None:
        cfg = get_config()
        self._min_obs: int = cfg["data"]["min_observations"]
        self._missing_threshold: float = cfg["data"]["missing_value_pct_threshold"]

    # ------------------------------------------------------------------
    def validate(self, df, mode: str = "flow") -> tuple[bool, str]:
        """Return (ok, message).  ok=True means the gate is passed."""
        import pandas as pd  # local import to keep module light

        # 1. Minimum row count
        if len(df) < self._min_obs:
            return (
                False,
                f"Too few observations: {len(df)} < {self._min_obs} required.",
            )

        # 2. Required columns present
        required = CANONICAL_FEATURES
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            return (
                False,
                f"Missing required columns: {missing_cols}",
            )

        # 3. Temporal ordering check (if timestamp column exists)
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            if not ts.is_monotonic_increasing:
                return (False, "DataFrame is not sorted in temporal order.")

        # 4. Missing-value percentage check (across canonical features only)
        total_cells = len(df) * len(required)
        missing_cells = df[required].isna().sum().sum()
        if total_cells > 0 and (missing_cells / total_cells) > self._missing_threshold:
            pct = missing_cells / total_cells
            return (
                False,
                f"Missing value percentage {pct:.1%} exceeds threshold "
                f"{self._missing_threshold:.1%}.",
            )

        return (True, "")
