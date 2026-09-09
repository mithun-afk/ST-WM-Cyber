"""tests/test_checkpoint1.py — Checkpoint 1 test suite."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SAMPLE_CSV = Path("D:/sih26/data/raw/cic_ids_2018_sample.csv")

_N_CANONICAL = 20


def _make_valid_df(n: int = 50) -> pd.DataFrame:
    """Synthetic DataFrame that passes the DataQualityGate."""
    from src2.data.schema import CANONICAL_FEATURES

    rng = np.random.default_rng(0)
    data = {col: rng.random(n) for col in CANONICAL_FEATURES}
    data["binary_label"] = rng.integers(0, 2, n).astype(float)
    base = pd.Timestamp("2024-01-01")
    data["Timestamp"] = [base + pd.Timedelta(seconds=i) for i in range(n)]
    df = pd.DataFrame(data)
    return df


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_config_loads():
    """Config loads successfully and contains all required top-level keys."""
    from src2.config import get_config

    cfg = get_config()
    for key in ("paths", "data", "model", "labels"):
        assert key in cfg, f"Missing top-level config key: {key}"
    assert "seq_len" in cfg["data"]
    assert "attack_keywords" in cfg["labels"]


def test_schema_canonical_features():
    """CANONICAL_FEATURES must have exactly 20 entries."""
    from src2.data.schema import CANONICAL_FEATURES

    assert len(CANONICAL_FEATURES) == _N_CANONICAL, (
        f"Expected 20 canonical features, got {len(CANONICAL_FEATURES)}"
    )


def test_data_quality_gate_pass():
    """DataQualityGate should pass on a well-formed synthetic DataFrame."""
    from src2.data.schema import DataQualityGate

    gate = DataQualityGate()
    df = _make_valid_df(50)
    ok, msg = gate.validate(df)
    assert ok, f"Gate failed unexpectedly: {msg}"


def test_data_quality_gate_fail_rows():
    """Gate must fail when row count is below min_observations."""
    from src2.data.schema import DataQualityGate

    gate = DataQualityGate()
    # Only 5 rows — well below default min_observations (20)
    df = _make_valid_df(5)
    ok, msg = gate.validate(df)
    assert not ok
    assert "Too few observations" in msg


def test_data_quality_gate_fail_missing():
    """Gate must fail when missing-value percentage exceeds threshold."""
    from src2.data.schema import DataQualityGate, CANONICAL_FEATURES

    gate = DataQualityGate()
    df = _make_valid_df(50)
    # Introduce >30 % NaN across canonical features
    for col in CANONICAL_FEATURES:
        df.loc[: len(df) // 2, col] = np.nan
    ok, msg = gate.validate(df)
    assert not ok
    assert "Missing value percentage" in msg or "missing" in msg.lower()


def test_csv_loader_sample():
    """CSV loader must return a DataFrame with all CANONICAL_FEATURES."""
    from src2.data.csv_loader import load_and_normalize_csv
    from src2.data.schema import CANONICAL_FEATURES

    if not _SAMPLE_CSV.exists():
        pytest.skip(f"Sample CSV not found: {_SAMPLE_CSV}")

    df, mode = load_and_normalize_csv(_SAMPLE_CSV)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in CANONICAL_FEATURES:
        assert col in df.columns, f"Missing canonical column: {col}"
    assert mode == "FLOW_ONLY_MODE"


def test_feature_engineering():
    """Engineered DataFrame must have exactly 25 feature columns."""
    from src2.data.feature_engineering import engineer_features, ENGINEERED_FEATURE_COLS

    df = _make_valid_df(50)
    df_eng = engineer_features(df)
    missing = [c for c in ENGINEERED_FEATURE_COLS if c not in df_eng.columns]
    assert len(missing) == 0, f"Missing engineered columns: {missing}"
    assert len(ENGINEERED_FEATURE_COLS) == 25, (
        f"Expected 25 engineered feature cols, got {len(ENGINEERED_FEATURE_COLS)}"
    )


def test_temporal_windows():
    """TemporalWindowBuilder must produce correctly shaped arrays."""
    from src2.data.feature_engineering import engineer_features, ENGINEERED_FEATURE_COLS
    from src2.data.temporal_windows import TemporalWindowBuilder

    df = _make_valid_df(50)
    df = engineer_features(df)
    tb = TemporalWindowBuilder(seq_len=5)
    X, y, ts = tb.build(df, ENGINEERED_FEATURE_COLS)

    expected_n = len(df) - 5  # 45 windows
    assert X.shape == (expected_n, 5, 25), f"X shape mismatch: {X.shape}"
    assert y.shape == (expected_n,), f"y shape mismatch: {y.shape}"
    assert len(ts) == expected_n
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_temporal_windows_insufficient():
    """TemporalWindowBuilder must raise ValueError with the exact message format."""
    from src2.data.temporal_windows import TemporalWindowBuilder
    from src2.data.feature_engineering import engineer_features, ENGINEERED_FEATURE_COLS

    df = _make_valid_df(6)          # 6 rows, seq_len=5 → only 1 window possible
    df = engineer_features(df)
    tb = TemporalWindowBuilder(seq_len=10)  # needs 11 rows

    with pytest.raises(ValueError) as exc_info:
        tb.build(df, ENGINEERED_FEATURE_COLS)

    msg = str(exc_info.value)
    assert "Insufficient temporal data" in msg
    assert "Required" in msg
    assert "Available" in msg
    assert "Please provide a longer capture" in msg


def test_demo_generator():
    """Demo generator must create a CSV with exactly 500 rows."""
    from src2.data.demo_generator import generate_demo_data

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = generate_demo_data(output_dir=tmpdir)
        assert Path(out_path).exists()
        # Skip the comment line at top
        df = pd.read_csv(out_path, comment="#")
        assert len(df) == 500, f"Expected 500 rows, got {len(df)}"
        assert "binary_label" in df.columns
        assert "attack_stage" in df.columns


def test_baseline_fit_predict():
    """Baseline must fit, predict, and evaluate with an f1 metric."""
    from src2.data.feature_engineering import engineer_features, ENGINEERED_FEATURE_COLS
    from src2.data.temporal_windows import TemporalWindowBuilder
    from src2.models.baseline import LogisticRegressionBaseline

    df = _make_valid_df(80)
    df = engineer_features(df)
    tb = TemporalWindowBuilder(seq_len=5)
    X, y, _ = tb.build(df, ENGINEERED_FEATURE_COLS)
    X_flat = X.reshape(len(X), -1)

    bl = LogisticRegressionBaseline()
    assert bl.status == "NOT_TRAINED"
    bl.fit(X_flat, y)
    assert bl.status == "TRAINED"

    preds = bl.predict(X_flat)
    assert "risk_prob" in preds
    assert preds["risk_prob"].shape == (len(X_flat),)

    results = bl.evaluate(X_flat, y)
    assert "f1" in results
    assert "accuracy" in results
    assert "fpr" in results
    assert "confusion_matrix" in results


def test_baseline_save_load():
    """Saved and reloaded baseline must produce identical predictions."""
    from src2.data.feature_engineering import engineer_features, ENGINEERED_FEATURE_COLS
    from src2.data.temporal_windows import TemporalWindowBuilder
    from src2.models.baseline import LogisticRegressionBaseline

    df = _make_valid_df(80)
    df = engineer_features(df)
    tb = TemporalWindowBuilder(seq_len=5)
    X, y, _ = tb.build(df, ENGINEERED_FEATURE_COLS)
    X_flat = X.reshape(len(X), -1)

    bl = LogisticRegressionBaseline()
    bl.fit(X_flat, y)
    preds_before = bl.predict(X_flat)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "baseline.joblib"
        bl.save(save_path)
        bl2 = LogisticRegressionBaseline.load(save_path)
        preds_after = bl2.predict(X_flat)

    np.testing.assert_array_equal(preds_before["risk_prob"], preds_after["risk_prob"])
    assert bl2.status == "TRAINED"
