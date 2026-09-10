"""
train_pipeline.py
-----------------
Retrains the ST-WM Spatial-Temporal World Model on CIC-IDS-2018 data.

CRITICAL DESIGN NOTE — Window-Level Training:
  The CIC-IDS-2018 dataset provides per-flow records (one row per TCP connection,
  mean duration ~9ms). But the live pipeline aggregates 5 seconds of raw packets
  into a single feature vector. Training on raw per-flow rows produces a scaler
  that is completely mismatched to live data (different mean, variance, and
  units for every feature). This causes catastrophic false positives in production.

  This pipeline solves the mismatch by pre-aggregating CIC flows into 5-second
  time buckets BEFORE computing the LSTM windows. The scaler and model weights
  are then trained on the same statistical distribution as live captures.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from src2.data.csv_loader import load_and_normalize_csv
from src2.data.feature_engineering import engineer_features, ENGINEERED_FEATURE_COLS
from src2.models.world_model import STGWMModel
from src2.models.baseline import LogisticRegressionBaseline
from src2.models.evaluate import evaluate_model, tune_threshold
from src2.data.temporal_windows import TemporalWindowBuilder


# ---------------------------------------------------------------------------
# Step 0: Config
# ---------------------------------------------------------------------------
WINDOW_SEC = 5           # seconds per temporal window (matches live pipeline)
SEQ_LEN    = 5           # LSTM lookback (number of windows)
EPOCHS     = 30
DATA_PATHS = [
    "data/raw/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "data/raw/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
]
OUT_DIR    = "eval_results"


def aggregate_to_windows(df: pd.DataFrame, window_sec: float = 5.0) -> pd.DataFrame:
    """
    Aggregate per-flow CIC-IDS-2018 records into `window_sec`-second bins.

    This is the critical training-time step that aligns CIC's per-flow features
    with the per-window statistics computed by live_aggregator.py. Without this,
    the StandardScaler is fit on a completely different data distribution from
    live traffic, causing the scaler to massively misrepresent normal traffic.

    Parameters
    ----------
    df         : normalized CIC dataframe with 'Timestamp' column
    window_sec : width of each time bucket in seconds

    Returns
    -------
    DataFrame with one row per time bucket, containing per-window statistics.
    """
    print(f"  Aggregating {len(df):,} per-flow records into {window_sec}s windows...")

    if "Timestamp" not in df.columns:
        # No timestamp — fall back to treating each row as 1 window
        print("  [WARN] No Timestamp column found. Using per-row mode (suboptimal).")
        return df

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values("Timestamp")

    # Create time buckets
    t_min = df["Timestamp"].min()
    df["_bucket"] = ((df["Timestamp"] - t_min).dt.total_seconds() // window_sec).astype(int)

    numeric_cols = ENGINEERED_FEATURE_COLS + ["binary_label"]
    # Only keep numeric cols that exist
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # Aggregate: mean for most features, max for label (any attack in window = attack window)
    agg_dict = {c: "mean" for c in numeric_cols if c != "binary_label"}
    if "binary_label" in df.columns:
        agg_dict["binary_label"] = "max"

    windowed = df.groupby("_bucket").agg(agg_dict).reset_index(drop=True)

    # Preserve stage_idx if present (majority vote within window)
    if "stage_idx" in df.columns:
        stage_per_bucket = df.groupby("_bucket")["stage_idx"].agg(
            lambda x: x.value_counts().index[0]
        ).reset_index(drop=True)
        windowed["stage_idx"] = stage_per_bucket

    # Recalculate ratio features that are distorted by averaging
    if "fwd_pkts" in windowed.columns and "bwd_pkts" in windowed.columns:
        windowed["pkt_ratio"] = windowed["fwd_pkts"] / (windowed["bwd_pkts"] + 1)
    if "fwd_bytes" in windowed.columns and "bwd_bytes" in windowed.columns:
        windowed["byte_ratio"] = windowed["fwd_bytes"] / (windowed["bwd_bytes"] + 1)
    if "syn_flag_cnt" in windowed.columns and "fwd_pkts" in windowed.columns:
        total = windowed["fwd_pkts"] + windowed.get("bwd_pkts", pd.Series(0.0, index=windowed.index))
        windowed["syn_rate"] = windowed["syn_flag_cnt"] / (total + 1)
    if "rst_flag_cnt" in windowed.columns and "fwd_pkts" in windowed.columns:
        total = windowed["fwd_pkts"] + windowed.get("bwd_pkts", pd.Series(0.0, index=windowed.index))
        windowed["rst_rate"] = windowed["rst_flag_cnt"] / (total + 1)
    if "flow_iat_std" in windowed.columns and "flow_iat_mean" in windowed.columns:
        windowed["iat_jitter"] = windowed["flow_iat_std"] / (windowed["flow_iat_mean"] + 1)

    print(f"  Aggregated to {len(windowed):,} temporal windows.")
    return windowed


def run_training():
    print("=" * 60)
    print("NTRO 26153 — ST-WM PIPELINE 2.0 TRAINING")
    print("Window-Level Training for Live Capture Accuracy")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load and concatenate all available raw data files
    # ------------------------------------------------------------------
    dfs = []
    available = [p for p in DATA_PATHS if os.path.exists(p)]
    if not available:
        print("No CIC data files found. Falling back to synthetic data.")
        from src2.data.demo_generator import generate_demo_data
        data_path = generate_demo_data()
        df_raw = pd.read_csv(data_path, comment="#")
        df_raw["binary_label"] = df_raw["binary_label"].astype(float)
        dfs = [df_raw]
    else:
        for path in available:
            print(f"Loading: {path}")
            df_i, mode = load_and_normalize_csv(path)
            print(f"  {len(df_i):,} rows, mode={mode}")
            dfs.append(df_i)

    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"Total raw rows loaded: {len(df_raw):,}")

    # ------------------------------------------------------------------
    # 2. Feature engineering (adds derived ratio features)
    # ------------------------------------------------------------------
    print("Engineering features...")
    df_proc = engineer_features(df_raw)

    # ------------------------------------------------------------------
    # 3. WINDOW AGGREGATION — The critical training/live alignment step
    # ------------------------------------------------------------------
    print("Aggregating into 5-second temporal windows...")
    df_windowed = aggregate_to_windows(df_proc, window_sec=WINDOW_SEC)

    # Fill any NaNs produced by aggregation
    df_windowed = df_windowed.fillna(0.0)

    # Ensure all required feature columns exist
    for col in ENGINEERED_FEATURE_COLS:
        if col not in df_windowed.columns:
            df_windowed[col] = 0.0

    print(f"Training on {len(df_windowed):,} window-level samples.")

    # Class balance report
    if "binary_label" in df_windowed.columns:
        n_attack = int(df_windowed["binary_label"].sum())
        n_benign = len(df_windowed) - n_attack
        print(f"Class distribution — Benign: {n_benign:,}  Attack: {n_attack:,}  "
              f"({100*n_attack/len(df_windowed):.1f}% attack rate)")

    # ------------------------------------------------------------------
    # 4. Build LSTM sequences from windowed data
    # ------------------------------------------------------------------
    print("Building LSTM sequence windows...")
    tb = TemporalWindowBuilder(seq_len=SEQ_LEN)

    if "binary_label" not in df_windowed.columns:
        df_windowed["binary_label"] = 0.0

    try:
        X, y_risk, ts = tb.build(df_windowed, ENGINEERED_FEATURE_COLS)
    except ValueError as e:
        print(f"ERROR building windows: {e}")
        sys.exit(1)

    y_stage = None
    if "stage_idx" in df_windowed.columns:
        y_stage = df_windowed["stage_idx"].values[SEQ_LEN:]

    # ------------------------------------------------------------------
    # 5. Strict chronological split — NO shuffling, NO leakage
    # ------------------------------------------------------------------
    N = len(X)
    split1 = int(0.70 * N)
    split2 = int(0.85 * N)

    X_train, y_train = X[:split1], y_risk[:split1]
    X_val,   y_val   = X[split1:split2], y_risk[split1:split2]
    X_test,  y_test  = X[split2:], y_risk[split2:]

    y_train_stage = y_stage[:split1] if y_stage is not None else None
    y_val_stage   = y_stage[split1:split2] if y_stage is not None else None
    y_test_stage  = y_stage[split2:] if y_stage is not None else None

    print(f"\nChronological split:")
    print(f"  Train : {len(X_train):,} sequences")
    print(f"  Val   : {len(X_val):,} sequences")
    print(f"  Test  : {len(X_test):,} sequences")

    # ------------------------------------------------------------------
    # 6. Baseline: Logistic Regression
    # ------------------------------------------------------------------
    print("\n[BASELINE] Training Logistic Regression...")
    lr_model = LogisticRegressionBaseline()
    lr_model.fit(X_train, y_train)
    lr_thresh = tune_threshold(lr_model, X_val, y_val)
    print(f"  Optimal threshold (Val): {lr_thresh:.4f}")
    lr_metrics = evaluate_model(lr_model, X_test, y_test, y_test_stage, threshold=lr_thresh)

    # ------------------------------------------------------------------
    # 7. ST-WM LSTM World Model
    # ------------------------------------------------------------------
    print(f"\n[ST-WM] Training LSTM World Model ({EPOCHS} epochs)...")
    lstm_model = STGWMModel(input_dim=len(ENGINEERED_FEATURE_COLS))
    log = lstm_model.fit(
        X_train, y_train,
        y_stage=y_train_stage,
        epochs=EPOCHS,
        patience=5
    )
    print(f"  Best val loss: {min(log.get('val_loss', [999])):.4f}")

    # Tune threshold on explicit validation set
    lstm_thresh = tune_threshold(lstm_model, X_val, y_val)
    print(f"  Optimal threshold (Val): {lstm_thresh:.4f}")

    lstm_metrics = evaluate_model(
        lstm_model, X_test, y_test, y_test_stage, threshold=lstm_thresh
    )

    # ------------------------------------------------------------------
    # 8. Save model + tuned threshold
    # ------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    lstm_model.save(OUT_DIR)

    # Save calibrated threshold so inference.py can use it
    import json
    thresh_path = os.path.join(OUT_DIR, "calibrated_threshold.json")
    with open(thresh_path, "w") as f:
        json.dump({"threshold": float(lstm_thresh), "window_sec": WINDOW_SEC}, f, indent=2)
    print(f"\nCalibrated threshold saved to {thresh_path}")

    # ------------------------------------------------------------------
    # 9. Save benchmark CSV for UI display
    # ------------------------------------------------------------------
    rows = []
    for metric in ["precision", "recall", "f1", "fpr", "brier_score"]:
        rows.append({
            "Metric": metric.upper().replace("_", " "),
            "Logistic Regression": round(lr_metrics.get(metric, 0.0), 4),
            "ST-WM (LSTM World Model)": round(lstm_metrics.get(metric, 0.0), 4),
        })
    bench_df = pd.DataFrame(rows)
    bench_df.to_csv(os.path.join(OUT_DIR, "cv_benchmark_summary.csv"), index=False)

    # ------------------------------------------------------------------
    # 10. Final report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL BENCHMARK — HELD-OUT TEST SET")
    print("=" * 60)
    print(f"{'Metric':<22} | {'Logistic Regression':>20} | {'ST-WM (LSTM)':>20}")
    print("-" * 68)
    for r in rows:
        print(f"{r['Metric']:<22} | {r['Logistic Regression']:>20.4f} | {r['ST-WM (LSTM World Model)']:>20.4f}")

    print(f"\nModel saved to {OUT_DIR}/")
    print("Training complete. DONE")


if __name__ == "__main__":
    run_training()
