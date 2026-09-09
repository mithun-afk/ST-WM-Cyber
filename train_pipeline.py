import os
import sys
import numpy as np
import pandas as pd
from src2.data.csv_loader import load_and_normalize_csv
from src2.data.feature_engineering import engineer_features, ENGINEERED_FEATURE_COLS
from src2.models.world_model import STGWMModel
from src2.models.baseline import LogisticRegressionBaseline
from src2.models.evaluate import evaluate_model, tune_threshold
from src2.data.temporal_windows import TemporalWindowBuilder

def run_integration():
    print("========================================")
    print("NTRO 26153 - PIPELINE 2.0 VALIDATION AUDIT")
    print("========================================")
    
    data_path = 'data/raw/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv'
    if not os.path.exists(data_path):
        print(f"Large CSV not found at {data_path}. Fallback to synthetic.")
        from src2.data.demo_generator import generate_demo_data
        data_path = generate_demo_data()
        
    print(f"Loading data from {data_path}...")
    if 'demo_traffic' in str(data_path):
        df_raw = pd.read_csv(data_path, comment='#')
    else:
        df_raw, mode = load_and_normalize_csv(data_path)
        
    print(f"Raw data shape: {df_raw.shape}")
    
    print("Engineering features...")
    df_proc = engineer_features(df_raw)
    
    print("Building temporal windows...")
    tb = TemporalWindowBuilder(seq_len=5)
    X, y_risk, ts = tb.build(df_proc, ENGINEERED_FEATURE_COLS)
    y_stage = df_proc['stage_idx'].values[5:] if 'stage_idx' in df_proc.columns else None
    
    # ---------------------------------------------------------
    # Strict Chronological Split
    # ---------------------------------------------------------
    # To prevent temporal leakage, windows do not overlap across splits.
    # We take contiguous blocks. 70% Train, 15% Val, 15% Test.
    N = len(X)
    split1 = int(0.70 * N)
    split2 = int(0.85 * N)
    
    X_train, y_train_risk = X[:split1], y_risk[:split1]
    X_val, y_val_risk     = X[split1:split2], y_risk[split1:split2]
    X_test, y_test_risk   = X[split2:], y_risk[split2:]
    
    y_train_stage = y_stage[:split1] if y_stage is not None else None
    y_val_stage   = y_stage[split1:split2] if y_stage is not None else None
    y_test_stage  = y_stage[split2:] if y_stage is not None else None
    
    print("\n--- TEMPORAL SPLIT (No leakage) ---")
    print(f"Total Windows: {N}")
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    
    # ---------------------------------------------------------
    # Baseline: Logistic Regression
    # ---------------------------------------------------------
    print("\nTraining Baseline: Logistic Regression...")
    lr_model = LogisticRegressionBaseline()
    # Fit only on train
    lr_model.fit(X_train, y_train_risk)
    
    # Tune on Val
    lr_thresh = tune_threshold(lr_model, X_val, y_val_risk)
    print(f"LR Optimal Threshold (Val): {lr_thresh:.3f}")
    
    # Eval on Test
    lr_metrics = evaluate_model(lr_model, X_test, y_test_risk, y_test_stage, threshold=lr_thresh)
    
    # ---------------------------------------------------------
    # ST-GWM (LSTM World Model)
    # ---------------------------------------------------------
    print("\nTraining ST-GWM Model...")
    lstm_model = STGWMModel(input_dim=len(ENGINEERED_FEATURE_COLS))
    
    # We combine train and val for fit() because fit() internally does chronological 80/20.
    # To be extremely strict and avoid double-splitting, we'll explicitly pass train and let 
    # the LSTM split the Train block 80/20 internally. Then we manually tune on the explicit Val block.
    # LSTM scaler is fit strictly on X_train.
    
    # Fit on X_train (internally splits 80/20 for early stopping)
    log = lstm_model.fit(X_train, y_train_risk, y_stage=y_train_stage, epochs=3, patience=2)
    print("LSTM Training Log:", log)
    
    # Tune on explicit X_val
    lstm_thresh = tune_threshold(lstm_model, X_val, y_val_risk)
    print(f"LSTM Optimal Threshold (Val): {lstm_thresh:.3f}")
    
    # Evaluate on explicit X_test
    lstm_metrics = evaluate_model(lstm_model, X_test, y_test_risk, y_test_stage, threshold=lstm_thresh)
    
    print("Saving model weights to eval_results/...")
    os.makedirs('eval_results', exist_ok=True)
    lstm_model.save('eval_results')
    
    # ---------------------------------------------------------
    # Output Comparison Table
    # ---------------------------------------------------------
    print("\n--- FINAL BENCHMARK COMPARISON ON HELD-OUT TEST SET ---")
    print(f"{'Metric':<20} | {'Logistic Regression':<20} | {'LSTM World Model':<20}")
    print("-" * 65)
    
    metrics_to_print = ['precision', 'recall', 'f1', 'fpr', 'brier_score']
    for m in metrics_to_print:
        lr_val = lr_metrics[m]
        lstm_val = lstm_metrics[m]
        print(f"{m:<20} | {lr_val:<20.4f} | {lstm_val:<20.4f}")
        
    print(f"\nLR Confusion Matrix: {lr_metrics['confusion_matrix']}")
    print(f"LSTM Confusion Matrix: {lstm_metrics['confusion_matrix']}")
    print("\nIntegration test complete.")

if __name__ == "__main__":
    run_integration()
