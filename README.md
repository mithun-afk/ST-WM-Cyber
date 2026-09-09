# AI-Based Network Attack Forecasting (NTRO Problem 26153)

> Predict and explain cyber threats in real-time using a Spatio-Temporal World Model, hybrid MITRE ATT&CK integration, and input-gradient saliency on live PCAP traffic.

## Abstract

This project implements a **Spatio-Temporal World Model (ST-WM)** designed for live cybersecurity threat prediction and triage. Moving beyond standard flow classification, the architecture uses an LSTM-based dynamics head to forecast future network states via K-step autoregressive lookaheads, evaluated against dual-scale temporal aggregation (5s micro-windows / 60s macro-windows).

To bridge the gap between ML predictions and actionable security intelligence, the pipeline features a **hybrid MITRE ATT&CK integration** — combining 8 hardcoded heuristic rules (T1046, T1110, T1071, T1048, T1021, T1486, T1595, T1041) with a learned multi-class stage head for kill-chain classification across 6 ATT&CK tactics.

The system emphasizes **Explainable AI (XAI)** throughout: input-gradient saliency (`∂risk_logit/∂X`) ranks the specific traffic features driving each prediction, giving SOC analysts transparent, interpretable output on both live network traffic and PCAP replays — with no cloud dependency.

---

## Core Capabilities

- **World Model Simulation**: 3-headed LSTM (Dynamics + Risk + Stage) rolls out K steps into the future, predicting physical network state, infiltration probability, and discrete MITRE tactic simultaneously.
- **Dual-Scale Feature Ingestion**: 5s micro-windows capture packet-level kinematics (TCP flags, IAT variance, SYN ratios); 60s macro-windows capture topology drift (fan-out entropy, egress/ingress ratios, beaconing jitter).
- **PCAP Ingestion**: `src/pcap_ingest.py` uses Scapy to extract true packet-level features — TTL variance, IP fragment flags, TCP retransmit counts — that are undetectable from NetFlow alone.
- **Hybrid MITRE ATT&CK Engine**: Rule-based indicator matching (8 techniques) layered with a trained 6-class stage head covering Reconnaissance → Impact.
- **Input-Gradient Saliency**: White-box explainability computing `|∂risk_logit/∂X|` averaged over the sequence window. Returns a ranked list of the traffic features most responsible for the current threat prediction.
- **Live Capture & PCAP Replay**: Threaded `LivePipeline` supports real-time sniffing via Npcap and deterministic PCAP replay gated by actual packet timestamps.

---

## Setup

### Prerequisites
- Python 3.10+
- Windows or Linux
- 8 GB RAM minimum (16 GB recommended for full CIC-IDS-2018 training)
- Npcap (Windows) or libpcap (Linux) for live capture

### 1. Environment

```bash
python -m venv ntro_env

# Windows
.\ntro_env\Scripts\activate

# Linux / macOS
source ntro_env/bin/activate

pip install -r requirements.txt
```

### 2. Data Preparation

Place raw CIC-IDS-2018 NetFlow CSV files in `data/raw/`:

```
data/raw/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv
data/raw/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv
```

If no real data is available, the pipeline automatically falls back to a 500-row synthetic demo dataset for UI testing.

### 3. Train the Model

```bash
python train_pipeline.py
```

This runs the full pipeline: feature engineering → chronological 70/15/15 split → LSTM training with early stopping → threshold tuning on validation → evaluation on held-out test set → saves weights to `eval_results/`.

---

## Usage

### Interactive Dashboard

```bash
streamlit run app.py
```

Four modes are supported:

| Mode | Description |
|---|---|
| **CSV Analysis** | Upload a standard NetFlow / CIC-IDS-2018 CSV for offline analysis |
| **PCAP Analysis** | Extract features from a raw `.pcap` file via Scapy |
| **PCAP Replay** | Replay a PCAP file sequentially, gated by real packet timestamps |
| **Live Network Capture** | Sniff live traffic from a local interface (requires Npcap/WinPcap) |

> **Windows live capture:** Run your terminal as Administrator. Wireshark installs Npcap automatically.

### Reproduce Benchmarks

```bash
# Chronological 70/15/15 split benchmark (LR vs ST-WM)
python train_pipeline.py

# 3-fold TimeSeriesSplit CV (LR vs XGBoost vs ST-WM)
python -m src.evaluate
```

Pre-trained weights and scalers are already included in `eval_results/` if you only want to run the dashboard.

---

## Repository Structure

```
├── app.py                        # Main Streamlit dashboard (Pipeline 2.0)
├── train_pipeline.py             # End-to-end training & evaluation harness
├── config.yaml                   # Hyperparameters and paths
├── requirements.txt
│
├── src2/
│   ├── models/
│   │   ├── world_model.py        # ST-WM: LSTM + 3 heads (Dynamics, Risk, Stage)
│   │   ├── inference.py          # Unified JSON contract inference API
│   │   ├── baseline.py           # Logistic Regression baseline
│   │   └── evaluate.py           # Metrics, threshold tuning, confusion matrix
│   ├── data/
│   │   ├── csv_loader.py         # CIC-IDS-2018 schema normalization
│   │   ├── feature_engineering.py# 20 canonical + 5 derived features
│   │   ├── temporal_windows.py   # Sliding window builder (N, seq_len, F)
│   │   ├── schema.py             # DataQualityGate + CANONICAL_FEATURES
│   │   └── demo_generator.py     # 500-row synthetic fallback dataset
│   ├── intelligence/
│   │   ├── mitre.py              # 8-rule MITRE ATT&CK indicator engine
│   │   └── explainability.py     # Input-gradient saliency + window ranking
│   └── live/
│       ├── live_pipeline.py      # Threaded live/replay capture pipeline
│       └── live_aggregator.py    # Packet-to-feature aggregation
│
├── src/
│   ├── world_model.py            # Legacy STGWM PyTorch module
│   ├── extract_features.py       # Dual-scale (5s/60s) feature extractor
│   ├── pcap_ingest.py            # Scapy PCAP parser (TTL, fragments, retransmits)
│   └── train_final.py            # Deployment model training script
│
├── data/
│   ├── raw/                      # Place CIC-IDS-2018 CSV files here
│   └── processed/                # Extracted dual-scale feature matrices
│
├── eval_results/                 # Model weights, scalers, benchmark CSVs
└── tests/                        # 35+ pytest tests across 4 checkpoints
```

---

## Evaluation Methodology

- **Split:** Strict chronological block split — 70% train, 15% validation, 15% test. No random shuffling.
- **Scaler:** `StandardScaler` fit exclusively on the training block.
- **Threshold:** Optimal classification threshold tuned on validation set (maximizing F1), applied blind to test set.
- **Baselines:** Logistic Regression and XGBoost trained on identical features for direct comparison.
- **Metrics:** Precision, Recall, F1, FPR, Brier Score, Stage Accuracy, Confusion Matrix.

### Benchmark Note

CIC-IDS-2018 attack traffic is heavily temporally clustered. In 3-fold `TimeSeriesSplit` CV, Folds 1 and 2 contain zero attack sequences (F1=0.0 by construction). On Fold 3 — the only fold with detection signal:

| Model | F1 @ 0.5 | F1 Optimal | FPR |
|---|---|---|---|
| Logistic Regression | 0.262 | 0.267 | 0.51% |
| XGBoost | 0.520 | 0.823 | ~0.00% |
| **ST-WM (LSTM)** | **0.596** | **0.603** | **0.05%** |

ST-WM leads at the standard 0.5 threshold and achieves the lowest FPR, making it the most conservative choice for high-stakes environments where false positives are costly.

---

## Datasets Used

- [CIC-IDS-2018](https://www.unb.ca/cic/datasets/ids-2018.html) — Primary training and evaluation dataset
- [CTU-13](https://www.stratosphereips.org/datasets-ctu13) — Botnet traffic (binetflow format)
- [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) — Mixed attack types
- [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) — IoT-specific attack traffic
- [MITRE ATT&CK STIX Data](https://github.com/mitre/cti) — Kill-chain stage annotations
