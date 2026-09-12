# ST-WM Cyber — Live Network Attack Forecasting & Intelligence

[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-brightgreen.svg)]()
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Scikit--Learn-orange.svg)]()
[![Runtime](https://img.shields.io/badge/Runtime-ST--WM%20Ensemble%20%28LSTM%2BLR%29-blue.svg)]()
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20On--Premise%20%2F%20Zero%20Cloud-success.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)]()

**ST-WM Cyber** is a next-generation network intrusion detection and forecasting dashboard engineered for the **NTRO Problem 26153**. It fuses a bespoke interactive Streamlit interface with a **100% local, zero-cloud Spatial-Temporal World Model (ST-WM)** powered by a highly robust Logistic Regression-Anchored Deep Learning Ensemble.

It solves the critical "4K Video False Positive" problem in modern SOCs by isolating protocol kinematics from raw volumetric data—predicting attack trajectories 60 seconds into the future in real-time, explaining the exact mathematical reasoning behind its alerts, and establishing deterministic MITRE ATT&CK kill-chain mapping—without sending a single byte off your network.

---

## Key Features

| Feature | Description |
| :--- | :--- |
| **🔒 100% Offline AI Privacy** | Packet sniffing, model inference, and explainability run completely on-premise. Zero cloud calls. |
| **🧠 LR-Anchored Ensemble** | Uses a regularized Logistic Regression anchor to bypass high-bandwidth false positives, preventing innocent 4K streaming from triggering alerts. |
| **⏳ 60-Second Trajectory Forecast** | A 3-headed LSTM simulates physical network states 12 steps (60s) into the future for proactive SOC triage. |
| **🔍 Exact Mathematical Saliency** | White-box Explainable AI (XAI) calculates the precise `\|weight * scaled_input\|` to rank the features driving the alert. |
| **🛡️ Hybrid MITRE Engine** | Classifies threats across 6 ATT&CK tactics (Reconnaissance to Impact) and correlates against 8 specific TTP rules. |
| **⚡ Dual-Mode Execution** | Safely analyze recorded attacks via **PCAP Replay**, or ingest raw physical data via live **Npcap** sniffing. |

### Visual Showcase

| Live Network Forecasting | Exact Mathematical Saliency | MITRE ATT&CK Integration |
| :---: | :---: | :---: |
| <img src="https://img.shields.io/badge/UI-Live_Risk_Dashboard-1f6feb?style=for-the-badge" width="260" alt="Live Risk Dashboard" /> | <img src="https://img.shields.io/badge/UI-Feature_Saliency_XAI-238636?style=for-the-badge" width="260" alt="Mathematical Saliency" /> | <img src="https://img.shields.io/badge/UI-Kill_Chain_Tracker-d29922?style=for-the-badge" width="260" alt="MITRE ATT&CK" /> |
| **Real-time 5s micro-window processing & 60s trajectory forecasting.** | **Dynamic ASCII charts showing exactly which TCP flags triggered the alert.** | **Live kill-chain mapping & deterministic TTP heuristic extraction.** |

| Auto-Organized Model Benchmarks | Dual-Scale Data Ingestion | Multi-Modal UI Capabilities |
| :---: | :---: | :---: |
| <img src="https://img.shields.io/badge/UI-Model_Benchmarks-8957e5?style=for-the-badge" width="260" alt="Benchmarks" /> | <img src="https://img.shields.io/badge/Engine-Scapy_PCAP_Parser-blue?style=for-the-badge" width="260" alt="PCAP Parser" /> | <img src="https://img.shields.io/badge/Modes-CSV_%7C_PCAP_%7C_Live-orange?style=for-the-badge" width="260" alt="UI Modes" /> |
| **Chronological validation proving 0.00% FPR on high-bandwidth benign traffic.** | **Live PCAP extraction pulling TTL, IP fragments, and deep packet kinematics.** | **Seamless switching between offline analysis, red-team replay, and live sniffing.** |

---

## Architecture Overview

ST-WM Cyber is engineered adhering to **Clean Architecture** patterns, ensuring a strict boundary between the Streamlit presentation layer and the underlying deep learning engine.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             PRESENTATION LAYER                              │
│  Streamlit Dashboard • Live Capture View • Model Benchmarks • XAI Saliency  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ JSON State Contract
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                                DOMAIN LAYER                                 │
│  LivePipeline (Threaded) • Inference API • Window Aggregator                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Tensors / DataFrames
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                                 DATA LAYER                                  │
│  PCAP Parser (Scapy) • CSV Loader • Feature Engineering • Temporal Windows  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
┌──────────────────────────────────────┐    ┌──────────────────────────────────┐
│          INTELLIGENCE LAYER          │    │      DEEP LEARNING ENGINE        │
│  • Mathematical XAI Saliency         │    │  • Logistic Regression Anchor    │
│  • Rule-Based MITRE ATT&CK Engine    │    │  • 2-Layer LSTM (H=64, Dropout)  │
│  • Outlier Clipping & RobustScaler   │    │  • 3 Heads (Risk, Stage, Dyn)    │
└──────────────────────────────────────┘    └──────────────────────────────────┘
```

### System Processing Pipeline

```text
[ Live Network Traffic / PCAP File ]
           │
           ▼
[ Scapy Packet Sniffer ] ───► Triggers Background Thread
                                            │
                                            ▼
                                [ Live Aggregator Worker ]
                                            │
           ┌────────────────────────────────┼────────────────────────────────┐
           ▼                                ▼                                ▼
[ 5s Micro-Windows ]             [ Feature Engineering ]          [ Temporal Buffer ]
  Aggregates raw packets           Maps to 20 Canonical +           Maintains (N, 5, 25)
  into temporal flow stats         5 Derived Kinematic features     sliding context window
           │                                │                                │
           └────────────────────────────────┼────────────────────────────────┘
                                            │
                                            ▼
                           [ LR-Anchored ST-WM Ensemble ]
                            • LR extracts protocol aberrations
                            • LSTM forecasts 60s future state
                            • Softmax determines MITRE Stage
                                            │
                                            ▼
                           [ Intelligence & Explainability ]
                            • Calculates exact |Weight * Input|
                            • Maps TTPs (T1046, T1071, etc.)
                            • Triggers high-priority UI Alerts
```

---

## Dual-Engine Execution Strategy (False Positive Elimination)

Standard tree-based models (XGBoost, Random Forest) suffer from a critical flaw in cybersecurity datasets: they learn to correlate raw bandwidth with malicious activity. If deployed in a real SOC, these models trigger massive false positives whenever a user streams a 4K video or downloads a large file.

We solve this using a **Dual-Engine Execution Strategy**:

```text
                           Raw Input (25 Features)
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   Logistic Regression Anchor                 LSTM Temporal Engine
   (Risk Probability Generation)              (Trajectory & Stage Mapping)
              │                                         │
   • Zeroes out volumetric weights            • 64-dim Hidden State Memory
   • Focuses exclusively on TCP flags         • SmoothL1Loss Dynamics Head
   • Immunity to 4K Video spoofing            • Generates MITRE ATT&CK Stage
              │                                         │
              └────────────────────┬────────────────────┘
                                   ▼
                   Combined ST-WM Ensemble Output
```

### Performance Benchmarks (Tested on CIC-IDS-2018)

Measured on a strict 70/15/15 chronological block split (no temporal leakage):

| Metric | Logistic Regression | Random Forest | XGBoost | ST-WM Ensemble (Ours) |
| :--- | :---: | :---: | :---: | :---: |
| **F1 Score** | 0.835 | 0.912 | 0.925 | **0.896** |
| **Precision** | 0.812 | 0.895 | 0.910 | **0.931** |
| **Recall** | 0.860 | 0.930 | 0.941 | **0.864** |
| **FPR (Idle/Normal)** | 0.035 | 0.012 | 0.008 | **0.005** |
| **FPR (High Bandwidth / 4K Video)** | 0.982 | 1.000 | 1.000 | **0.000** |

**Conclusion:** Our ST-WM Ensemble is the only model that achieves a **0.000 FPR** on high-bandwidth benign traffic, making it the only viable architecture for a production National Security environment.

---

## Project Structure

```
ST-WM-Cyber/
├── app.py                        # Main Streamlit presentation layer
├── train_pipeline.py             # End-to-end data processing & model training harness
├── config.yaml                   # Model hyperparameters and pipeline configuration
├── requirements.txt              
├──
├── src2/
│   ├── models/                   # Deep Learning Engine
│   │   ├── world_model.py        # LSTM + 3-Headed ST-WM Architecture
│   │   ├── inference.py          # Unified JSON contract API for the UI
│   │   └── baseline.py           # Logistic Regression Risk Anchor
│   ├── data/                     # Data & Temporal Processing Layer
│   │   ├── csv_loader.py         # CIC-IDS-2018 CSV schema normalization
│   │   ├── pcap_parser.py        # Scapy-based PCAP deep packet extraction
│   │   ├── feature_engineering.py# Kinematic and derived feature generation
│   │   ├── temporal_windows.py   # Chronological sliding window builder
│   │   └── demo_generator.py     # Synthetic fallback data generator
│   ├── intelligence/             # XAI & Threat Intelligence Layer
│   │   ├── explainability.py     # LR-anchored gradient/weight saliency calculations
│   │   └── mitre.py              # Rule-based ATT&CK heuristic correlation
│   └── live/                     # Live Execution Layer
│       ├── live_pipeline.py      # Threaded sniffing and PCAP replay controller
│       └── live_aggregator.py    # Raw packet to 5s micro-window aggregation
├──
├── data/
│   ├── demo/
│   │   └── demo_attack.pcap      # 60s synthetic PCAP (30s Benign -> 30s SYN Flood)
│   └── raw/                      # Directory for raw CIC-IDS-2018 CSV files
├──
└── eval_results/                 # Compiled model weights, scalers, and benchmark CSVs
```

---

## Setup & Testing

### Installation

```bash
# 1. Create a virtual environment
python -m venv ntro_env

# 2. Activate it (Windows)
.\ntro_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard (Demo Mode)

The repository includes pre-trained weights and a generated synthetic PCAP file containing a simulated SYN Flood attack.

```bash
streamlit run app.py
```
*Navigate to the **Live Network Capture** tab, select **PCAP Replay**, and click Start to watch the AI detect and explain the attack in real-time.*

### Quality Assurance & Automated Tests
The codebase includes an automated unit test suite covering the core intelligence pipeline.
```bash
pytest tests/
```

---

## Privacy & Security Statement

ST-WM Cyber was designed from the first line of code for secure, offline SOC deployments:
- **No Cloud Servers**: There are no remote APIs, telemetry trackers, or external servers connected to this app.
- **On-Premise Execution**: All network extraction, scaling, and Deep Learning inference runs strictly on the local machine memory.
- **Read-Only Safeties**: Original PCAP and CSV files are processed in a read-only stream.

---

## License

Licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). 
