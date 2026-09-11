# Architecture Document - AI Network Attack Forecasting (Pipeline 2.0)

## Overview
This system uses a **Spatial-Temporal World Model (ST-WM) Ensemble** implementation tailored for cybersecurity. It is designed to learn the normal progression of network traffic, identify protocol anomalies immune to bandwidth spoofing, and map them to the MITRE ATT&CK kill chain.

## Pipeline 2.0 Architecture

1. **Data Ingestion (`src2/data/csv_loader.py` & `src2/data/pcap_parser.py`)**
   - Ingests standard flow logs (CIC-IDS-2018 format) and raw PCAP captures (via Scapy).
   - Real-time aggregation thread converts live PCAP streams into chronological 5-second windows.

2. **Feature Engineering (`src2/data/feature_engineering.py`)**
   - Maps inputs to 20 canonical features.
   - Derives 5 critical secondary features (`byte_ratio`, `pkt_ratio`, `iat_jitter`, `syn_rate`, `rst_rate`).

3. **Temporal Processing (`src2/data/temporal_windows.py`)**
   - Groups chronological sequences into (N, seq_len, 25) tensors.
   - Pre-scaled using robust quantiles (`RobustScaler`) clipped to [-10, 10] to prevent mathematical explosions on outlier traffic.

4. **ST-WM Ensemble Architecture (`src2/models/world_model.py` & `src2/models/inference.py`)**
   - **Risk Anchor (Logistic Regression):** Generates the primary probability of compromise. Weights for pure volumetric features are heavily regularized/zeroed out to prevent False Positives on benign 4K video streams.
   - **Temporal Core:** 2-layer LSTM ($H=64$, Dropout=0.3) representing the Spatio-Temporal memory.
   - **Head 1 (Dynamics):** Predicts $S_{t+1}$ using SmoothL1Loss in standardized space. Enables 12-step (60-second) autoregressive rollouts.
   - **Head 2 (Stage):** Multi-class CrossEntropy tracking 6 stages of the MITRE kill-chain.

5. **Cybersecurity Intelligence (`src2/intelligence/`)**
   - **Explainability (XAI):** Calculates the exact mathematical contribution (`|weight * scaled_input|`) of the Risk Anchor to dynamically rank the features driving the alert.
   - **Rule Engine (`mitre.py`):** Correlates patterns against 8 specific MITRE techniques (e.g., T1046, T1110, T1071).

6. **Unified UI API (`app.py`)**
   - Streamlit dashboard providing 4 operational modes: Offline CSV, Offline PCAP, Live Npcap Capture, and PCAP Replay (for safe red-team demo execution).
