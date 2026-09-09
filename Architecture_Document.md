# Architecture Document — AI Network Attack Forecasting (Pipeline 2.0)

## Overview
This system uses a **Spatio-Temporal Graph World Model (ST-GWM)** implementation tailored for cybersecurity. It is designed to learn the normal progression of network traffic and identify anomalies, mapping them to the MITRE ATT&CK kill chain.

## Pipeline 2.0 Architecture

1. **Data Ingestion (`src2/data/csv_loader.py` & `src2/data/pcap_parser.py`)**
   - Ingests standard flow logs (CIC-IDS-2018 format) and raw PCAP captures (via Scapy).
   - Strict data quality gates reject captures with insufficient history.

2. **Feature Engineering (`src2/data/feature_engineering.py`)**
   - Maps inputs to 20 canonical features.
   - Derives 5 critical secondary features (`byte_ratio`, `pkt_ratio`, `iat_jitter`, `syn_rate`, `rst_rate`).

3. **Temporal Processing (`src2/data/temporal_windows.py`)**
   - Groups chronological sequences into (N, seq_len, 25) tensors.

4. **ST-GWM Architecture (`src2/models/world_model.py`)**
   - **Core:** 2-layer LSTM ($H=32$) representing the Spatio-Temporal memory.
   - **Head 1 (Dynamics):** Predicts $S_{t+1}$ using SmoothL1Loss in standardized space. Enables $K$-step autoregressive rollouts.
   - **Head 2 (Risk):** Binary FocalLoss ($\alpha=0.75, \gamma=2.0$) predicting probability of compromise.
   - **Head 3 (Stage):** Multi-class CrossEntropy tracking 6 stages of the MITRE kill-chain.

5. **Cybersecurity Intelligence (`src2/intelligence/`)**
   - **Rule Engine (`mitre.py`):** Correlates patterns against 8 specific MITRE techniques (e.g., T1046, T1110, T1071).
   - **Explainability (`explainability.py`):** Uses backpropagated Gradient Saliency to rank the features that most contributed to the risk prediction.

6. **Unified UI API (`src2/models/inference.py` & `app2.py`)**
   - The UI contains **zero** logic. It simply renders a strict JSON contract returned by the `run_inference()` API.
