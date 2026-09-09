# Detailed Technical Specification: AI-Based Network Attack Forecasting (Pipeline 2.0)
**Project / Problem Statement:** SIH 26153 (NTRO)

## 1. Executive Summary
This document specifies the architecture and validation methodology of our **Spatial-Temporal World Model (ST-WM)**. 

**Note on Graph Naming convention:** The architecture captures spatial topology via explicit O(1) *Inductive Structural Features* (e.g. port diversity, flow degree ratio, IP interaction symmetries) rather than explicit Graph Neural Network (GNN) message passing. This was an explicit engineering trade-off: traditional message-passing over global adjacency matrices introduces a high latency penalty that violates the strict 5-second Live Capture bound required for active telemetry forecasting. We therefore treat spatial geometry as structural embeddings fused into the temporal LSTM pipeline.

Recognizing that modern cyber attacks (APTs, ransomware) unfold sequentially over time, our system departs from traditional static intrusion detection. Instead of merely classifying a flow as anomalous, our LSTM-based World Model predicts the *future physical state* of the network, enabling proactive K-step attack trajectory forecasting and mapping to the MITRE ATT&CK framework.

## 2. Core Architecture (Pipeline 2.0)

### 2.1. Data Ingestion & Quality Gates
- **Multi-Modal Ingestion:** Supports standard NetFlow CSVs (CIC-IDS-2018 format) and parses raw PCAP files via Scapy.
- **Data Quality Gates:** Automatically detects if essential packet-level telemetry (e.g., `ttl_mean`, `retransmit_count`) is missing. If missing, the pipeline gracefully degrades to `FLOW_ONLY_MODE` without failing or padding with fake data.

### 2.2. Feature Engineering
Inputs are mapped to a robust schema of 25 features:
- **20 Canonical Features:** Core flow statistics (bytes/sec, packets/sec, duration, flags).
- **5 Engineered Features:** Deep contextual indicators (`byte_ratio`, `pkt_ratio`, `iat_jitter`, `syn_rate`, `rst_rate`).

### 2.3. Temporal Processing
- Features are normalized using `StandardScaler` (fitted strictly on training data).
- The pipeline organizes continuous traffic into sliding temporal blocks of $T=5$ windows ($seq\_len=5$).

### 2.4. ST-GWM (LSTM World Model)
The model is implemented in PyTorch and utilizes a three-headed architecture on top of a 2-layer LSTM ($H=32$):
1. **Dynamics Head (The World Model):** Uses `SmoothL1Loss` to predict the 25-dimensional feature vector of the *next* temporal window ($S_{t+1}$). This is the core innovation, allowing autoregressive K-step rollouts.
2. **Risk Head:** Evaluates the probability of compromise using binary `FocalLoss` ($\alpha=0.75, \gamma=2.0$), handling severe class imbalance.
3. **Stage Head:** A multi-class `CrossEntropyLoss` classifier that maps the network state to one of 6 MITRE kill-chain stages (Benign, Reconnaissance, InitialAccess, LateralMovement, CommandAndControl, Impact).

## 3. Cybersecurity Intelligence Layer
To bridge the gap between deep learning and SOC operational needs, we wrap the neural network in an intelligence layer:
- **Gradient Saliency Explanation:** Backpropagates the risk prediction to the input features, ranking the top indicators driving the forecast (e.g., identifying that a sudden spike in `rst_rate` triggered the alert).
- **Rule-Based MITRE Indicators:** Evaluates 8 deterministic MITRE rules (e.g., T1046 Network Scanning, T1071 C2 Application Layer) on the predicted states, grounding the AI in verifiable threat intelligence.

## 4. Empirical Validation Protocol

To prove operational viability and prevent temporal leakage, the system was subjected to a rigorous evaluation against a Logistic Regression baseline on the 1,048,575-row real CIC-IDS2018 dataset.

### 4.1. Strict Chronological Split
Random splitting is invalid for time-series forecasting. We employed a strict, contiguous chronological block split:
- **Train (70%):** Earlier periods. Used for fitting the scaler, baseline, and LSTM.
- **Validation (15%):** Middle periods. Used strictly for tuning the optimal decision threshold (maximizing F1) and LSTM early stopping.
- **Test (15%):** Completely held-out later periods representing future out-of-distribution traffic.

### 4.2. Final Benchmark Metrics (On Held-Out Test Set)
**Methodology:** strict chronological 70% train / 15% validation / 15% held-out test; preprocessing fitted on training data only; threshold selected on validation data.

| Metric | Logistic Regression | LSTM World Model |
| :--- | :--- | :--- |
| **Precision** | 0.5028 | 0.5181 |
| **Recall** | 0.9847 | 0.9660 |
| **F1 Score** | 0.6657 | **0.6744** |
| **FPR** | 0.9686 | **0.8939** |
| **Brier Score** | 0.2948 | **0.2458** |
| **Test Samples** | 157,286 | 157,286 |

## 5. Technical Conclusion & Limitations
**Key Finding:** Temporal world modelling improved F1, probability calibration, and false-positive rate versus Logistic Regression on a strictly chronological held-out test set. 

**The True Differentiator:** While the 7.5% reduction in False Positive Rate and improved F1 are valuable, the primary technical achievement is the model's ability to forecast. Logistic Regression produces a static, reactive alert. The LSTM World Model produces a K-step forward trajectory ($t+1 \dots t+7$), plotting the attack progression over time and mapping it to the MITRE framework before the impact stage is reached.

**Observed Limitations:** Both models exhibit a high False Positive Rate (FPR) on the held-out test set. This suggests substantial difficulty under later traffic distribution shifts (out-of-distribution performance), motivating future work in probability calibration, domain adaptation, and broader multi-dataset validation. Furthermore, a strict "Lead Time" metric evaluation remains unavailable for this specific data subset due to the lack of exact timeline onset annotations.
