---
title: "AI-Based Network Attack Forecasting"
subtitle: "Spatial-Temporal World Model (ST-WM) Ensemble"
author: "Team AI Forecaster"
theme: "dark"
---

# Slide 1: The Problem
## Traditional IDS Failures in Modern Environments
- **The False Positive Crisis:** Standard machine learning algorithms (Random Forest, XGBoost) heavily correlate volumetric spikes with attacks.
- **The Blind Spot:** In a live environment, normal activities like streaming a 4K video trigger catastrophic false positives.
- **Static Detection vs Dynamic Trajectories:** Traditional IDS classifies single flows. Modern Advanced Persistent Threats (APTs) execute multi-stage attacks over time.
- **The Goal:** A system that learns protocol aberrations rather than volume, forecasting attack trajectories up to 60 seconds into the future.

---

# Slide 2: Pipeline 2.0 Architecture
## Spatial-Temporal World Model (ST-WM)
- **Dual-Scale Feature Ingestion:** Combines Flow logs and deep Packet telemetry (TCP Window, URG, Scan Entropy) for rich inductive structural features.
- **Real-Time Temporal Aggregator:** Translates continuous packets into unified 5-second temporal windows.
- **Robust Pre-scaling:** Clips extreme volumetric outliers to maintain mathematical stability.
- **The 3-Headed Network:**
  1. **Risk Anchor:** Logistic regression evaluating immediate probability of compromise.
  2. **Stage Head:** Multi-class classification into MITRE ATT&CK stages.
  3. **Dynamics Head:** Predicts the next physical state, enabling autoregressive rollouts.

---

# Slide 3: Explanatory AI (XAI) & Threat Intelligence
## Bridging AI and Analyst Operations
- **Mathematical Saliency (XAI):** Calculates absolute feature contributions to dynamically expose the anomalous protocol behaviors driving the alert.
- **Temporal Attention:** Highlights the specific 5-second windows that deviate from the baseline.
- **Rule-Based Grounding:** The neural network outputs are verified against 8 deterministic MITRE rules (e.g., T1046 Network Scanning) to map the attack trajectory reliably.
- **Fully Offline:** Engineered for air-gapped National Security environments—no cloud APIs.

---

# Slide 4: Empirical Validation & Benchmarks
## 0% False Positive Rate on High-Bandwidth Benign Traffic
- **Strict Chronological Split:** Avoids data leakage by evaluating strictly on held-out future periods (15% test set).
- **The XGBoost Failure:** While tree-based models achieve 0.925 F1 on the base dataset, they hit a 100% FPR when tested on normal high-bandwidth streams.
- **ST-WM Ensemble Success:** Our LR-Anchored Ensemble maintains a 0.896 F1 score while achieving a **0.000 FPR on benign high-volume traffic**, proving its production viability.

---

# Slide 5: Real-Time Demonstration UI
## Interactive & Tactical Response
- **4 Operational Modes:** Live Network Capture, PCAP Replay, PCAP Upload, and CSV Upload.
- **Live 60-Second Forecast:** A dynamic gauge and timeline showing the trajectory of an attack before it reaches maximum impact.
- **Threat Intelligence Panel:** Immediate tactical advice based on the detected MITRE stage (e.g., "Isolate affected subnet") alongside top risk-driving features.
- **The Future of IDS:** Moving from reactive alert fatigue to proactive attack trajectory forecasting.
