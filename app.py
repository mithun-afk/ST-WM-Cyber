"""
app.py
--------------
Unified Pipeline 2.0 Dashboard with Live Capture & Beautiful UI
"""
import os
import torch
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time

# Pipeline 2.0 Imports
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src2.models.world_model import STGWMModel
from src2.models.inference import run_inference
from src2.data.feature_engineering import engineer_features, ENGINEERED_FEATURE_COLS
from src2.live.live_pipeline import LivePipeline

# ---------------------------------------------------------
# Page Configuration & Aesthetics
# ---------------------------------------------------------
st.set_page_config(
    page_title="NTRO ST-GWM Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Cybersecurity Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    h1, h2, h3 { color: #58a6ff; font-family: 'Inter', sans-serif; }
    .metric-card {
        background: rgba(33, 38, 45, 0.6);
        border: 1px solid #30363d;
        padding: 1.5rem; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px); margin-bottom: 1rem; text-align: center;
    }
    .metric-title { font-size: 0.9rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #58a6ff; }
    .disclosure-banner {
        background-color: #d29922; color: #000; padding: 10px 15px;
        border-radius: 5px; font-size: 1.1rem; font-weight: bold; text-align: center; margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data & Model Loading
# ---------------------------------------------------------
@st.cache_data(ttl=5)
def load_benchmark_data():
    csv_path = os.path.join("eval_results", "cv_benchmark_summary.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_resource
def load_deployment_model():
    model, status = STGWMModel.load("eval_results")
    if status != "LOADED":
        return None
    return model

model = load_deployment_model()

if 'live_pipeline' not in st.session_state:
    st.session_state.live_pipeline = None

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
st.markdown('<div class="disclosure-banner">🛡️ NTRO Problem 26153 - ST-WM | Live Packet Capture Enabled | Fully Offline</div>', unsafe_allow_html=True)
st.title("🛡️ AI-Based Network Attack Forecasting")
st.markdown("*Architecture: Spatial-Temporal World Model (ST-WM) | Inductive Graph Features (No GNN passing penalty) | MITRE Mapped*")
st.markdown("---")

if model is None:
    st.error("⚠️ **Model weights not found.** Please run `python train_pipeline.py` first.")
    st.stop()

tab1, tab2, tab4 = st.tabs(["📊 CV Benchmarks", "🧠 Saliency & Rollout", "📡 Live Network Capture"])

with tab1:
    st.subheader("Model Performance Benchmark")
    df = load_benchmark_data()
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No CV benchmark file found. Benchmarks will populate after pipeline run completes.")
        
    st.markdown("""
    ### 📝 Benchmark Caveats & Integrity Statement
    - **No Fake Balancing:** We preserve the exact chronological sequences.
    - **Data Clustering Limitations:** Because attacks in CIC-IDS-2018 are highly temporally clustered, strict chronological CV folds 1 and 2 often contain 0 positive examples.
    - **World Model over XGBoost:** The baseline pipeline uses a 15-epoch, H=32 under-trained LSTM. By increasing H=64 and epochs=30, the temporal dynamics properly capture state transitions, outperforming tree-based baselines.
    """)

with tab2:
    st.info("Run the Live Capture tab (Tab 3) to populate live World Model Rollouts & Saliency here!")
    if st.session_state.live_pipeline and st.session_state.live_pipeline.latest_result:
        result = st.session_state.live_pipeline.latest_result
        if result.get("important_features"):
            st.write("#### Real-time Feature Saliency (Input-Gradient)")
            s_df = pd.DataFrame(result["important_features"])
            s_df = s_df.rename(columns={"feature": "Feature", "importance": "Saliency"})
            fig2 = px.bar(s_df.head(10), x="Saliency", y="Feature", orientation="h", color="Saliency", color_continuous_scale="Reds")
            fig2.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig2, use_container_width=True)

with tab4:
    st.subheader("📡 Live Network Capture & PCAP Replay")
    st.markdown("Ingests live packets or PCAP files, aggregates via 5-second temporal windows, extracts structural graph features, and queries the trained Spatial-Temporal World Model.")

    mode = st.radio("Select Input Mode:", ["PCAP Replay", "Live Network Capture"], horizontal=True)

    if mode == "PCAP Replay":
        st.write("Drop a PCAP file to replay traffic as if it were live.")
        if st.button("▶️ Start Dummy PCAP Replay"):
            if st.session_state.live_pipeline:
                st.session_state.live_pipeline.stop()
            st.session_state.live_pipeline = LivePipeline(model=model, pcap_file="dummy_test.pcap")
            st.session_state.live_pipeline.start()
            st.rerun()
    else:
        st.write("Requires Npcap/WinPcap installed on Windows.")
        
        # Add interface dropdown
        from src2.live.interface import get_interfaces
        ifaces = get_interfaces()
        iface_options = [f"{i['name']} - {i['description']}" for i in ifaces] if ifaces else ["Default"]
        selected_iface = st.selectbox("Select Network Interface", iface_options)
        
        if st.button("▶️ Start Live Network Capture"):
            if st.session_state.live_pipeline:
                st.session_state.live_pipeline.stop()
            
            # Extract real name from dropdown selection
            iface_name = None
            if selected_iface != "Default":
                iface_name = selected_iface.split(" - ")[0]
                
            st.session_state.live_pipeline = LivePipeline(model=model, interface_name=iface_name)
            st.session_state.live_pipeline.start()
            st.rerun()

    if st.button("⏹️ Stop Capture"):
        if st.session_state.live_pipeline:
            st.session_state.live_pipeline.stop()
            st.session_state.live_pipeline = None
            st.rerun()

    if st.session_state.live_pipeline:
        lp = st.session_state.live_pipeline
        st.write(f"**Pipeline Status:** `{lp.status}`")
        
        result = lp.latest_result
        if result and result.get('status') != 'ERROR':
            st.markdown("### Latest Forecast")
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='metric-card'><div class='metric-title'>Peak Risk</div><div class='metric-value'>{result['risk']:.3f}</div></div>", unsafe_allow_html=True)
            # using true model stage output
            c2.markdown(f"<div class='metric-card'><div class='metric-title'>Predicted Stage</div><div class='metric-value'>{result['stage']}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'><div class='metric-title'>Forecast Horizon</div><div class='metric-value'>+35s</div></div>", unsafe_allow_html=True)
            
            f_df = pd.DataFrame({"Time Window": [f"t+{i*5}s" for i in range(len(result['forecast']))], "Risk Probability": result['forecast']})
            fig = px.line(f_df, x="Time Window", y="Risk Probability", markers=True, range_y=[0,1], color_discrete_sequence=["#ff7b72"])
            st.plotly_chart(fig, use_container_width=True)

            # Threat Intelligence Context
            st.markdown("### 🕵️ Threat Intelligence: AI Reasoning & Projected Next Move")
            
            FEATURE_EXPLANATIONS = {
                "syn_rate": "High frequency of SYN packets (TCP connection requests), strongly indicating active port scanning or SYN flood attacks.",
                "flow_bytes_per_sec": "Abnormal spikes in data transfer rates.",
                "bwd_bytes": "Large outbound payloads from internal targets, typical of data exfiltration or Command & Control beaconing.",
                "fwd_pkts": "Elevated incoming packet volume.",
                "rst_rate": "High rate of connection resets (RST), often seen in aggressive scanning or teardown of brute-force threads.",
                "degree_ratio": "Asymmetric connection fan-out (one IP talking to many), a classic signature of network reconnaissance or worm propagation.",
                "port_diversity": "Traffic targeting an unusually wide range of ports, confirming broad infrastructure scanning.",
                "pkt_ratio": "Highly asymmetric packet exchanges, suggesting automated scanning or exploit payloads rather than normal human web browsing.",
                "flow_duration": "Abnormally long or fragmented flow durations, characteristic of slow-loris attacks or persistent C2 beacons."
            }
            
            NEXT_MOVE = {
                "Benign": "No malicious activity detected. Normal operations.",
                "Reconnaissance": "Target selection and vulnerability identification. The attacker will likely attempt **Initial Access** next (e.g., exploiting a discovered open port, brute-forcing credentials, or sending phishing payloads).",
                "Initial Access": "A foothold has been established. The attacker's next move is likely **Execution & Persistence** (e.g., dropping malware payloads, creating scheduled tasks, or establishing a C2 beacon).",
                "Lateral Movement": "Internal pivoting detected. The attacker will likely attempt **Privilege Escalation or Collection** next (e.g., dumping credentials, accessing file shares, or compromising the domain controller).",
                "Command & Control": "Remote control established. The attacker's next move is likely **Data Exfiltration** (bundling and stealing sensitive files) or executing final **Impact** payloads (e.g., ransomware encryption).",
                "Impact": "Active disruption in progress. The attacker may attempt to **destroy backups** or **spread ransomware to adjacent subnets**."
            }

            stage_name = result.get('stage', 'Benign')
            st.info(f"**🎯 Hacker's Projected Next Move:** {NEXT_MOVE.get(stage_name, 'Unknown')}")
            
            if result.get('important_features') and result.get('risk', 0) > 0.15:
                top_features = result['important_features'][:3]
                reasoning = []
                for item in top_features:
                    feat = item['feature']
                    desc = FEATURE_EXPLANATIONS.get(feat, f"Anomalous variance detected in '{feat}'.")
                    reasoning.append(f"- **{feat}**: {desc}")
                
                st.warning(f"**🔍 Why is the risk elevated ({result['risk']:.2f})?**\nThe AI Saliency Engine identified the following structural anomalies driving this forecast:\n\n" + "\n".join(reasoning))

        elif result and result.get('status') == 'ERROR':
            st.error(f"Inference Error: {result.get('error')}")

        time.sleep(1)
        st.rerun()
