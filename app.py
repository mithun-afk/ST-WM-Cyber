"""
app.py — ST-WM Network Attack Forecasting Dashboard
NTRO Problem 26153 | Spatial-Temporal World Model | Fully Offline
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src2.models.world_model import STGWMModel
from src2.models.inference import run_inference, STAGE_NAMES
from src2.data.feature_engineering import engineer_features, ENGINEERED_FEATURE_COLS
from src2.live.live_pipeline import LivePipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ST-WM | Network Attack Forecasting",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium dark-mode CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { color: #58a6ff; }
.stSidebar { background-color: #161b22; border-right: 1px solid #30363d; }

.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
.metric-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
.metric-value { font-size: 2.2rem; font-weight: 700; color: #58a6ff; line-height: 1; }
.metric-sub   { font-size: 0.85rem; color: #8b949e; margin-top: 6px; }

.risk-safe    { color: #3fb950 !important; }
.risk-medium  { color: #d29922 !important; }
.risk-high    { color: #f85149 !important; }

.stage-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
    margin: 4px 0;
}
.badge-benign  { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid #3fb950; }
.badge-recon   { background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid #d29922; }
.badge-access  { background: rgba(248,81,73,0.15); color: #f85149; border: 1px solid #f85149; }
.badge-lateral { background: rgba(248,81,73,0.2);  color: #ff7b72; border: 1px solid #ff7b72; }
.badge-c2      { background: rgba(200,50,50,0.25); color: #ffa198; border: 1px solid #ffa198; }
.badge-impact  { background: rgba(200,0,0,0.3);    color: #ff6e5a; border: 1px solid #ff6e5a; }

.disclosure-bar {
    background: linear-gradient(90deg, #1f2937, #1a2744, #1f2937);
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 20px;
    text-align: center;
    font-weight: 600;
    font-size: 0.9rem;
    color: #58a6ff;
    margin-bottom: 20px;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model, status = STGWMModel.load("eval_results")
    if status != "LOADED":
        return None
    return model

@st.cache_data(ttl=10)
def load_benchmark():
    path = os.path.join("eval_results", "cv_benchmark_summary.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

model = load_model()

# Session state
if "live_pipeline" not in st.session_state:
    st.session_state.live_pipeline = None
if "risk_history" not in st.session_state:
    st.session_state.risk_history = []
if "stage_history" not in st.session_state:
    st.session_state.stage_history = []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
STAGE_BADGE = {
    "Benign":            ("badge-benign",  "✅"),
    "Reconnaissance":    ("badge-recon",   "🔍"),
    "Initial Access":    ("badge-access",  "⚠️"),
    "Lateral Movement":  ("badge-lateral", "🔄"),
    "Command & Control": ("badge-c2",      "📡"),
    "Impact":            ("badge-impact",  "🚨"),
}

NEXT_MOVE = {
    "Benign":
        "No malicious activity detected. Network operating normally.",
    "Reconnaissance":
        "Attacker is mapping the network — scanning ports and fingerprinting services. "
        "**Next expected action:** Exploitation of discovered open services or credential brute-force.",
    "Initial Access":
        "A foothold has been established on an endpoint. "
        "**Next expected action:** Payload deployment, persistence mechanisms (scheduled tasks, registry keys), "
        "or establishing a Command & Control beacon.",
    "Lateral Movement":
        "Attacker is pivoting inside the network — accessing file shares, using pass-the-hash, or RDP hopping. "
        "**Next expected action:** Privilege escalation to Domain Admin or access to critical data stores.",
    "Command & Control":
        "Remote control channel established. Attacker is issuing commands to compromised hosts. "
        "**Next expected action:** Bulk data staging and exfiltration, or ransomware deployment.",
    "Impact":
        "Active destructive action in progress — ransomware encryption, data deletion, or DDoS. "
        "**Immediate response required.** Isolate affected segments and invoke IR playbook.",
}

FEATURE_EXPLAIN = {
    "syn_rate":           "Abnormally high SYN packet rate — classic signature of port scanning or SYN flood.",
    "flow_bytes_per_sec": "Extreme throughput spike — consistent with data exfiltration or volumetric DoS.",
    "bwd_bytes":          "Large inbound data volume — may indicate C2 payload delivery or large file transfer.",
    "fwd_pkts":           "Elevated packet count — aggressive scanning, flooding, or high-frequency beaconing.",
    "rst_rate":           "High TCP RST rate — brute-force teardowns or aggressive port scanning.",
    "pkt_ratio":          "Highly asymmetric packet exchange — automated tool traffic, not human browsing.",
    "byte_ratio":         "Asymmetric byte flow — consistent with one-way exfiltration or reflective amplification.",
    "iat_jitter":         "Irregular inter-arrival timing — evasive slow-scan or jittered C2 beaconing.",
    "flow_iat_mean":      "Unusual flow timing — may indicate timing-based evasion or slow-loris attack.",
    "pkt_len_mean":       "Atypical packet size distribution — malformed packets or exploit payload framing.",
    "syn_flag_cnt":       "Elevated SYN count — TCP handshake flooding or broad connection enumeration.",
    "rst_flag_cnt":       "Elevated RST count — aggressive teardown of failed connection attempts.",
    "down_up_ratio":      "Imbalanced traffic direction — upload-heavy flows suggest data theft.",
}

def risk_color(r: float) -> str:
    if r < 0.30: return "#3fb950"
    if r < 0.60: return "#d29922"
    return "#f85149"

def risk_label(r: float) -> str:
    if r < 0.30: return "SAFE"
    if r < 0.60: return "ELEVATED"
    return "CRITICAL"

def stage_badge_html(stage: str) -> str:
    cls, icon = STAGE_BADGE.get(stage, ("badge-recon", "❓"))
    return f'<span class="stage-badge {cls}">{icon} {stage}</span>'

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="disclosure-bar">🛡️ NTRO PROBLEM 26153 &nbsp;|&nbsp; '
    'ST-WM SPATIAL-TEMPORAL WORLD MODEL &nbsp;|&nbsp; '
    'REAL-TIME NETWORK ATTACK FORECASTING &nbsp;|&nbsp; FULLY OFFLINE</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.shields.io/badge/ST--WM-v2.0-blue?style=for-the-badge", use_container_width=True)
    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["▶️ Live Network Capture", "📊 Model Benchmarks", "🧠 Feature Saliency"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Architecture**")
    st.caption("3-Head LSTM: Risk · Stage · Dynamics")
    st.caption("Input: 25 inductive graph features")
    st.caption("Forecast horizon: +35 seconds")
    st.caption("Window size: 5 seconds")

    if model is None:
        st.error("⚠️ Model not trained. Run `python train_pipeline.py`")
    else:
        st.success("✅ Model loaded")

# ---------------------------------------------------------------------------
# PAGE: Live Network Capture
# ---------------------------------------------------------------------------
if page == "▶️ Live Network Capture":
    st.title("📡 Live Network Attack Forecasting")
    st.markdown("*Real-time packet capture → Spatial-Temporal World Model → 35-second attack trajectory forecast*")
    st.markdown("---")

    if model is None:
        st.error("⚠️ Model not found. Please run `python train_pipeline.py` first.")
        st.stop()

    # --- Controls ---
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        mode_select = st.radio(
            "Input Mode",
            ["🔴 Live Network Capture", "📁 PCAP Replay"],
            horizontal=True
        )

    with col_ctrl2:
        st.markdown("<br>", unsafe_allow_html=True)

    if "Live" in mode_select:
        from src2.live.interface import get_interfaces
        ifaces = get_interfaces()
        iface_options = [f"{i['name']} — {i['description']}" for i in ifaces] if ifaces else ["Default"]
        selected_iface = st.selectbox("🔌 Network Interface", iface_options)
        iface_name = selected_iface.split(" — ")[0] if selected_iface != "Default" else None
    else:
        iface_name = None

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("▶️ Start Capture", type="primary", use_container_width=True):
            if st.session_state.live_pipeline:
                st.session_state.live_pipeline.stop()
            st.session_state.risk_history = []
            st.session_state.stage_history = []
            if "PCAP" in mode_select:
                st.session_state.live_pipeline = LivePipeline(model=model, pcap_file="data/demo/demo_traffic.csv")
            else:
                st.session_state.live_pipeline = LivePipeline(model=model, interface_name=iface_name)
            st.session_state.live_pipeline.start()
            st.rerun()

    with btn_col2:
        if st.button("⏹️ Stop Capture", use_container_width=True):
            if st.session_state.live_pipeline:
                st.session_state.live_pipeline.stop()
                st.session_state.live_pipeline = None
            st.rerun()

    st.markdown("---")

    lp = st.session_state.live_pipeline
    if lp is None:
        st.info("👆 Select a network interface and click **Start Capture** to begin real-time monitoring.")
        st.stop()

    # Status badge
    status_color = "#3fb950" if "FORECAST" in lp.status else "#d29922"
    st.markdown(
        f'<div style="background:{status_color}22; border:1px solid {status_color}; '
        f'border-radius:8px; padding:8px 16px; color:{status_color}; '
        f'font-weight:600; margin-bottom:16px;">⚡ {lp.status}</div>',
        unsafe_allow_html=True
    )

    result = lp.latest_result

    if result and result.get("status") == "OK":
        risk = result.get("risk", 0.0)
        stage = result.get("stage", "Benign")
        forecast = result.get("forecast", [])
        features = result.get("important_features", [])

        # Track rolling history
        st.session_state.risk_history.append(risk)
        st.session_state.stage_history.append(stage)
        if len(st.session_state.risk_history) > 60:
            st.session_state.risk_history.pop(0)
            st.session_state.stage_history.pop(0)

        # ---------------------------------------------------------------
        # Top KPI row
        # ---------------------------------------------------------------
        k1, k2, k3, k4 = st.columns(4)
        rc = risk_color(risk)
        rl = risk_label(risk)
        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Risk</div>
                <div class="metric-value" style="color:{rc}">{risk:.1%}</div>
                <div class="metric-sub" style="color:{rc}">● {rl}</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            cls, icon = STAGE_BADGE.get(stage, ("badge-recon", "❓"))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Detected Stage</div>
                <div style="margin-top:12px">{stage_badge_html(stage)}</div>
                <div class="metric-sub">MITRE ATT&CK Mapped</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            peak = max(forecast) if forecast else risk
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Peak Forecast (+35s)</div>
                <div class="metric-value" style="color:{risk_color(peak)}">{peak:.1%}</div>
                <div class="metric-sub">Autoregressive Rollout</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            n_flagged = result.get("n_flagged", 0)
            n_windows = result.get("n_windows", 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Windows Flagged</div>
                <div class="metric-value">{n_flagged}/{n_windows}</div>
                <div class="metric-sub">In current sequence</div>
            </div>""", unsafe_allow_html=True)

        # ---------------------------------------------------------------
        # Charts row: Risk gauge | Rolling history | Forecast
        # ---------------------------------------------------------------
        chart_c1, chart_c2, chart_c3 = st.columns([1, 2, 2])

        with chart_c1:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk * 100,
                number={"suffix": "%", "font": {"size": 28, "color": rc}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                    "bar": {"color": rc, "thickness": 0.25},
                    "bgcolor": "#161b22",
                    "bordercolor": "#30363d",
                    "steps": [
                        {"range": [0, 30],  "color": "rgba(63,185,80,0.1)"},
                        {"range": [30, 60], "color": "rgba(210,153,34,0.1)"},
                        {"range": [60, 100],"color": "rgba(248,81,73,0.1)"},
                    ],
                    "threshold": {"line": {"color": "#ffffff", "width": 2}, "value": risk * 100},
                },
                title={"text": "Risk Score", "font": {"color": "#8b949e", "size": 13}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                height=220, margin=dict(t=40, b=10, l=20, r=20),
                font={"color": "#c9d1d9"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with chart_c2:
            # Rolling 60-second risk history
            hist = st.session_state.risk_history
            x_hist = list(range(-len(hist) + 1, 1))
            fig_hist = go.Figure()
            # Color segments by threshold
            fig_hist.add_trace(go.Scatter(
                x=x_hist, y=hist,
                fill="tozeroy",
                mode="lines",
                line=dict(color=rc, width=2),
                fillcolor=f"rgba({','.join(str(int(int(rc[1:], 16) >> shift & 0xff)) for shift in [16,8,0])},0.15)",
                name="Risk",
            ))
            fig_hist.add_hline(y=0.3, line_dash="dot", line_color="#3fb950", annotation_text="Safe", annotation_font_color="#3fb950")
            fig_hist.add_hline(y=0.6, line_dash="dot", line_color="#d29922", annotation_text="Elevated", annotation_font_color="#d29922")
            fig_hist.update_layout(
                title="Rolling Risk History (last 60 windows)",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                height=220, margin=dict(t=40, b=20, l=10, r=10),
                font={"color": "#c9d1d9"},
                yaxis=dict(range=[0, 1], gridcolor="#21262d", title="Risk"),
                xaxis=dict(gridcolor="#21262d", title="Windows ago"),
                showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with chart_c3:
            # Forecast trajectory
            if forecast:
                fx = [f"t+{(i+1)*5}s" for i in range(len(forecast))]
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=fx, y=forecast,
                    mode="lines+markers",
                    line=dict(color="#58a6ff", width=2),
                    marker=dict(color=[risk_color(r) for r in forecast], size=8),
                    name="Forecast",
                ))
                fig_fc.add_hrect(y0=0, y1=0.3, fillcolor="rgba(63,185,80,0.05)", line_width=0)
                fig_fc.add_hrect(y0=0.3, y1=0.6, fillcolor="rgba(210,153,34,0.05)", line_width=0)
                fig_fc.add_hrect(y0=0.6, y1=1.0, fillcolor="rgba(248,81,73,0.05)", line_width=0)
                fig_fc.update_layout(
                    title="35-Second Attack Trajectory Forecast",
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    height=220, margin=dict(t=40, b=20, l=10, r=10),
                    font={"color": "#c9d1d9"},
                    yaxis=dict(range=[0, 1], gridcolor="#21262d", title="Risk Prob"),
                    xaxis=dict(gridcolor="#21262d"),
                    showlegend=False,
                )
                st.plotly_chart(fig_fc, use_container_width=True)

        # ---------------------------------------------------------------
        # Threat Intelligence
        # ---------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🧠 Threat Intelligence")

        ti_c1, ti_c2 = st.columns([2, 1])

        with ti_c1:
            cls, icon = STAGE_BADGE.get(stage, ("badge-recon", "❓"))
            st.markdown(f"**Detected Stage:** {stage_badge_html(stage)}", unsafe_allow_html=True)

            next_move = NEXT_MOVE.get(stage, "Unknown stage.")
            if stage == "Benign":
                st.success(f"**AI Assessment:** {next_move}")
            elif stage in ("Reconnaissance", "Initial Access"):
                st.warning(f"**AI Assessment:** {next_move}")
            else:
                st.error(f"**AI Assessment:** {next_move}")

            # Saliency-driven reasoning
            if features and risk > 0.20:
                st.markdown("**Why is the AI raising this alert?**")
                for item in features[:3]:
                    feat = item.get("feature", "")
                    imp = item.get("importance", 0)
                    desc = FEATURE_EXPLAIN.get(feat, f"Anomalous variance in `{feat}`.")
                    bar = "█" * int(imp * 20) + "░" * (20 - int(imp * 20))
                    st.markdown(
                        f"- **`{feat}`** `{bar}` *(importance: {imp:.2f})*  \n  ↳ {desc}"
                    )

        with ti_c2:
            # Kill-chain progress
            st.markdown("**MITRE ATT&CK Kill Chain**")
            for i, s in enumerate(STAGE_NAMES):
                if s == "Benign":
                    continue
                is_current = (s == stage)
                past_stages = STAGE_NAMES[:STAGE_NAMES.index(stage) + 1] if stage in STAGE_NAMES else []
                is_past = s in past_stages and not is_current

                if is_current:
                    icon = "🔴"
                    color = "#f85149"
                elif is_past:
                    icon = "🟡"
                    color = "#d29922"
                else:
                    icon = "⚪"
                    color = "#484f58"

                st.markdown(
                    f'<div style="color:{color}; padding:2px 0; font-size:0.9rem;">'
                    f'{icon} {s}</div>',
                    unsafe_allow_html=True
                )

    elif result and result.get("status") == "ERROR":
        st.error(f"⚠️ Inference Error: `{result.get('error')}`")
    else:
        st.info("⏳ Collecting traffic windows... First forecast in a few seconds.")

    # Auto-refresh only while capture is running
    if lp and lp.is_running:
        time.sleep(1)
        st.rerun()

# ---------------------------------------------------------------------------
# PAGE: Model Benchmarks
# ---------------------------------------------------------------------------
elif page == "📊 Model Benchmarks":
    st.title("📊 Model Performance Benchmarks")
    st.markdown("*Evaluated on held-out test set using strict chronological split — no temporal leakage.*")
    st.markdown("---")

    df_bench = load_benchmark()
    if df_bench is not None:
        # Style the table
        st.dataframe(
            df_bench.style.highlight_max(
                subset=[c for c in df_bench.columns if c != "Metric"],
                color="#0d2136", axis=1
            ),
            use_container_width=True, height=220
        )
    else:
        st.warning("No benchmark data found. Run `python train_pipeline.py` to generate benchmarks.")

    st.markdown("---")
    st.markdown("""
    ### Methodology & Integrity Statement

    | Guarantee | Detail |
    |-----------|--------|
    | **No Temporal Leakage** | Chronological split: 70% Train / 15% Val / 15% Test. Windows never overlap across splits. |
    | **Window-Level Training** | Training data aggregated into 5-second windows matching live capture granularity. Scaler fit on window distribution, not CIC per-flow records. |
    | **Calibrated Threshold** | Decision threshold tuned on validation set via F1-maximization. Replaces hard-coded 0.5 which produced systematic false positives. |
    | **Balanced Evaluation** | All metrics computed with `class_weight='balanced'` for Logistic Regression baseline; LSTM loss uses weighted binary cross-entropy. |
    | **Dataset** | CIC-IDS-2018 (Canadian Institute for Cybersecurity). Wednesday capture: 7 attack categories. |
    """)

    st.markdown("---")
    st.markdown("### Architecture Diagram")
    st.code("""
Input: [5 × 25 feature window]
         │
         ▼
  ┌──────────────┐
  │  LSTM Layers │  h=64, L=2, dropout=0.3
  └──────┬───────┘
         │ hidden state h_t
    ┌────┴────┐────────────┐
    ▼         ▼            ▼
 Risk Head  Stage Head  Dynamics Head
 sigmoid()  softmax(6)  linear(25)
    │         │            │
 P(attack)  MITRE Stage  ŜS_{t+1}
             
  Autoregressive rollout: feed ŜS_{t+1} back as input for +35s forecast
    """, language="text")

# ---------------------------------------------------------------------------
# PAGE: Feature Saliency
# ---------------------------------------------------------------------------
elif page == "🧠 Feature Saliency":
    st.title("🧠 Feature Saliency & Explainability")
    st.markdown("*Input-gradient saliency — showing which features drove the latest prediction.*")
    st.markdown("---")

    lp = st.session_state.live_pipeline
    if lp and lp.latest_result and lp.latest_result.get("important_features"):
        result = lp.latest_result
        features = result["important_features"]

        feat_df = pd.DataFrame(features).rename(columns={"feature": "Feature", "importance": "Saliency"})
        feat_df = feat_df.sort_values("Saliency", ascending=True)

        fig_sal = go.Figure(go.Bar(
            x=feat_df["Saliency"],
            y=feat_df["Feature"],
            orientation="h",
            marker=dict(
                color=feat_df["Saliency"],
                colorscale=[[0, "#3fb950"], [0.5, "#d29922"], [1.0, "#f85149"]],
                showscale=True,
                colorbar=dict(title="Saliency", tickfont=dict(color="#c9d1d9"), titlefont=dict(color="#c9d1d9")),
            ),
        ))
        fig_sal.update_layout(
            title="Feature Importance (Input-Gradient Saliency)",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font={"color": "#c9d1d9"},
            height=400,
            margin=dict(t=50, b=20, l=10, r=10),
            xaxis=dict(gridcolor="#21262d", title="Gradient × Input"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig_sal, use_container_width=True)

        st.markdown("### Feature Descriptions")
        for item in features:
            feat = item.get("feature", "")
            desc = FEATURE_EXPLAIN.get(feat, f"Network flow statistic: `{feat}`")
            st.markdown(f"- **`{feat}`** — {desc}")
    else:
        st.info("💡 Start a Live Network Capture first. Saliency data populates here after the first inference.")
        st.markdown("""
        ### How Input-Gradient Saliency Works
        
        For each prediction, the AI computes:
        
        $$\\text{Saliency}_i = \\left| \\frac{\\partial \\hat{y}}{\\partial x_i} \\right| \\cdot |x_i|$$
        
        This measures how much a unit change in feature $x_i$ would change the risk prediction $\\hat{y}$.
        Features with high saliency are the **primary drivers** of the alert — the equivalent of a human
        analyst saying *"I flagged this because the SYN rate spiked three standard deviations above baseline."*
        """)
