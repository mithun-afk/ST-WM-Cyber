import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = re.compile(r'        # ---------------------------------------------------------------\n        # Top Row.*?elif result and result\.get\("status"\) == "ERROR":', re.DOTALL)

replacement = '''        # ---------------------------------------------------------------
        # Top Row
        # ---------------------------------------------------------------
        top_c1, top_c2, top_c3 = st.columns([1, 2, 1])

        with top_c1:
            st.markdown("**Infiltration Probability**")
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
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                height=220, margin=dict(t=20, b=10, l=20, r=20),
                font={"color": "#c9d1d9"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with top_c2:
            st.markdown("**Infiltration Prediction (Next 10 Steps)**")
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
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    height=220, margin=dict(t=20, b=20, l=10, r=10),
                    font={"color": "#c9d1d9"},
                    yaxis=dict(range=[0, 1], gridcolor="#21262d", title="Risk Prob"),
                    xaxis=dict(gridcolor="#21262d"),
                    showlegend=False,
                )
                st.plotly_chart(fig_fc, use_container_width=True)
            else:
                st.info("Gathering history...")

        with top_c3:
            st.markdown("**MITRE ATT&CK Mapping**")
            for i, s in enumerate(STAGE_NAMES):
                if s == "Benign": continue
                is_current = (s == stage)
                past_stages = STAGE_NAMES[:STAGE_NAMES.index(stage) + 1] if stage in STAGE_NAMES else []
                is_past = s in past_stages and not is_current
                if is_current:
                    icon, color, weight = "🔴", "#f85149", "bold"
                elif is_past:
                    icon, color, weight = "🟡", "#d29922", "normal"
                else:
                    icon, color, weight = "⚪", "#484f58", "normal"
                st.markdown(
                    f'<div style="color:{color}; padding:4px 0; font-size:0.95rem; font-weight:{weight};">'
                    f'{icon} &nbsp; {s}</div>',
                    unsafe_allow_html=True
                )

        # ---------------------------------------------------------------
        # Middle Row: Rolling History & AI Threat Reasoning
        # ---------------------------------------------------------------
        st.markdown("---")
        mid_c1, mid_c2 = st.columns([1, 1.5])

        with mid_c1:
            st.markdown("**Rolling Risk History (Safe / Elevated / Danger)**")
            hist = st.session_state.risk_history
            x_hist = list(range(-len(hist) + 1, 1))
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=x_hist, y=hist, fill="tozeroy", mode="lines",
                line=dict(color=rc, width=2),
                fillcolor="rgba(88,166,255,0.15)",
                name="Risk",
            ))
            fig_hist.add_hline(y=0.3, line_dash="dot", line_color="#3fb950", annotation_text="Safe", annotation_font_color="#3fb950")
            fig_hist.add_hline(y=0.6, line_dash="dot", line_color="#d29922", annotation_text="Elevated", annotation_font_color="#d29922")
            fig_hist.add_hline(y=0.9, line_dash="dot", line_color="#f85149", annotation_text="Danger", annotation_font_color="#f85149")
            fig_hist.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                height=220, margin=dict(t=20, b=20, l=10, r=10),
                font={"color": "#c9d1d9"},
                yaxis=dict(range=[0, 1], gridcolor="#21262d", title="Risk"),
                xaxis=dict(gridcolor="#21262d", title="Windows ago"),
                showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with mid_c2:
            st.markdown("**AI Threat Reasoning & Analyst Actions**")
            next_move = NEXT_MOVE.get(stage, "Unknown stage.")
            if stage == "Benign":
                st.success(f"**AI Assessment:** {next_move}")
            elif stage in ("Reconnaissance", "Initial Access"):
                st.warning(f"**Predicted Motive:** Potential lateral movement preparation or endpoint enumeration.<br>**Action:** {next_move}")
            else:
                st.error(f"**Predicted Motive:** Active exploitation, data exfiltration, or immediate operational disruption.<br>**Action:** {next_move}")
            
            if features and risk > 0.3:
                st.markdown("**Why was this flagged? (Explainable AI)**")
                for item in features[:3]:
                    feat = item.get("feature", "")
                    imp = item.get("importance", 0)
                    desc = FEATURE_EXPLAIN.get(feat, f"Anomalous variance in {feat}.")
                    st.markdown(f"- **{feat}** *(importance: {imp:.2f})*  \n  ↳ {desc}")
            else:
                st.markdown("*Traffic matches benign baseline. No anomalous protocol behavior detected.*")


        # ---------------------------------------------------------------
        # Bottom Row
        # ---------------------------------------------------------------
        st.markdown("---")
        bot_c1, bot_c2, bot_c3 = st.columns([1.5, 1, 1.2])

        with bot_c1:
            st.markdown("**Network State Representation (Graph)**")
            edges = result.get("indicators", {}).get("edges", [])
            if edges:
                import networkx as nx
                G = nx.Graph()
                for e in edges:
                    G.add_edge(e["source"], e["target"], weight=e["weight"])
                    
                pos = nx.spring_layout(G, seed=42)
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1.5, color="rgba(139, 148, 158, 0.5)"),
                    hoverinfo="none", mode="lines")
                    
                node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x); node_y.append(y)
                    degree = G.degree(node)
                    node_text.append(f"<b>IP:</b> {node}<br><b>Connections:</b> {degree}")
                    node_color.append("#f85149" if risk > 0.6 else "#58a6ff")
                    node_size.append(15 + 5 * degree)
                    
                node_trace = go.Scatter(
                    x=node_x, y=node_y, mode="markers+text",
                    textposition="bottom center", text=[str(n) for n in G.nodes()],
                    hoverinfo="text", hovertext=node_text,
                    marker=dict(color=node_color, size=node_size, line_width=2, line_color="#ffffff")
                )
                        
                fig_net = go.Figure(data=[edge_trace, node_trace],
                             layout=go.Layout(
                                showlegend=False, hovermode="closest",
                                margin=dict(b=10,l=10,r=10,t=10),
                                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=300
                                )
                )
                st.plotly_chart(fig_net, use_container_width=True)
            else:
                st.info("No connections mapped yet.")

        with bot_c2:
            st.markdown("**Top Contributing Features (SHAP)**")
            if features:
                import pandas as pd
                feat_df = pd.DataFrame(features).rename(columns={"feature": "Feature", "importance": "Saliency"})
                feat_df = feat_df.sort_values("Saliency", ascending=True).tail(5)
                fig_sal = go.Figure(go.Bar(
                    x=feat_df["Saliency"], y=feat_df["Feature"],
                    orientation="h",
                    marker=dict(
                        color=feat_df["Saliency"],
                        colorscale=[[0, "#3fb950"], [0.5, "#d29922"], [1.0, "#f85149"]],
                    ),
                ))
                fig_sal.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font={"color": "#c9d1d9"}, height=300,
                    margin=dict(t=10, b=10, l=10, r=10),
                    xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
                )
                st.plotly_chart(fig_sal, use_container_width=True)
            else:
                st.success("Traffic matches benign baseline.")

        with bot_c3:
            st.markdown("**Flagged Traffic & Key Indicators**")
            indicators = result.get("indicators", {})
            src_ip = indicators.get("src_ip", "N/A")
            dst_ip = indicators.get("dst_ip", "N/A")
            ports = indicators.get("ports", "N/A")
            
            st.markdown(f"""
            <style>
            .indicator-table {{ width: 100%; border-collapse: collapse; font-size: 0.90rem; margin-top: 5px; }}
            .indicator-table th, .indicator-table td {{ border-bottom: 1px solid #30363d; padding: 10px 4px; text-align: left; }}
            .indicator-table th {{ color: #8b949e; font-weight: 500; width: 40%; }}
            .indicator-table td {{ color: #c9d1d9; font-weight: 600; word-break: break-all; }}
            </style>
            <table class="indicator-table">
                <tr><th>Flagged Source IP(s)</th><td><span style="color:#f85149">{src_ip}</span></td></tr>
                <tr><th>Flagged Target IP(s)</th><td>{dst_ip}</td></tr>
                <tr><th>Exploited Ports</th><td>{ports}</td></tr>
                <tr><th>Analysis Window Size</th><td>5 Sec</td></tr>
                <tr><th>Threat Status</th><td>{rl}</td></tr>
            </table>
            """, unsafe_allow_html=True)

    elif result and result.get("status") == "ERROR":'''

new_content = pattern.sub(replacement, content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
