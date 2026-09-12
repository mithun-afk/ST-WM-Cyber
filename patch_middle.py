import re

with open('app.py', 'r', encoding='utf-8') as f:
    text = f.read()

middle_row = '''        # ---------------------------------------------------------------
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
            MOTIVES = {
                "Reconnaissance": "Endpoint enumeration, port scanning, or vulnerability discovery.",
                "Initial Access": "Attempting to breach perimeter defences or exploit exposed services.",
                "Lateral Movement": "Seeking to expand footprint across the subnet or escalate privileges.",
                "Command & Control": "Establishing persistent outbound beaconing to adversary infrastructure.",
                "Impact": "Active data exfiltration, service disruption, or payload detonation."
            }
            motive = MOTIVES.get(stage, "Unknown objective.")
            if stage == "Benign":
                st.success(f"**AI Assessment:** {next_move}")
            elif stage in ("Reconnaissance", "Initial Access"):
                st.warning(f"**Predicted Motive:** {motive}<br>**Action:** {next_move}", icon="⚠️")
            else:
                st.error(f"**Predicted Motive:** {motive}<br>**Action:** {next_move}", icon="🚨")
            
            if features and risk > 0.3:
                st.markdown("**Why was this flagged? (Explainable AI)**")
                for item in features[:3]:
                    feat = item.get("feature", "")
                    imp = item.get("importance", 0)
                    desc = FEATURE_EXPLAIN.get(feat, f"Anomalous variance in {feat}.")
                    st.markdown(f"- **{feat}** *(importance: {imp:.2f})*  \\n  ↳ {desc}")
            else:
                st.markdown("*Traffic matches benign baseline. No anomalous protocol behavior detected.*")

        # ---------------------------------------------------------------
        # Bottom Row
        # ---------------------------------------------------------------'''

text = text.replace('''        # ---------------------------------------------------------------
        # Bottom Row
        # ---------------------------------------------------------------''', middle_row)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(text)
