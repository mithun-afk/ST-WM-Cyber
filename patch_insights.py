import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

pattern_mid = re.compile(r'st\.warning\(f"\*\*Predicted Motive:\*\* Potential.*?Action:\*\* \{next_move\}"\)\s*else:\s*st\.error\(f"\*\*Predicted Motive:\*\* Active.*?Action:\*\* \{next_move\}"\)', re.DOTALL)

replacement_mid = '''MOTIVES = {
                "Reconnaissance": "Endpoint enumeration, port scanning, or vulnerability discovery.",
                "Initial Access": "Attempting to breach perimeter defences or exploit exposed services.",
                "Lateral Movement": "Seeking to expand footprint across the subnet or escalate privileges.",
                "Command & Control": "Establishing persistent outbound beaconing to adversary infrastructure.",
                "Impact": "Active data exfiltration, service disruption, or payload detonation."
            }
            motive = MOTIVES.get(stage, "Unknown objective.")
            if stage in ("Reconnaissance", "Initial Access"):
                st.warning(f"**Predicted Motive:** {motive}<br>**Action:** {next_move}")
            else:
                st.error(f"**Predicted Motive:** {motive}<br>**Action:** {next_move}")'''

content = pattern_mid.sub(replacement_mid, content)

pattern_bot3 = re.compile(r'        with bot_c3:.*?</style>.*?</table>.*?<table.*?</table>\n            """, unsafe_allow_html=True)', re.DOTALL)

replacement_bot3 = '''        with bot_c3:
            st.markdown("**Detailed Prediction Insights & Flagged Traffic**")
            indicators = result.get("indicators", {})
            src_ip = indicators.get("src_ip", "N/A")
            dst_ip = indicators.get("dst_ip", "N/A")
            ports = indicators.get("ports", "N/A")
            
            # Show Flagged Traffic Summary
            st.markdown(f"""
            <div style="background-color:#161b22; padding:10px; border-radius:5px; border:1px solid #30363d; margin-bottom:10px; font-size:0.9rem;">
                <b>Flagged Source:</b> <span style="color:#f85149">{src_ip}</span><br>
                <b>Target IPs:</b> {dst_ip}<br>
                <b>Exploited Ports:</b> {ports}
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed Prediction Table (T+1 to T+10)
            st.markdown("**Trajectory Forecast (Next Steps)**")
            if forecast and "forecast_stages" in result:
                table_html = """
                <style>
                .pred-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
                .pred-table th { color: #8b949e; text-align: left; padding: 4px; border-bottom: 1px solid #30363d; }
                .pred-table td { color: #c9d1d9; padding: 4px; border-bottom: 1px solid #21262d; }
                </style>
                <table class="pred-table">
                <tr><th>Time Step</th><th>Predicted Stage</th><th>Risk Prob</th></tr>
                """
                for i in range(min(len(forecast), 10)):
                    prob = forecast[i]
                    f_stage = result["forecast_stages"][i]
                    prob_color = "#3fb950" if prob < 0.3 else "#d29922" if prob < 0.6 else "#f85149"
                    
                    table_html += f"<tr><td>T+{ (i+1)*5 }s</td><td>{f_stage}</td><td><span style='color:{prob_color}'>{prob*100:.1f}%</span></td></tr>"
                table_html += "</table>"
                st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.info("Forecast data gathering...")'''

content = pattern_bot3.sub(replacement_bot3, content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
