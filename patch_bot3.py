import re
with open('app.py', 'r', encoding='utf-8') as f:
    text = f.read()

pattern = re.compile(r'        with bot_c3:\n            st.markdown\("\*\*Flagged Traffic & Key Indicators\*\*" \).*?elif result and result\.get\("status"\) == "ERROR":', re.DOTALL)

replacement = '''        with bot_c3:
            st.markdown("**Detailed Prediction Insights**")
            indicators = result.get("indicators", {})
            src_ip = indicators.get("src_ip", "N/A")
            dst_ip = indicators.get("dst_ip", "N/A")
            ports = indicators.get("ports", "N/A")
            
            st.markdown(f"""
            <div style="background-color:#161b22; padding:10px; border-radius:5px; border:1px solid #30363d; margin-bottom:10px; font-size:0.9rem;">
                <b>Flagged Source:</b> <span style="color:#f85149">{src_ip}</span><br>
                <b>Target IPs:</b> {dst_ip}<br>
                <b>Exploited Ports:</b> {ports}
            </div>
            """, unsafe_allow_html=True)
            
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
                st.info("Forecast data gathering...")

    elif result and result.get("status") == "ERROR":'''

text = pattern.sub(replacement, text)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(text)
