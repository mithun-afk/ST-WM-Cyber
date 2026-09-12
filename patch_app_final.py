with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'st.warning(f"**Predicted Motive:** Potential lateral movement' in line:
        lines[i] = '''            MOTIVES = {
                "Reconnaissance": "Endpoint enumeration, port scanning, or vulnerability discovery.",
                "Initial Access": "Attempting to breach perimeter defences or exploit exposed services.",
                "Lateral Movement": "Seeking to expand footprint across the subnet or escalate privileges.",
                "Command & Control": "Establishing persistent outbound beaconing to adversary infrastructure.",
                "Impact": "Active data exfiltration, service disruption, or payload detonation."
            }
            motive = MOTIVES.get(stage, "Unknown objective.")
            if stage in ("Reconnaissance", "Initial Access"):
                st.warning(f"**Predicted Motive:** {motive}<br>**Action:** {next_move}")
'''
    elif 'st.error(f"**Predicted Motive:** Active exploitation' in line:
        lines[i] = '                st.error(f"**Predicted Motive:** {motive}<br>**Action:** {next_move}")\n'
        
    elif 'st.markdown("**Flagged Traffic & Key Indicators**")' in line:
        lines[i] = '''            st.markdown("**Detailed Prediction Insights & Flagged Traffic**")
'''
    elif '<tr><th>Flagged Source IP(s)</th><td><span style="color:#f85149">{src_ip}</span></td></tr>' in line:
        pass # We'll replace the whole table manually below

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
