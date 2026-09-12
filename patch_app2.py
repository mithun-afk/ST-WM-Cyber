import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = re.compile(r'    st\.markdown\("---"\)\n    st\.markdown\("### 🗃️ Offline Batch Analysis & Evaluation"\).*?                    st\.code\(traceback\.format_exc\(\)\)', re.DOTALL)

new_content = pattern.sub("", content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
