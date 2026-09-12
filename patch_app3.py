import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the Feature Saliency choice from sidebar
content = content.replace('["▶️ Live Network Capture", "📊 Model Benchmarks", "🧠 Feature Saliency"]', '["▶️ Live Network Capture", "📊 Model Benchmarks"]')

# Remove the Feature Saliency page entirely
pattern = re.compile(r'# ---------------------------------------------------------------------------\n# PAGE: Feature Saliency\n# ---------------------------------------------------------------------------\nelif page == "\?\? Feature Saliency":.*', re.DOTALL)
new_content = pattern.sub("", content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
