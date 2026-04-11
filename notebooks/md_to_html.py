import markdown
import sys
from pathlib import Path

def convert_md_to_html(md_path, html_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Заменяем абсолютные пути картинок на относительные или локальные для браузера
    # Браузер должен уметь их читать
    text = text.replace('C:/Users/Даня/.gemini/antigravity/brain/', 'file:///C:/Users/Даня/.gemini/antigravity/brain/')

    html_content = markdown.markdown(text, extensions=['tables', 'fenced_code'])

    # Добавляем базовые стили (Github-подобные) для красоты
    html_template = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Отчет: Этап 2</title>
<style>
    body {{
        font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
        line-height: 1.6;
        color: #24292e;
        max-width: 900px;
        margin: 0 auto;
        padding: 40px;
    }}
    h1, h2, h3 {{ border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
    img {{ max-width: 100%; height: auto; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #dfe2e5; padding: 6px 13px; }}
    th {{ background-color: #f6f8fa; }}
    blockquote {{ border-left: 0.25em solid #dfe2e5; color: #6a737d; padding: 0 1em; }}
</style>
</head>
<body>
{html_content}
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"HTML successfully generated at: {html_path}")

if __name__ == "__main__":
    md_file = r"C:\Users\Даня\.gemini\antigravity\brain\4845dc6e-05c3-47d5-a3aa-ea9b56675ba0\walkthrough.md"
    html_file = str(Path(__file__).resolve().parents[1] / "report.html")
    convert_md_to_html(md_file, html_file)
