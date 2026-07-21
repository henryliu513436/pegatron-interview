"""
將 report.md 轉為列印用的 report.html，供瀏覽器 Print-to-PDF 匯出。

用法:
    pip install markdown        # 建置工具依賴，非執行期依賴
    python3 build_report.py
    open report.html            # 瀏覽器 Cmd+P → 另存為 PDF

不採用 pandoc + xelatex 的原因：CJK 字型設定成本高於效益。
瀏覽器列印可直接使用系統中文字型，且能即時預覽頁數。
"""

import sys
from pathlib import Path

# ---------- 頁數控制的主要旋鈕 ----------
# 報告含 5 張圖與 4 張表，不限制圖高必定超出 3 頁。
# 若列印預覽超頁，優先調降此值（建議下限 45mm）。
IMAGE_MAX_HEIGHT_MM = 60

BASE_FONT_PT = 10.5
PAGE_MARGIN_MM = 15

SOURCE_PATH = Path(__file__).parent / "report.md"
OUTPUT_PATH = Path(__file__).parent / "report.html"

PRINT_CSS = f"""
@page {{
    size: A4;
    margin: {PAGE_MARGIN_MM}mm;
}}

body {{
    font-family: "PingFang TC", "Heiti TC", "Noto Sans TC", sans-serif;
    font-size: {BASE_FONT_PT}pt;
    line-height: 1.5;
    color: #1a1a1a;
    max-width: 180mm;
    margin: 0 auto;
    padding: {PAGE_MARGIN_MM}mm;
}}

h1 {{ font-size: 17pt; margin: 0 0 4mm; border-bottom: 2px solid #333; padding-bottom: 2mm; }}
h2 {{ font-size: 13pt; margin: 5mm 0 2mm; border-left: 4px solid #333; padding-left: 3mm; }}
h3 {{ font-size: 11pt; margin: 3mm 0 1.5mm; }}

p, li {{ margin: 1.5mm 0; }}

/* 圖片：頁數控制的關鍵 */
img {{
    display: block;
    max-height: {IMAGE_MAX_HEIGHT_MM}mm;
    max-width: 100%;
    width: auto;
    margin: 2mm auto;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    margin: 2mm 0;
    font-size: {BASE_FONT_PT - 1}pt;
}}
th, td {{ border: 1px solid #999; padding: 1.2mm 2mm; text-align: left; }}
th {{ background: #ececec; font-weight: 600; }}

code {{
    font-family: "SF Mono", Menlo, monospace;
    font-size: {BASE_FONT_PT - 1.5}pt;
    background: #f4f4f4;
    padding: 0.3mm 1mm;
    border-radius: 2px;
}}
pre {{
    background: #f7f7f7;
    border-left: 3px solid #bbb;
    padding: 2mm 3mm;
    overflow-x: auto;
    font-size: {BASE_FONT_PT - 2}pt;
    line-height: 1.35;
}}
pre code {{ background: none; padding: 0; }}

blockquote {{
    border-left: 3px solid #d0d0d0;
    margin: 2mm 0;
    padding: 1mm 3mm;
    color: #666;
}}

hr {{ border: none; border-top: 1px solid #ddd; margin: 4mm 0; }}

/* 避免元素被切斷在跨頁處 */
table, pre, img, blockquote {{ page-break-inside: avoid; }}
h1, h2, h3 {{ page-break-after: avoid; }}

@media print {{
    body {{ padding: 0; max-width: none; }}
}}
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{content}
</body>
</html>
"""


def main() -> int:
    try:
        import markdown
    except ImportError:
        print("缺少 markdown 套件。請先執行：pip install markdown", file=sys.stderr)
        return 1

    if not SOURCE_PATH.exists():
        print(f"找不到來源檔：{SOURCE_PATH}", file=sys.stderr)
        return 1

    source = SOURCE_PATH.read_text(encoding="utf-8")
    body = markdown.markdown(
        source,
        extensions=["tables", "fenced_code", "sane_lists"],
    )

    OUTPUT_PATH.write_text(
        HTML_TEMPLATE.format(
            title="智慧工廠設備異常告警 AI Agent — 技術報告",
            css=PRINT_CSS,
            content=body,
        ),
        encoding="utf-8",
    )

    print(f"已產生 {OUTPUT_PATH}")
    print(f"圖片高度上限 {IMAGE_MAX_HEIGHT_MM}mm（超頁時調降此值）")
    print("下一步：open report.html → Cmd+P → 另存為 PDF（A4、邊界設「無」、勾選背景圖形）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
