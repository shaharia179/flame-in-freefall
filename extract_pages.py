import pdfplumber, re
from pathlib import Path

Path("out").mkdir(exist_ok=True)
rowpat = re.compile(r"^FLEX[–-]\d{3}\s", re.M)

with pdfplumber.open("data/flex_detailed_results.pdf") as pdf, \
     open("out/flex_pages_20_30.txt", "w", encoding="utf-8") as f:
    for p in range(20, 51):
        text = pdf.pages[p - 1].extract_text() or ""
        f.write(f"\n\n===== PAGE {p} =====\n{text}")
        print(p, "rows:", len(rowpat.findall(text)))