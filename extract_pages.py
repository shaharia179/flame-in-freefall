import pdfplumber
from pathlib import Path

jobs = [
    ("data/flex_detailed_results.pdf", range(20, 31), "flex_pages_20_30.txt"),
    ("data/bass2_summary_report.pdf", range(20, 27), "bass2_pages_20_26.txt"),
]

Path("out").mkdir(exist_ok=True)

for pdf_file, pages, out_name in jobs:
    with pdfplumber.open(pdf_file) as pdf:
        with open(Path("out") / out_name, "w", encoding="utf-8") as f:
            for p in pages:
                if p <= len(pdf.pages):
                    f.write(f"\n\n===== PAGE {p} =====\n")
                    f.write(pdf.pages[p - 1].extract_text() or "")
    print("done:", out_name)