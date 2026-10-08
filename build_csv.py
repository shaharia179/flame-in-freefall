import re
import pandas as pd
from pathlib import Path

text = Path("out/flex_pages_20_30.txt").read_text(encoding="utf-8")

# Table IX: FLEX-001 থেকে শুরু হওয়া লাইন
pattern = re.compile(
    r"^FLEX[–-](\d{3})\s+(\S+)\s+(\d+/\d+/\d+)\s+(\S+)\s+(Methanol|Heptane)\s+(.*)$"
)
# কিছু row-এ time নেই ('–'), তাই alternative pattern
pattern2 = re.compile(
    r"^FLEX[–-](\d{3})\s+(\S+)\s+(\d+/\d+/\d+)\s+(–)\s+(Methanol|Heptane)\s+(.*)$"
)

def num(x):
    x = x.strip()
    return None if x in ("–", "-", "") else float(x)

rows = []
for line in text.splitlines():
    line = line.strip()
    m = pattern.match(line) or pattern2.match(line)
    if not m:
        continue
    test, ident, date, tm, fuel, rest = m.groups()
    toks = rest.split()
    # rest = P O2 N2 CO2 He D0 Dext rate burntime end
    if len(toks) != 10:
        print("SKIP (unexpected columns):", line)
        continue
    P, O2, N2, CO2, He, D0, Dext, rate, bt, end = toks
    rows.append(dict(
        test=int(test), identifier=ident, date=date, fuel=fuel,
        pressure_mmHg=num(P), O2=num(O2), N2=num(N2), CO2=num(CO2), He=num(He),
        D0_mm=num(D0), ext_diameter_mm=num(Dext), burn_rate_mm2s=num(rate),
        burn_time_s=num(bt), test_end=end,
    ))

df = pd.DataFrame(rows)
df.to_csv("data/flex_table_ix.csv", index=False)
print(df.shape)
print(df["test_end"].value_counts())
print(df["fuel"].value_counts())
print(df.isna().sum())
print(df.head())