import re
import pandas as pd
from pathlib import Path

text = Path("out/flex_pages_20_30.txt").read_text(encoding="utf-8")

pattern = re.compile(
    r"^FLEX[–-](\d{3})\s+(\S+)\s+(\d+/\d+/\d+)\s+(\S+)\s+(Methanol|Heptane)\s+(.*)$"
)

def num(x):
    x = x.strip()
    if x in ("–", "-", ""):
        return None
    return float(x.lstrip("~"))

rows = []
for line in text.splitlines():
    line = line.strip()
    m = pattern.match(line)
    if not m:
        continue
    if "~" in line:
        print("APPROX (tilde removed):", line)
    test, ident, date, tm, fuel, rest = m.groups()
    toks = rest.split()
    if len(toks) != 10:
        print("SKIP (unexpected columns):", line)
        continue
    P, O2, N2, CO2, He, D0, Dext, rate, bt, end = toks
    try:
        rows.append(dict(
            test=int(test), identifier=ident, date=date, fuel=fuel,
            pressure_mmHg=num(P), O2=num(O2), N2=num(N2), CO2=num(CO2), He=num(He),
            D0_mm=num(D0), ext_diameter_mm=num(Dext), burn_rate_mm2s=num(rate),
            burn_time_s=num(bt), test_end=end,
        ))
    except ValueError as e:
        print("SKIP (bad number):", line, "|", e)

df = pd.DataFrame(rows)
df.to_csv("data/flex_table_ix.csv", index=False)
print(df.shape, "| test min/max:", df.test.min(), df.test.max(), "| unique:", df.test.is_unique)
print("missing test numbers:", sorted(set(range(1, int(df.test.max()) + 1)) - set(df.test)))
print(df["test_end"].value_counts())
print(df["fuel"].value_counts())
print(df.isna().sum())