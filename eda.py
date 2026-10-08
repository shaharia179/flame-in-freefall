import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/flex_table_ix.csv")
Path("out").mkdir(exist_ok=True)

# pressure 0 স্পষ্টতই ভুল (FLEX-114) -> missing ধরি
df.loc[df["pressure_mmHg"] <= 0, "pressure_mmHg"] = float("nan")
df.to_csv("data/flex_clean.csv", index=False)

# Chart 1: O2 বনাম CO2, test_end অনুযায়ী রং
fig, ax = plt.subplots(figsize=(7, 5))
colors = {"Extinction": "tab:green", "Disruption": "tab:red", "Completion": "tab:blue"}
for k, g in df.groupby("test_end"):
    ax.scatter(g["CO2"], g["O2"], label=k, alpha=0.6, c=colors[k])
ax.set_xlabel("CO2 mole fraction")
ax.set_ylabel("O2 mole fraction")
ax.legend()
fig.savefig("out/eda_o2_co2.png", dpi=120, bbox_inches="tight")

# Chart 2: burn rate বনাম O2, fuel অনুযায়ী
fig, ax = plt.subplots(figsize=(7, 5))
for k, g in df.groupby("fuel"):
    ax.scatter(g["O2"], g["burn_rate_mm2s"], label=k, alpha=0.6)
ax.set_xlabel("O2 mole fraction")
ax.set_ylabel("Burning rate (mm^2/s)")
ax.legend()
fig.savefig("out/eda_burnrate_o2.png", dpi=120, bbox_inches="tight")

print(df.groupby(["fuel", "test_end"]).size())
print(df[["pressure_mmHg", "O2", "CO2", "D0_mm", "burn_rate_mm2s"]].describe().round(2))