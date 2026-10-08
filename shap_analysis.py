import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/flex_clean.csv")
FEATS = ["fuel", "pressure_mmHg", "O2", "CO2", "He", "N2", "D0_mm"]
X = df[FEATS]

Path("out").mkdir(exist_ok=True)

for name, path in [("clf", "models/clf_extinction.joblib"),
                   ("reg", "models/reg_burnrate.joblib")]:
    pipe = joblib.load(path)
    Xt = pipe.named_steps["p"].transform(X)
    names = pipe.named_steps["p"].get_feature_names_out()
    model = pipe.named_steps["m"]
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    plt.figure()
    shap.summary_plot(sv, Xt, feature_names=names, show=False)
    plt.savefig(f"out/shap_{name}.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("saved", name)