import warnings
warnings.filterwarnings("ignore", message="The groups parameter")
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("data/flex_clean.csv")
df["extinguished"] = (df["test_end"] == "Extinction").astype(int)

FEATS = ["fuel", "pressure_mmHg", "O2", "CO2", "He", "N2", "D0_mm"]
CAT = ["fuel"]
NUM = [c for c in FEATS if c not in CAT]

# same atmosphere (pressure level + O2 + CO2 + He) = same group
pbin = pd.cut(df["pressure_mmHg"], [0, 600, 900, 1700, 2500], labels=False)
groups = (pbin.astype(str) + "_" + df["O2"].round(2).astype(str) + "_"
          + df["CO2"].round(2).astype(str) + "_" + df["He"].round(2).astype(str))
groups = groups.fillna("missing")
print("tests:", len(df), "| distinct atmospheres (groups):", groups.nunique())

def prep(scale):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    return ColumnTransformer([("num", Pipeline(steps), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])

X, y = df[FEATS], df["extinguished"]
models = {
    "Baseline (majority)": Pipeline([("p", prep(False)), ("m", DummyClassifier(strategy="most_frequent"))]),
    "Logistic Regression": Pipeline([("p", prep(True)), ("m", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "Random Forest": Pipeline([("p", prep(False)), ("m", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42))]),
}
cvs = {"random 5-fold": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
       "grouped 5-fold": GroupKFold(n_splits=5)}

print("\n=== CLASSIFICATION: extinction vs not ===")
for cvname, cv in cvs.items():
    print(f"--- {cvname} ---")
    for name, pipe in models.items():
        s = cross_validate(pipe, X, y, cv=cv, groups=groups,
                           scoring=["accuracy", "recall", "precision", "f1", "roc_auc"])
        print(f"{name:22s} acc={s['test_accuracy'].mean():.2f} recall={s['test_recall'].mean():.2f} "
              f"prec={s['test_precision'].mean():.2f} f1={s['test_f1'].mean():.2f} auc={s['test_roc_auc'].mean():.2f}")

best = models["Random Forest"]
for cvname, cv in cvs.items():
    pred = cross_val_predict(best, X, y, cv=cv, groups=groups)
    print(f"\nRandom Forest confusion matrix, {cvname} [rows=actual 0/1, cols=pred 0/1]")
    print(confusion_matrix(y, pred))
    if cvname.startswith("grouped"):
        print(classification_report(y, pred, target_names=["No extinction", "Extinction"]))

print("\n=== REGRESSION: burn rate ===")
dr = df.dropna(subset=["burn_rate_mm2s"])
Xr, yr, gr = dr[FEATS], dr["burn_rate_mm2s"], groups.loc[dr.index]
reg = Pipeline([("p", prep(False)), ("m", RandomForestRegressor(n_estimators=300, random_state=42))])
base = Pipeline([("p", prep(False)), ("m", DummyRegressor())])
rcvs = {"random 5-fold": KFold(n_splits=5, shuffle=True, random_state=42),
        "grouped 5-fold": GroupKFold(n_splits=5)}
for cvname, cv in rcvs.items():
    for name, pipe in [("Baseline (mean)", base), ("Random Forest", reg)]:
        s = cross_validate(pipe, Xr, yr, cv=cv, groups=gr, scoring=["r2", "neg_mean_absolute_error"])
        print(f"{cvname:15s} {name:16s} R2={s['test_r2'].mean():.2f} MAE={-s['test_neg_mean_absolute_error'].mean():.3f}")

Path("models").mkdir(exist_ok=True)
best.fit(X, y)
reg.fit(Xr, yr)
joblib.dump(best, "models/clf_extinction.joblib")
joblib.dump(reg, "models/reg_burnrate.joblib")
print("\nSaved models/ folder")