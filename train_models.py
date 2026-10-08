import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, cross_val_predict
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

FEATS = ["fuel", "pressure_mmHg", "O2", "CO2", "N2", "D0_mm"]
CAT = ["fuel"]
NUM = [c for c in FEATS if c not in CAT]

def prep(scale):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])

# ---------- Classification ----------
X, y = df[FEATS], df["extinguished"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Baseline (majority)": Pipeline([("p", prep(False)), ("m", DummyClassifier(strategy="most_frequent"))]),
    "Logistic Regression": Pipeline([("p", prep(True)), ("m", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "Random Forest": Pipeline([("p", prep(False)), ("m", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42))]),
}

print("=== CLASSIFICATION (5-fold CV) ===")
for name, pipe in models.items():
    s = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "recall", "precision", "f1", "roc_auc"])
    print(f"{name:22s} acc={s['test_accuracy'].mean():.2f} "
          f"recall={s['test_recall'].mean():.2f} prec={s['test_precision'].mean():.2f} "
          f"f1={s['test_f1'].mean():.2f} auc={s['test_roc_auc'].mean():.2f}")

best = models["Random Forest"]
pred = cross_val_predict(best, X, y, cv=cv)
print("\nRandom Forest confusion matrix [rows=actual 0/1, cols=pred 0/1]")
print(confusion_matrix(y, pred))
print(classification_report(y, pred, target_names=["No extinction", "Extinction"]))

# ---------- Regression ----------
dr = df.dropna(subset=["burn_rate_mm2s"])
Xr, yr = dr[FEATS], dr["burn_rate_mm2s"]
cvr = KFold(n_splits=5, shuffle=True, random_state=42)
reg = Pipeline([("p", prep(False)), ("m", RandomForestRegressor(n_estimators=300, random_state=42))])
base = Pipeline([("p", prep(False)), ("m", DummyRegressor())])

print("\n=== REGRESSION: burn rate (5-fold CV) ===")
for name, pipe in [("Baseline (mean)", base), ("Random Forest", reg)]:
    s = cross_validate(pipe, Xr, yr, cv=cvr, scoring=["r2", "neg_mean_absolute_error"])
    print(f"{name:18s} R2={s['test_r2'].mean():.2f} MAE={-s['test_neg_mean_absolute_error'].mean():.3f} mm2/s")

# ---------- Final fit & save ----------
Path("models").mkdir(exist_ok=True)
best.fit(X, y)
reg.fit(Xr, yr)
joblib.dump(best, "models/clf_extinction.joblib")
joblib.dump(reg, "models/reg_burnrate.joblib")
print("\nSaved models/ folder")