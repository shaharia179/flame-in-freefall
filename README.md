# Flame in Freefall

AI-powered fire-safety insights from real NASA microgravity combustion data.
Submission for the NASA Space Apps Challenge: *"Flame in Freefall: AI-Powered Fire Safety Insights from Microgravity Combustion Data"*.

## Team

- MD Shaharia Hasan
- MD Abdullah Al sami
- Md. Abdur Razzaque Siam
- Md. Sanowas Hossain


- Live app: https://flame-in-freefall-xxjhcu8eymmkafnbgtgjuq.streamlit.app/
- Code: https://github.com/shaharia179/flame-in-freefall

> Independent hackathon project. Not affiliated with or endorsed by NASA.

## What this project does

On the International Space Station, the Flame Extinguishment Experiment (FLEX) burned small fuel droplets in different gas mixtures and recorded whether each flame went out. This project:

1. Turns the full FLEX results table into a clean dataset.
2. Trains two small models on the **starting conditions** of each test:
   - **Classification:** did the flame go out (extinction) or not?
   - **Regression:** what was the burning-rate constant (mm²/s)?
3. Explains what the models learned (SHAP) and shows it all in an interactive Streamlit app (survival map, real data vs model, dashboards, a downloadable report).

## Data

All data comes from NASA/TP-2015-216046, "FLEX detailed results", **Table IX** (units in mmHg), PDF pages 26 to 34. No data was invented or simulated.

- **276 tests** (FLEX-001 to FLEX-280).
- Test numbers absent from the report: **156, 205, 213, 228**.
- Outcomes: Extinction 186, Disruption 62, Completion 28.
- Fuels: Methanol 157 tests (126 extinction, 80%), Heptane 119 tests (60 extinction, 50%).
- Ranges: pressure 526 to 2320 mmHg, O₂ 0.12 to 0.34, CO₂ 0 to 0.70, He 0 to 0.45, initial droplet diameter 1.1 to 4.67 mm, burning rate 0.05 to 1.06 mm²/s.
- Helium is above 0 in 51 tests (40 extinction, 5 disruption, 6 completion). CO₂ and He are never both above 0.

### Data cleaning notes

- FLEX-227 and FLEX-229 have "~6" and "~9" as burn time in the report (approximate values). The tilde was removed and the numbers kept.
- FLEX-114 has a pressure of 0 in the report, which is clearly wrong. It is treated as missing.
- FLEX-070 and FLEX-073 share one identifier in the report. Kept as printed.
- Missing values: initial diameter 12, extinction diameter 78, burn rate 16, burn time 6, pressure 1.
- Tables VI to VIII (atm units) are not used, only Table IX.
- Disruption is not a clean "no extinction". Some disruptions may come from fuel-needle contamination (report Sec. 5.2).

## Method

- **Inputs (starting conditions only):** fuel, pressure, O₂, CO₂, He, N₂, initial droplet diameter.
- Burn time, burn rate and extinction diameter are measured *after* a test, so they are **never** used as classification inputs (this avoids data leakage).
- **Classification target:** Extinction (1) vs Disruption or Completion (0).
- **Regression target:** burning-rate constant.
- **Models:** Logistic Regression and Random Forest, compared with a majority-class baseline. The app uses the Random Forest.
- **Validation:**
  - Random 5-fold cross-validation.
  - **Grouped 5-fold cross-validation** (`GroupKFold`). Tests with the same atmosphere are kept together in the same fold. An "atmosphere" is defined by the pressure level (four bins: up to 600, 600-900, 900-1700 and 1700-2500 mmHg) plus O₂, CO₂ and He rounded to 2 decimals. Fuel and droplet size are not part of the group key. This gives 47 groups (the test with missing pressure forms its own group). This is the main, more honest number, because the model has to predict atmospheres it has not seen.

## Results

### Extinction classification (276 tests)

| Model | Validation | Accuracy | Recall | Precision | F1 | AUC |
|---|---|---|---|---|---|---|
| Baseline (majority) | random | 0.67 | 1.00 | 0.67 | 0.81 | 0.50 |
| Logistic Regression | random | 0.80 | 0.79 | 0.90 | 0.84 | 0.89 |
| Random Forest | random | 0.82 | 0.87 | 0.86 | 0.87 | 0.89 |
| Baseline (majority) | **grouped** | 0.67 | 1.00 | 0.67 | 0.80 | 0.50 |
| Logistic Regression | **grouped** | 0.79 | 0.76 | 0.89 | 0.81 | 0.88 |
| Random Forest | **grouped** | 0.78 | 0.85 | 0.84 | 0.84 | 0.85 |

Honest reading: with grouped validation the simple Logistic Regression is as good as the Random Forest (slightly better on AUC). On a dataset this small, a simple model is competitive.

### The safety-critical error

Random Forest confusion matrix, grouped 5-fold (rows = actual, columns = predicted):

|  | predicted: no extinction | predicted: extinction |
|---|---|---|
| actual: no extinction (90) | 57 | **33** |
| actual: extinction (186) | 28 | 158 |

The dangerous mistake is a **false positive**: the model says the flame "goes out" but in reality it does not. This happened for **33 of 90** non-extinction tests (about 37%). The model must not be used on its own to decide anything about fire safety.

### Burning-rate regression

| Validation | Model | R² | MAE (mm²/s) |
|---|---|---|---|
| random 5-fold | Baseline (mean) | -0.01 | 0.107 |
| random 5-fold | Random Forest | 0.68 | 0.052 |
| grouped 5-fold | Baseline (mean) | -0.05 | 0.108 |
| grouped 5-fold | Random Forest | 0.66 | 0.051 |

An earlier version trained on only 170 tests (AUC 0.90, R² 0.63) is outdated and no longer used.

## What the models learned (SHAP)

SHAP shows what the model learned, **not** what causes what.

- Higher O₂ goes with a lower chance of extinction and a higher burning rate.
- Higher CO₂ goes with a higher chance of extinction and a lower burning rate.
- Methanol tests went out more often than heptane tests (80% vs 50%).
- Helium raises the predicted burning rate. Its effect on extinction is less clear: the summary plot leans toward extinction, but the prediction for a single input can differ. Only 51 tests contain He, so this is weak evidence.
- O₂, CO₂, He and N₂ sum to 1, so their effects cannot be cleanly separated.

## Limits

- Liquid droplets only (methanol and n-heptane). Not solid materials, not real cabin or spacecraft fires.
- Small dataset (276 tests), few fixed pressure levels, and a few atmospheres repeated many times.
- Possible fuel-needle contamination behind some disruptions.
- Strongly correlated gas fractions, so SHAP effects are entangled.
- False positives (see above) make the model unsuitable as a safety tool.
- Estimates far from any real test are extrapolation. The app warns about this and hides map cells that are far from real data.

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```

To rebuild everything from the report PDF:

```bash
python extract_pages.py     # PDF pages -> text
python build_csv.py         # text -> data/flex_table_ix.csv
python eda.py               # cleaning -> data/flex_clean.csv
python train_models.py      # cross-validation + models/*.joblib
python shap_analysis.py     # out/shap_clf.png, out/shap_reg.png
```

The NASA report PDF (NASA/TP-2015-216046) is not included in this repository. To rebuild from scratch, download it and save it as `data/flex_detailed_results.pdf`. `extract_pages.py` reads PDF pages 20 to 50 (Table IX is on pages 26 to 34) and writes `out/flex_pages_20_30.txt`; the file name is a leftover from an earlier version.

## Repository files

| File | Purpose |
|---|---|
| `extract_pages.py` | Extracts Table IX pages from the NASA PDF |
| `build_csv.py` | Parses the text into `data/flex_table_ix.csv` |
| `eda.py` | Cleaning and checks, writes `data/flex_clean.csv` |
| `train_models.py` | Baselines, models, random and grouped cross-validation |
| `shap_analysis.py` | SHAP plots |
| `app.py` | Streamlit app |
| `models/` | Trained models |
| `out/` | SHAP figures |


## Credit

Data: NASA FLEX results, NASA/TP-2015-216046, Table IX. Built for the NASA Space Apps Challenge. This project is independent and is not affiliated with or endorsed by NASA.
