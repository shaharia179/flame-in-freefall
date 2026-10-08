# Flame in Freefall: Droplet Combustion Insights

NASA Space Apps Challenge 2026: "Flame in Freefall: AI-Powered Fire Safety Insights from Microgravity Combustion Data"

Author: MD SHAHARIA HASAN 

Live demo: https://flame-in-freefall-xxjhcu8eymmkafnbgtgjuq.streamlit.app/

Code: https://github.com/shaharia179/flame-in-freefall
## What this project does
A small, honest machine-learning prototype built on real NASA FLEX
(Flame Extinguishment Experiment) data from the International Space Station.
From initial conditions only (fuel, O2, CO2, N2, pressure, droplet size) it estimates:
1. the probability that the flame ends in **extinction**
2. the **burning-rate constant** (mm²/s)

A Streamlit dashboard lets users change conditions and see predictions,
explore the data, and read the limits.

## Data
- Source: NASA/TP-2015-216046, "Flame Extinguishment Experiment (FLEX)
  detailed results", Table IX (170 tests, methanol and n-heptane droplets).
- Table IX was transcribed from the PDF text with a script (`build_csv.py`);
  no values were invented. Missing values ("–" in the report) are kept as missing.
- Test FLEX-114 lists pressure 0, clearly an error; treated as missing.
- Test FLEX-156 is absent from Table IX, so there are 170 rows, not 171.
- FLEX-070 and FLEX-073 share the same identifier in the report; both kept as printed.
- Raw data are available from NASA PSI (psi.nasa.gov, registration required).

## Method
- Features: fuel, pressure, O2, CO2, N2, initial droplet diameter.
  Post-test quantities (extinction diameter, burn time, burn rate) are NOT
  used as classification inputs, to avoid data leakage.
- Classification target: Extinction (1) vs Disruption/Completion (0).
- Models: Logistic Regression, Random Forest; majority baseline for comparison.
- 5-fold cross-validation; missing values imputed inside each fold.
- SHAP used to inspect what the models learned.

## Results (5-fold CV)
| Model | Accuracy | AUC |
|---|---|---|
| Majority baseline | 0.61 | 0.50 |
| Logistic Regression | 0.81 | 0.89 |
| Random Forest | 0.81 | 0.90 |

Random Forest confusion matrix: 87 true extinctions and 51 true non-extinctions
correct; 15 false "extinction" calls and 17 missed extinctions.
The 15 false "extinction" calls are the safety-critical error type: the model
said the flame would go out when it did not.

Burn rate regression: R² 0.63, MAE 0.057 mm²/s (baseline R² -0.05).

Random Forest and Logistic Regression perform about the same, so the
simple model is nearly as good.

Note: CV scores come from 170 tests and carry real uncertainty; treat them as approximate.

## Insights (what the models learned)
- Higher O2 → higher burning rate and lower extinction probability.
- Higher CO2 → higher extinction probability and lower burning rate.
- Methanol tests ended in extinction more often than heptane (75/104 vs 29/66).

These agree with expected combustion physics, but SHAP shows what the
model learned, not proof of causation.

## Limitations (please read)
- Only liquid fuel droplets (methanol, heptane), not solid materials or real cabin fires.
- Small dataset (170 tests); results are approximate.
- "Disruption" is not a clean "no extinction" outcome. NASA notes some
  disruptions may be linked to fuel-needle contamination (Section 5.2).
- O2, CO2 and N2 sum to about 1, so their individual effects cannot be cleanly separated.
- Predictions outside the data range (O2 0.12–0.34, CO2 0–0.70) are unreliable.
- A high extinction probability is NOT a safety guarantee.
  Not for real safety decisions.

## Run it
Download the NASA report PDF (NASA/TP-2015-216046) and save it as
`data/flex_detailed_results.pdf`, then:

```
pip install pandas numpy matplotlib scikit-learn shap streamlit joblib pdfplumber
python extract_pages.py
python build_csv.py
python eda.py
python train_models.py
python shap_analysis.py
streamlit run app.py
```

## AI tool usage declaration
Claude (Anthropic) was used as a coding and planning assistant: it helped
write the data-extraction, modelling and app code and the README draft.
The data selection, running, checking and decisions were done by the author.

## Credits
NASA Glenn Research Center, FLEX experiment data, NASA/TP-2015-216046
(NASA Physical Sciences Informatics, psi.nasa.gov).
