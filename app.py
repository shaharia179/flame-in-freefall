import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="FLEX Fire Safety Insights", layout="wide")
st.title("🔥 Flame in Freefall: Droplet Combustion Insights")
st.caption("Data: NASA FLEX experiments on the ISS (NASA/TP-2015-216046, Table IX). "
           "Research prototype, not for real safety decisions.")

clf = joblib.load("models/clf_extinction.joblib")
reg = joblib.load("models/reg_burnrate.joblib")
df = pd.read_csv("data/flex_clean.csv")

tab1, tab2, tab3 = st.tabs(["Predict", "Data explorer", "About & limits"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fuel = st.selectbox("Fuel", ["Methanol", "Heptane"])
        o2 = st.slider("O2 mole fraction", 0.12, 0.34, 0.21, 0.01)
        co2 = st.slider("CO2 mole fraction", 0.0, 0.70, 0.0, 0.01)
    with c2:
        p = st.slider("Pressure (mmHg)", 526, 2320, 760, 10)
        d0 = st.slider("Initial droplet diameter (mm)", 1.1, 4.7, 3.0, 0.1)

    n2 = round(1.0 - o2 - co2, 3)
    if n2 < 0:
        st.error("O2 + CO2 cannot exceed 1.0")
    else:
        st.write(f"N2 (computed as remainder): **{n2:.2f}**")
        x = pd.DataFrame([{"fuel": fuel, "pressure_mmHg": p, "O2": o2,
                           "CO2": co2, "N2": n2, "D0_mm": d0}])
        prob = clf.predict_proba(x)[0, 1]
        rate = reg.predict(x)[0]
        m1, m2 = st.columns(2)
        m1.metric("Estimated probability of extinction", f"{prob:.0%}")
        m2.metric("Estimated burning rate", f"{rate:.2f} mm²/s")
        st.progress(float(prob))

        lo, hi = df[["O2", "CO2"]].min(), df[["O2", "CO2"]].max()
        if not (lo["O2"] <= o2 <= hi["O2"] and lo["CO2"] <= co2 <= hi["CO2"]):
            st.warning("Outside the range of the FLEX data. Prediction unreliable.")
        st.info("Cross-validated model (small dataset, ~170 tests). "
                "A high extinction probability is NOT a safety guarantee.")

with tab2:
    fuel_f = st.multiselect("Fuel filter", df["fuel"].unique(), default=list(df["fuel"].unique()))
    d = df[df["fuel"].isin(fuel_f)]
    st.scatter_chart(d, x="O2", y="burn_rate_mm2s", color="fuel")
    st.write("Test outcomes by fuel")
    st.dataframe(pd.crosstab(d["fuel"], d["test_end"]))
    st.dataframe(d.drop(columns=["identifier"]), width="stretch")

with tab3:
    st.markdown("""
**What this is:** two small models trained on 170 real FLEX droplet tests
(methanol and n-heptane) predicting (1) whether the flame ends in extinction
and (2) the burning-rate constant, from initial conditions only.

**Limits**
- Only droplet combustion, not solid-material fires.
- CV results: extinction AUC ≈ 0.90, burn-rate R² ≈ 0.63.
- Some disruptions may be linked to needle contamination (NASA report, Sec. 5.2).
- O2, CO2 and N2 are strongly correlated.
- Prototype for a hackathon, not a safety tool.
""")