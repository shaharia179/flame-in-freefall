import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Flame in Freefall", page_icon="🔥", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;800&display=swap');
html, body, [class*="css"], .stMarkdown {font-family:'Sora',sans-serif;}
.stApp {background:radial-gradient(1100px 520px at 8% -8%,#1a2d7a66,transparent),
        radial-gradient(800px 480px at 100% 0%,#ff7a1a2e,transparent),#05070f;}
header, footer, #MainMenu {visibility:hidden;}
.block-container {padding-top:1.4rem; max-width:1400px;}
.hero h1 {font-weight:800; font-size:2.9rem; line-height:1.05; margin:0; letter-spacing:-.02em;
          background:linear-gradient(95deg,#fff 10%,#8ab8ff 55%,#ffa24a);
          -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.hero p {color:#93a4c8; font-weight:300; margin:.5rem 0 1.1rem; font-size:1.02rem;}
.kpi {border:1px solid #ffffff1c; border-radius:14px; padding:.7rem 1rem;
      background:linear-gradient(150deg,#ffffff0f,#ffffff03);}
.kpi b {display:block; font-size:1.65rem; font-weight:800; color:#fff;}
.kpi span {color:#8195bd; font-size:.82rem;}
.panel {border:1px solid #ffffff1c; border-radius:18px; padding:1rem 1.2rem; background:#0b1226cc;}
.verdict {font-size:1.05rem; font-weight:600; padding:.6rem .9rem; border-radius:12px;
          border:1px solid #ffffff22; background:#ffffff0a;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load():
    return (joblib.load("models/clf_extinction.joblib"),
            joblib.load("models/reg_burnrate.joblib"),
            pd.read_csv("data/flex_clean.csv"))


clf, reg, df = load()

# ---------------- Hero + KPIs (all computed from the real table) ----------------
st.markdown('<div class="hero"><h1>Flame in Freefall</h1>'
            '<p>What decides whether a flame survives in microgravity? '
            'Explore real NASA ISS droplet-combustion tests and what a small model learned from them.</p></div>',
            unsafe_allow_html=True)
ext_rate = (df.test_end == "Extinction").mean()
kpis = [(len(df), "real ISS tests (FLEX)"), (f"{ext_rate:.0%}", "ended in extinction"),
        (f"{df.O2.min():.2f} to {df.O2.max():.2f}", "O₂ mole fraction tested"),
        (f"{df.CO2.max():.2f}", "highest CO₂ fraction tested"),
        ("0.85", "extinction AUC (grouped CV)")]
for col, (v, l) in zip(st.columns(5), kpis):
    col.markdown(f'<div class="kpi"><b>{v}</b><span>{l}</span></div>', unsafe_allow_html=True)
st.write("")

# ---------------- Controls | Flame | Results ----------------
left, mid, right = st.columns([1, 1.55, 1.1], gap="large")
with left:
    st.markdown("##### Conditions")
    fuel = st.radio("Fuel", ["Methanol", "Heptane"], horizontal=True)
    o2 = st.slider("O₂ mole fraction", 0.12, 0.34, 0.21, 0.01)
    gas = st.radio("Added gas (CO₂ and He never appear together in the real tests)",
                   ["CO₂", "He"], horizontal=True)
    if gas == "CO₂":
        co2 = st.slider("CO₂ mole fraction", 0.0, 0.70, 0.0, 0.01, key="co2_slider")
        he = 0.0
    else:
        he = st.slider("He mole fraction", 0.0, 0.45, 0.0, 0.01, key="he_slider")
        co2 = 0.0
    p = st.slider("Pressure (mmHg)", 526, 2320, 760, 10)
    d0 = st.slider("Droplet diameter (mm)", 1.1, 4.7, 3.0, 0.1)

n2 = round(1.0 - o2 - co2 - he, 3)
if n2 < 0:
    st.error("O₂ + added gas cannot exceed 1.0. Lower one of them.")
    st.stop()
x = pd.DataFrame([dict(fuel=fuel, pressure_mmHg=p, O2=o2, CO2=co2, He=he, N2=n2, D0_mm=d0)])
pe = float(clf.predict_proba(x)[0, 1])
br = float(reg.predict(x)[0])
near = float(np.sqrt(((df.O2 - o2) / .22) ** 2 + ((df.CO2 - co2) / .7) ** 2
                     + ((df.He - he) / .45) ** 2).min())

FLAME = """<body style="margin:0;background:transparent"><canvas id="c" style="width:100%;height:440px;border-radius:20px;background:#03050b"></canvas>
<script>
const C=__CFG__,cv=document.getElementById('c'),g=cv.getContext('2d');
cv.width=cv.clientWidth*2;cv.height=880;const W=cv.width,H=cv.height,cx=W/2,cy=H/2;
const S=Array.from({length:120},()=>[Math.random()*W,Math.random()*H,Math.random()*2+.6,Math.random()*9]);
const P=Array.from({length:80},()=>[Math.random()*6.28,.82+Math.random()*.4,.15+Math.random()*.6]);
let t=0;(function f(){t+=.016;g.clearRect(0,0,W,H);
S.forEach(s=>{g.globalAlpha=.2+.6*Math.abs(Math.sin(t+s[3]));g.fillStyle='#fff';g.fillRect(s[0],s[1],s[2],s[2])});
const life=1-.7*C.pe,R=(70+Math.max(.05,C.br)*300)*life*(1+.03*Math.sin(t*6)+.02*Math.sin(t*11)),r=6+C.d0*7;
for(let i=0;i<3;i++){const rr=R*(1+i*.22),a=.55*life/(i+1),gr=g.createRadialGradient(cx,cy,rr*.5,cx,cy,rr);
gr.addColorStop(0,'rgba(40,110,255,0)');gr.addColorStop(.8,`rgba(${80+i*60},${150+i*10},255,${a})`);gr.addColorStop(1,'rgba(255,150,50,0)');
g.globalAlpha=1;g.fillStyle=gr;g.beginPath();g.arc(cx,cy,rr,0,6.283);g.fill()}
P.forEach(q=>{const a=q[0]+t*q[2],rr=R*q[1];g.globalAlpha=.75*life;g.fillStyle='#ffb347';
g.beginPath();g.arc(cx+Math.cos(a)*rr,cy+Math.sin(a)*rr,3,0,6.283);g.fill()});
const d=g.createRadialGradient(cx,cy,0,cx,cy,r*1.7);d.addColorStop(0,'#fff');d.addColorStop(.5,'#9cc8ff');d.addColorStop(1,'rgba(80,140,255,0)');
g.globalAlpha=1;g.fillStyle=d;g.beginPath();g.arc(cx,cy,r*1.7,0,6.283);g.fill();
g.fillStyle='#8195bd';g.font='26px sans-serif';g.fillText('droplet '+C.d0.toFixed(1)+' mm  |  burn rate '+C.br.toFixed(2)+' mm²/s',28,H-28);
requestAnimationFrame(f)})();
</script></body>"""

with mid:
    components.html(FLAME.replace("__CFG__", json.dumps(dict(pe=pe, br=br, d0=d0))), height=450)
    st.caption("Illustration driven by the model outputs: flame size follows the predicted burn rate and fades as "
               "extinction becomes likely. Without buoyancy the flame is a sphere. Not a physics simulation.")


def style(fig, h):
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(255,255,255,.03)", margin=dict(l=10, r=10, t=10, b=10),
                      font=dict(family="Sora"))
    return fig


with right:
    st.markdown("##### Model estimate")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=pe * 100, number=dict(suffix="%", font=dict(size=44)),
        title=dict(text="chance the flame goes out", font=dict(size=13)),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#ffa24a"), bgcolor="#ffffff12", borderwidth=0)))
    st.plotly_chart(style(gauge, 230))
    st.metric("Burning-rate constant", f"{br:.2f} mm²/s")
    st.markdown(f'<div class="verdict">N₂ is {n2:.2f} (the remainder). '
                f'{"Extinction more likely than not." if pe >= .5 else "The model leans toward the flame surviving."}</div>',
                unsafe_allow_html=True)
    if near > .12:
        st.warning("No real test is close to this gas mix. Treat the estimate as extrapolation.")
    st.caption("Small dataset. Not a safety guarantee. In grouped cross-validation the model wrongly predicted "
               "extinction for 33 of 90 tests that did not go out.")

# ---------------- Tabs ----------------
t1, t2, t3, t4, t5, t6 = st.tabs(["Survival map", "Real data vs model", "What the model learned",
                                  "Dataset dashboard", "Analytics", "Safety insights"])

with t1:
    amax = 0.70 if gas == "CO₂" else 0.45
    ycol = "CO2" if gas == "CO₂" else "He"
    sub = df[df.He == 0] if gas == "CO₂" else df[df.CO2 == 0]
    yin = co2 if gas == "CO₂" else he


    @st.cache_data
    def grid(fuel, p, d0, gas):
        o = np.round(np.arange(0.12, 0.3401, 0.01), 2)
        c = np.round(np.arange(0.0, amax + 1e-4, 0.02), 2)
        O, C = np.meshgrid(o, c)
        g = pd.DataFrame(dict(fuel=fuel, pressure_mmHg=p, O2=O.ravel(),
                              CO2=C.ravel() if gas == "CO₂" else 0.0,
                              He=C.ravel() if gas == "He" else 0.0, D0_mm=d0))
        g["N2"] = 1 - g.O2 - g.CO2 - g.He
        z = clf.predict_proba(g)[:, 1].reshape(O.shape)
        dist = np.sqrt(((O[..., None] - sub.O2.values) / .22) ** 2
                       + ((C[..., None] - sub[ycol].values) / amax) ** 2).min(-1)
        z[(g.N2.values.reshape(O.shape) < 0) | (dist > .12)] = np.nan
        return o, c, z


    o, c, z = grid(fuel, p, d0, gas)
    fig = go.Figure(go.Heatmap(x=o, y=c, z=z, zmin=0, zmax=1, hoverongaps=False,
                               colorscale=[[0, "#ff4d2e"], [.5, "#ffd166"], [1, "#2ec4ff"]],
                               colorbar=dict(title="P(extinction)")))
    for k, sym in [("Extinction", "circle"), ("Disruption", "square"), ("Completion", "diamond")]:
        d = sub[(sub.fuel == fuel) & (sub.test_end == k)]
        fig.add_trace(go.Scatter(x=d.O2, y=d[ycol], mode="markers", name=f"real: {k}",
                                 marker=dict(symbol=sym, size=9, color="white", line=dict(width=1.2, color="black"))))
    fig.add_trace(go.Scatter(x=[o2], y=[yin], mode="markers", name="your input",
                             marker=dict(symbol="star", size=20, color="#fff", line=dict(width=2, color="#ff7a1a"))))
    fig.update_layout(xaxis_title="O₂ mole fraction", yaxis_title=f"{gas} mole fraction")
    st.plotly_chart(style(fig, 470))
    st.caption(f"Model estimate for {fuel}, {p} mmHg, {d0} mm droplet, with {gas} as the added gas (the other is 0). "
               "Red: flame tends to survive. Blue: tends to go out. "
               "Cells far from every real test are hidden. White markers are the real FLEX tests for this fuel "
               "and this added gas (plus tests with neither).")

with t2:
    fig = go.Figure()
    for f_, col in [("Methanol", "#4da3ff"), ("Heptane", "#ffa24a")]:
        d = df[df.fuel == f_]
        fig.add_trace(go.Scatter(x=d.O2, y=d.burn_rate_mm2s, mode="markers", name=f"{f_} (real)",
                                 marker=dict(color=col, size=8, opacity=.7)))
    sw = pd.DataFrame(dict(fuel=fuel, pressure_mmHg=p, O2=np.linspace(.12, .34, 40), CO2=co2, He=he, D0_mm=d0))
    sw["N2"] = 1 - sw.O2 - sw.CO2 - sw.He
    sw = sw[sw.N2 >= 0]
    fig.add_trace(go.Scatter(x=sw.O2, y=reg.predict(sw), mode="lines", name="model (your settings)",
                             line=dict(color="#fff", width=3)))
    fig.update_layout(xaxis_title="O₂ mole fraction", yaxis_title="Burning rate (mm²/s)")
    st.plotly_chart(style(fig, 430))
    out = pd.crosstab(df.fuel, df.test_end)
    st.dataframe(out)
    with st.expander(f"All {len(df)} tests"):
        st.dataframe(df.drop(columns=["identifier"]))

with t3:
    a, b = st.columns(2)
    for col, f_, title in [(a, "out/shap_clf.png", "Extinction model (SHAP)"), (b, "out/shap_reg.png", "Burn-rate model (SHAP)")]:
        col.markdown(f"**{title}**")
        if Path(f_).exists():
            col.image(f_)
    st.markdown("""
<div class="panel">

**Read this first**
- More O₂ raises the burning rate and lowers the chance of extinction. More CO₂ does the opposite.
- He raises the predicted burning rate. Its effect on extinction is unclear (the model's answer for a single input can differ from the summary plot), and only 51 tests contain He, so treat it cautiously.
- SHAP shows what the model learned, not proof of cause. O₂, CO₂, He and N₂ are tied together (they sum to 1).
- Only liquid droplets (methanol, n-heptane). Not solid materials, not cabin fires.
- Some disruptions may come from fuel-needle contamination (NASA/TP-2015-216046, Sec. 5.2).
- Extinction model, AUC: 0.89 with random 5-fold CV, 0.85 with grouped CV (same atmosphere kept together). Burn-rate R²: 0.68 random, 0.66 grouped. 276 tests, so numbers are approximate.
- The safety-critical error is the false positive: in grouped CV the model said "extinguishes" for 33 of 90 tests that did not go out.

</div>
""", unsafe_allow_html=True)

with t4:
    for col, f_ in zip(st.columns(2), ["Methanol", "Heptane"]):
        d = df[df.fuel == f_]
        col.markdown(f"**{f_}**: {len(d)} real tests")
        m1, m2, m3, m4 = col.columns(4)
        m1.metric("Avg burn rate", f"{d.burn_rate_mm2s.mean():.2f} mm²/s")
        m2.metric("Avg burn time", f"{d.burn_time_s.mean():.1f} s")
        m3.metric("Avg start size", f"{d.D0_mm.mean():.2f} mm")
        m4.metric("Went out", f"{(d.test_end == 'Extinction').mean():.0%}")

    r1, r2 = st.columns(2)
    oc = df.test_end.value_counts()
    cmap = {"Extinction": "#2ec4ff", "Disruption": "#ff4d6d", "Completion": "#ffd166"}
    fig = go.Figure(go.Pie(labels=oc.index, values=oc.values, hole=.62,
                           marker=dict(colors=[cmap[k] for k in oc.index])))
    r1.markdown("**Outcome of all tests**")
    r1.plotly_chart(style(fig, 330))

    fig = go.Figure()
    for f_, col_ in [("Methanol", "#4da3ff"), ("Heptane", "#ffa24a")]:
        d = df[df.fuel == f_]
        fig.add_trace(go.Scatter(x=d.pressure_mmHg, y=d.burn_rate_mm2s, mode="markers", name=f_,
                                 marker=dict(color=col_, size=8, opacity=.7)))
    fig.update_layout(xaxis_title="Pressure (mmHg)", yaxis_title="Burning rate (mm²/s)")
    r2.markdown("**Pressure vs burning rate (real tests)**")
    r2.plotly_chart(style(fig, 330))

    r3, r4 = st.columns(2)
    fig = go.Figure()
    for k, col_ in cmap.items():
        d = df[df.test_end == k]
        fig.add_trace(go.Scatter3d(x=d.O2, y=d.CO2, z=d.burn_rate_mm2s, mode="markers", name=k,
                                   marker=dict(size=4, color=col_)))
    fig.update_layout(scene=dict(xaxis_title="O₂", yaxis_title="CO₂", zaxis_title="Burn rate"))
    r3.markdown("**Gas mix vs burning rate (3D, real tests)**")
    r3.plotly_chart(style(fig, 380))

    fig = go.Figure()
    for f_, col_ in [("Methanol", "#4da3ff"), ("Heptane", "#ffa24a")]:
        d = df[df.fuel == f_]
        fig.add_trace(go.Box(y=d.burn_time_s, name=f_, marker_color=col_))
    fig.update_layout(yaxis_title="Burn time (s), measured after the test")
    r4.markdown("**Burn time by fuel (real tests)**")
    r4.plotly_chart(style(fig, 380))

    imp = pd.Series(clf.named_steps["m"].feature_importances_,
                    index=clf.named_steps["p"].get_feature_names_out()).sort_values()
    fig = go.Figure(go.Bar(x=imp.values, y=[i.split("__")[1] for i in imp.index],
                           orientation="h", marker_color="#ffa24a"))
    fig.update_layout(xaxis_title="Random Forest importance (extinction model)")
    st.markdown("**What the extinction model relies on**")
    st.plotly_chart(style(fig, 300))

    st.markdown("**All tests from Table IX**")
    st.dataframe(df.drop(columns=["identifier"]), height=300)
    st.caption("Burn time, burn rate and extinction diameter are measured after each test. "
               "They are shown as data here and are never used as model inputs.")

with t5:
    a1, a2, a3, a4, a5 = st.tabs(["Trends", "Correlation", "Comparison", "Distribution", "3D map"])
    FC = [("Methanol", "#4da3ff"), ("Heptane", "#ffa24a")]
    with a1:
        pb = pd.cut(df.pressure_mmHg, [500, 600, 900, 1700, 2400], labels=["~535", "~765", "~1550", "~2100-2320"])
        fig = go.Figure()
        for f_, col_ in FC:
            m = df[df.fuel == f_].groupby(pb[df.fuel == f_], observed=True).burn_rate_mm2s.mean()
            fig.add_trace(go.Scatter(x=m.index.astype(str), y=m.values, mode="lines+markers", name=f_,
                                     line=dict(color=col_)))
        fig.update_layout(xaxis_title="Pressure group (mmHg)", yaxis_title="Mean burning rate (mm²/s)")
        st.plotly_chart(style(fig, 380))
        st.caption("Average of the real tests in each pressure group. Pressure was tested at a few fixed levels, "
                   "so there are gaps in between.")
    with a2:
        cc = ["pressure_mmHg", "O2", "N2", "CO2", "He", "D0_mm", "burn_rate_mm2s", "burn_time_s"]
        cm = df[cc].corr()
        fig = go.Figure(go.Heatmap(z=cm.values, x=cc, y=cc, zmin=-1, zmax=1, colorscale="RdBu",
                                   text=cm.round(2).values, texttemplate="%{text}"))
        st.plotly_chart(style(fig, 460))
        st.caption("Correlation is not cause. O₂, N₂, CO₂ and He move together because they sum to 1.")
    with a3:
        ob = pd.cut(df.O2, [0, .15, .21, .5], labels=["O₂ 0.15 or less", "above 0.15 to 0.21", "above 0.21"])
        fig = go.Figure()
        for f_, col_ in FC:
            s = (df.test_end == "Extinction")[df.fuel == f_].groupby(ob[df.fuel == f_], observed=True).mean() * 100
            fig.add_trace(go.Bar(x=s.index.astype(str), y=s.values, name=f_, marker_color=col_))
        fig.update_layout(barmode="group", yaxis_title="Tests that went out (%)")
        st.plotly_chart(style(fig, 380))
        st.caption("Share of real tests ending in extinction, by fuel and O₂ band.")
    with a4:
        fig = go.Figure()
        for f_, col_ in FC:
            fig.add_trace(go.Histogram(x=df[df.fuel == f_].burn_rate_mm2s, name=f_, marker_color=col_,
                                       opacity=.65, nbinsx=18))
        fig.update_layout(barmode="overlay", xaxis_title="Burning rate (mm²/s)", yaxis_title="Tests")
        st.plotly_chart(style(fig, 380))
    with a5:
        @st.cache_data
        def surf(fuel, co2, he, d0):
            o = np.linspace(.12, .34, 23)
            pr = np.linspace(526, 2320, 40)
            O, Pm = np.meshgrid(o, pr)
            g = pd.DataFrame(dict(fuel=fuel, pressure_mmHg=Pm.ravel(), O2=O.ravel(), CO2=co2, He=he, D0_mm=d0))
            g["N2"] = 1 - g.O2 - co2 - he
            z = reg.predict(g).reshape(O.shape)
            dd = df.dropna(subset=["pressure_mmHg"])
            dist = np.sqrt(((O[..., None] - dd.O2.values) / .22) ** 2
                           + ((Pm[..., None] - dd.pressure_mmHg.values) / 1800) ** 2
                           + ((co2 - dd.CO2.values) / .7) ** 2
                           + ((he - dd.He.values) / .45) ** 2).min(-1)
            z[(g.N2.values.reshape(O.shape) < 0) | (dist > .15)] = np.nan
            return o, pr, z


        o_, pr_, z_ = surf(fuel, co2, he, d0)
        fig = go.Figure(go.Surface(x=o_, y=pr_, z=z_, colorscale="Inferno"))
        fig.update_layout(scene=dict(xaxis_title="O₂", yaxis_title="Pressure (mmHg)", zaxis_title="Burn rate"))
        st.plotly_chart(style(fig, 480))
        st.caption(f"Model-estimated burning rate for {fuel}, CO₂ {co2:.2f}, He {he:.2f}, {d0} mm droplet. "
                   "Areas far from any real test are left blank.")

with t6:
    sh = lambda m: (m.test_end == "Extinction").mean()
    hi = df[df.CO2 >= .3]
    lo = df[(df.CO2 == 0) & (df.He == 0)]
    hh = df[df.He > 0]
    me, he_ = df[df.fuel == "Methanol"], df[df.fuel == "Heptane"]
    st.markdown("##### Key findings, computed live from the real tests")
    st.markdown(f"""
- With CO₂ at 0.30 or more, **{sh(hi):.0%}** of {len(hi)} tests went out, versus **{sh(lo):.0%}** of {len(lo)} tests with no CO₂ and no He.
- In the {len(hh)} tests that contained He, **{sh(hh):.0%}** went out.
- O₂ and burning rate move together (correlation **{df.O2.corr(df.burn_rate_mm2s):.2f}**).
- Methanol went out in **{sh(me):.0%}** of tests, heptane in **{sh(he_):.0%}**.
""")
    st.markdown("##### What this suggests for safety (droplet tests only)")
    st.markdown("""
- CO₂ or He dilution went with more extinctions here. It is a lead worth studying, not a ready-made suppression rule.
- Oxygen-richer air raised the burning rate, so enriched atmospheres are the riskier direction.
- Heptane survived more often than methanol, so the fuel matters.
- The model can wrongly say a flame will go out. That is the dangerous mistake, so never rely on it alone.
- None of this covers solid materials or real spacecraft fires.
""")
    rep = f"""# Flame in Freefall report
Fuel: {fuel}; O2 {o2:.2f}; CO2 {co2:.2f}; He {he:.2f}; N2 {n2:.2f}; pressure {p} mmHg; droplet {d0} mm
Estimated chance the flame goes out: {pe:.0%}
Estimated burning-rate constant: {br:.2f} mm2/s
Closeness to a real test (0 = identical): {near:.2f}{' (extrapolation)' if near > .12 else ''}
Data: NASA FLEX, NASA/TP-2015-216046, Table IX ({len(df)} tests). Extinction AUC 0.89 (random 5-fold) and 0.85 (grouped 5-fold); burn-rate R2 0.68 (random) and 0.66 (grouped).
Limits: liquid droplets only, small dataset, the model wrongly predicted extinction for 33 of 90 non-extinction tests in grouped CV, not a safety guarantee.
"""
    st.download_button("Download report for the current settings", rep, file_name="flame_report.md")
    with st.expander("Data fields and missing values"):
        st.dataframe(pd.DataFrame({"column": df.columns, "missing": df.isna().sum().values}))

st.caption("Data: NASA FLEX, NASA/TP-2015-216046, Table IX. Built with Claude (Anthropic) as a coding assistant. "
           "Independent hackathon project, not affiliated with or endorsed by NASA.")