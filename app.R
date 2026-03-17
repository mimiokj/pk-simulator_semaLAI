import streamlit as st
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="DWJ1691 + Wegovy PK/PD Simulator",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a3040;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1a2035 0%, #1e2640 100%);
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: #8892a4;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value { font-size: 1.8rem; font-weight: 700; line-height: 1.1; }
    .kpi-unit  { font-size: 0.72rem; color: #6b7585; margin-top: 4px; }
    .kpi-blue  { color: #60a5fa; }
    .kpi-red   { color: #f87171; }
    .kpi-green { color: #34d399; }
    .kpi-amber { color: #fbbf24; }
    .section-header {
        font-size: 0.72rem;
        font-weight: 700;
        color: #7eb8f7;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        border-bottom: 1px solid #2a3550;
        padding-bottom: 6px;
        margin-bottom: 12px;
        margin-top: 8px;
    }
    hr { border-color: #2a3040 !important; }
    .stButton > button {
        background: #2166ac; color: white; border: none;
        border-radius: 8px; font-weight: 600;
        width: 100%; padding: 0.5rem;
    }
    .stButton > button:hover { background: #1a5490; }
    .main-title { font-size: 1.4rem; font-weight: 700; color: #e2e8f0; }
    .main-subtitle { font-size: 0.85rem; color: #64748b; margin-bottom: 20px; }
    .badge {
        display: inline-block; background: #1e3a5f; color: #60a5fa;
        font-size: 0.68rem; font-weight: 600; padding: 2px 10px;
        border-radius: 20px; border: 1px solid #2a4a7a; margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY DARK THEME
# ============================================================
PLOT_LAYOUT = dict(
    paper_bgcolor="#161b27",
    plot_bgcolor="#0f1117",
    font=dict(family="Inter, sans-serif", color="#c8d0e0", size=12),
    xaxis=dict(
        gridcolor="#1e2640", gridwidth=0.5,
        linecolor="#2a3550", tickcolor="#2a3550",
        title_font=dict(size=12, color="#8892a4")
    ),
    yaxis=dict(
        gridcolor="#1e2640", gridwidth=0.5,
        linecolor="#2a3550", tickcolor="#2a3550",
        title_font=dict(size=12, color="#8892a4")
    ),
    margin=dict(l=50, r=20, t=60, b=50),
    hovermode="x unified"
)

COHORT_COLORS = {
    "Reference":               "#94a3b8",
    "Cohort I (W-W-T-W-W)":   "#60a5fa",
    "Cohort II (W-W-W-T-W)":  "#34d399",
    "Cohort III (W-W-W-W-T)": "#f87171",
}

# ============================================================
# MODEL PARAMETERS — Phoenix NLME fixef estimates
# Modeler: Taeheon Kim, Ph.D. (2026-02-19)
# Units: Dose(µg), A1(µg), C = A1/V (µg/L), V(L), CL(L/h), rate(h-1)
# ============================================================
DEFAULT = dict(
    V         = 25.0,    # Central volume (L)
    Cl        = 3.5,     # Clearance (L/h)
    Ka        = 5.2,     # Wegovy fast-depot absorption rate (h-1)
    ka_SC     = 1.0,     # SC R-compartment absorption rate (h-1)
    F_SC      = 0.9,     # SC bioavailability
    Scale_LAI = 0.2,     # LAI dose scaling factor
    F_DR      = 0.2,     # Delayed release fraction
    kdr       = 1.0,     # Transit rate for delayed release (h-1)
    bw_base   = 100.0,   # Baseline body weight (kg)
    Imax      = 0.21,    # Maximum inhibition (0-1)
    IC50      = 55.0,    # IC50 (µg/L)
    Gamma     = 0.5,     # Hill coefficient
    kout      = 0.00039, # BW turnover rate (h-1)
    E0_AE     = 0.4833,  # Baseline GI AE (0-1)
    Emax_AE   = 0.2867,  # Max drug-induced GI AE
    EC50_AE   = 32.98,   # EC50 for GI AE (µg/L)
)

# ============================================================
# DOSING HELPERS
# ============================================================
def make_pulsed_fn(times_h, amts, dur=0.5):
    pairs = list(zip(times_h, amts))
    def fn(t):
        for (dt, amt) in pairs:
            if dt <= t < dt + dur:
                return amt / dur
        return 0.0
    return fn

def build_sema_doses(skip_block=None):
    levels_mg = [0.25, 0.5, 1.0, 1.7, 2.4]
    times_h, amts = [], []
    for block in range(5):
        if block == skip_block:
            continue
        dose_ug = levels_mg[block] * 1000 * DEFAULT['F_SC']
        for w in range(4):
            times_h.append((block * 28 + w * 7) * 24)
            amts.append(dose_ug)
    return times_h, amts

def build_lai_doses(skip_block, dwj_dose_ug):
    if skip_block is None:
        return [], [], [], []
    t_h    = skip_block * 28 * 24
    Scale  = DEFAULT['Scale_LAI']
    F_FR   = DEFAULT['F_SC'] - DEFAULT['F_DR']
    F_DR   = DEFAULT['F_DR']
    return [t_h], [dwj_dose_ug * Scale * F_FR], [t_h], [dwj_dose_ug * Scale * F_DR]

COHORT_SKIP = {
    "Reference":               None,
    "Cohort I (W-W-T-W-W)":   2,
    "Cohort II (W-W-W-T-W)":  3,
    "Cohort III (W-W-W-W-T)": 4,
}

# ============================================================
# SIMULATION
# ============================================================
@st.cache_data(show_spinner=False)
def run_simulation(cohorts_tuple, dwj_dose_ug, sim_weeks,
                   IC50_val, EC50_AE_val, kout_val):
    p = dict(
        V         = DEFAULT['V'],
        Cl        = DEFAULT['Cl'],
        Ka        = DEFAULT['Ka'],
        ka_SC     = DEFAULT['ka_SC'],
        F_SC      = DEFAULT['F_SC'],
        Scale_LAI = DEFAULT['Scale_LAI'],
        F_DR      = DEFAULT['F_DR'],
        kdr       = DEFAULT['kdr'],
        bw_base   = DEFAULT['bw_base'],
        Imax      = DEFAULT['Imax'],
        IC50      = IC50_val,
        Gamma     = DEFAULT['Gamma'],
        kout      = kout_val,
        E0_AE     = DEFAULT['E0_AE'],
        Emax_AE   = DEFAULT['Emax_AE'],
        EC50_AE   = EC50_AE_val,
    )

    y0    = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, DEFAULT['bw_base']]
    t_end = sim_weeks * 7 * 24
    t_vec = np.linspace(0, t_end, t_end + 1)

    results = {}
    for coh in cohorts_tuple:
        skip = COHORT_SKIP[coh]
        fn_sema = make_pulsed_fn(*build_sema_doses(skip))
        fr_h, fr_a, dr_h, dr_a = build_lai_doses(skip, dwj_dose_ug)
        fn_fr = make_pulsed_fn(fr_h, fr_a)
        fn_dr = make_pulsed_fn(dr_h, dr_a)

        def ode_wrap(y, t, p=p, fn_sema=fn_sema, fn_fr=fn_fr, fn_dr=fn_dr):
            A1, FR, DR, DR1, DR2, DR3, R, BW = y
            C = A1 / p['V']

            dA1  = -(p['Cl'] * C) + (p['Ka'] * FR) + (p['kdr'] * DR3) + (p['ka_SC'] * R)
            dFR  = -(FR * p['Ka'])  + fn_fr(t)
            dDR  = -(DR * p['kdr']) + fn_dr(t)
            dDR1 = (DR * p['kdr'])  - (DR1 * p['kdr'])
            dDR2 = (DR1 * p['kdr']) - (DR2 * p['kdr'])
            dDR3 = (DR2 * p['kdr']) - (DR3 * p['kdr'])
            dR   = -(R * p['ka_SC']) + fn_sema(t)

            E    = (p['Imax'] * C**p['Gamma']) / (p['IC50']**p['Gamma'] + C**p['Gamma'] + 1e-12)
            kin  = p['kout'] * (100.0 - 6.0 * (1.0 - np.exp(-0.0001 * t)))
            dBW  = kin * (1.0 - E) - p['kout'] * BW

            return [dA1, dFR, dDR, dDR1, dDR2, dDR3, dR, dBW]

        sol    = odeint(ode_wrap, y0, t_vec, mxstep=10000)
        C_ugL  = sol[:, 0] / p['V']
        BW_arr = sol[:, 7]
        GI     = p['E0_AE'] + p['Emax_AE'] * C_ugL / (p['EC50_AE'] + C_ugL + 1e-12)

        results[coh] = dict(
            t_hours = t_vec,
            C_ugL   = C_ugL,
            BW_pct  = (BW_arr - DEFAULT['bw_base']) / DEFAULT['bw_base'] * 100,
            GI_rate = GI * 100,
        )
    return results

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 💊 DWJ1691 + Wegovy")
    st.markdown("**PK/PD Simulator** `v1.0`")
    st.markdown("---")

    st.markdown('<div class="section-header">Cohort Selection</div>', unsafe_allow_html=True)
    sel = {}
    for coh in COHORT_COLORS:
        sel[coh] = st.checkbox(coh, value=True, key=f"chk_{coh}")

    st.markdown("---")
    st.markdown('<div class="section-header">DWJ1691 Dose</div>', unsafe_allow_html=True)
    dwj_dose = st.slider("Monthly SC dose (µg)", 100, 10000, 1000, 100)

    st.markdown('<div class="section-header">Simulation</div>', unsafe_allow_html=True)
    sim_weeks = st.slider("Duration (weeks)", 12, 36, 25, 1)

    st.markdown("---")
    st.markdown('<div class="section-header">PD Parameters</div>', unsafe_allow_html=True)
    IC50_slider    = st.slider("IC50 — BW (µg/L)",    10, 150, int(DEFAULT['IC50']),    5)
    EC50_AE_slider = st.slider("EC50 — GI AE (µg/L)",  5, 100, int(DEFAULT['EC50_AE']), 5)
    kout_slider    = st.slider("kout × 10⁻⁴ (h⁻¹)",   1,  20, int(DEFAULT['kout']*10000), 1)

    st.markdown("---")
    st.button("▶  Run Simulation")

# ============================================================
# MAIN
# ============================================================
st.markdown("""
<div class="main-title">DWJ1691 + Wegovy
  <span class="badge">PK/PD/Safety</span>
  <span class="badge">Phoenix NLME</span>
</div>
<div class="main-subtitle">
  1-Cpt Multiple Absorption (LAI + SC) · Indirect Response BW · Simple Emax GI AE
</div>
""", unsafe_allow_html=True)

active_cohorts = [c for c, v in sel.items() if v]
if not active_cohorts:
    st.warning("Please select at least one cohort.")
    st.stop()

with st.spinner("Running ODE simulation..."):
    results = run_simulation(
        tuple(active_cohorts), dwj_dose, sim_weeks,
        IC50_slider, EC50_AE_slider, kout_slider * 1e-4
    )

# KPI Cards
all_C  = np.concatenate([r['C_ugL']  for r in results.values()])
all_bw = np.concatenate([r['BW_pct'] for r in results.values()])
all_gi = np.concatenate([r['GI_rate'] for r in results.values()])

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">C<sub>max</sub></div>
      <div class="kpi-value kpi-blue">{np.max(all_C):.1f}</div>
      <div class="kpi-unit">µg/L</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">IC50 / EC50<sub>AE</sub></div>
      <div class="kpi-value kpi-red">{IC50_slider} / {EC50_AE_slider}</div>
      <div class="kpi-unit">µg/L</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Max BW Loss</div>
      <div class="kpi-value kpi-green">{np.min(all_bw):.1f}%</div>
      <div class="kpi-unit">body weight</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Peak GI AE</div>
      <div class="kpi-value kpi-amber">{np.max(all_gi):.1f}%</div>
      <div class="kpi-unit">adverse event</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# PK Chart
st.markdown('<div class="section-header">PK Profile — Plasma Concentration</div>', unsafe_allow_html=True)
fig_pk = go.Figure()
for coh, r in results.items():
    fig_pk.add_trace(go.Scatter(
        x=r['t_hours'][::6], y=r['C_ugL'][::6], name=coh,
        line=dict(color=COHORT_COLORS[coh], width=2),
        hovertemplate="%{x:.0f} h — %{y:.2f} µg/L"
    ))
fig_pk.update_layout(**PLOT_LAYOUT, height=340,
    xaxis_title="Time (Hour)", yaxis_title="Plasma concentration (µg/L)",
    legend=dict(bgcolor="#1a2035", bordercolor="#2a3550", borderwidth=1,
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)))
st.plotly_chart(fig_pk, use_container_width=True)

# BW + GI
col_bw, col_gi = st.columns(2)
with col_bw:
    st.markdown('<div class="section-header">Body Weight Change (%BW)</div>', unsafe_allow_html=True)
    fig_bw = go.Figure()
    for coh, r in results.items():
        fig_bw.add_trace(go.Scatter(
            x=r['t_hours'][::6], y=r['BW_pct'][::6], name=coh,
            line=dict(color=COHORT_COLORS[coh], width=2),
            hovertemplate="%{x:.0f} h — %{y:.2f}%"
        ))
    fig_bw.update_layout(**PLOT_LAYOUT, height=280,
        xaxis_title="Time (Hour)", yaxis_title="BW change (%)", showlegend=False)
    st.plotly_chart(fig_bw, use_container_width=True)

with col_gi:
    st.markdown('<div class="section-header">GI Adverse Event Rate</div>', unsafe_allow_html=True)
    fig_gi = go.Figure()
    for coh, r in results.items():
        fig_gi.add_trace(go.Scatter(
            x=r['t_hours'][::6], y=r['GI_rate'][::6], name=coh,
            line=dict(color=COHORT_COLORS[coh], width=2),
            hovertemplate="%{x:.0f} h — %{y:.1f}%"
        ))
    fig_gi.update_layout(**PLOT_LAYOUT, height=280,
        xaxis_title="Time (Hour)", yaxis_title="GI AE rate (%)",
        yaxis_range=[0, 100], showlegend=False)
    st.plotly_chart(fig_gi, use_container_width=True)

# Summary Table
st.markdown('<div class="section-header">PK Summary Table</div>', unsafe_allow_html=True)
rows = []
for coh, r in results.items():
    th = r['t_hours']
    rows.append({
        "Cohort":           coh,
        "Cmax (µg/L)":     round(float(np.max(r['C_ugL'])), 2),
        "Tmax (h)":        round(float(th[np.argmax(r['C_ugL'])]), 1),
        "Clast (µg/L)":    round(float(r['C_ugL'][-1]), 2),
        "Max BW loss (%)": round(float(np.min(r['BW_pct'])), 2),
        "Peak GI AE (%)":  round(float(np.max(r['GI_rate'])), 1),
    })
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True,
    column_config={
        "Cohort":          st.column_config.TextColumn(width="large"),
        "Cmax (µg/L)":     st.column_config.NumberColumn(format="%.2f"),
        "Tmax (h)":        st.column_config.NumberColumn(format="%.1f"),
        "Clast (µg/L)":    st.column_config.NumberColumn(format="%.2f"),
        "Max BW loss (%)": st.column_config.NumberColumn(format="%.2f"),
        "Peak GI AE (%)":  st.column_config.NumberColumn(format="%.1f"),
    })

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download Summary CSV", csv, file_name="pkpd_summary.csv", mime="text/csv")

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#3a4560; font-size:0.75rem; padding:8px 0'>
  Phoenix NLME Model · Taeheon Kim, Ph.D. · 2026-02-19 ·
  1-Cpt Multiple Absorption (LAI + SC) · Indirect Response BW · Simple Emax GI AE
</div>
""", unsafe_allow_html=True)
```

---

**GitHub 붙여넣기 방법:**
```
GitHub → app.py → ✏️ Edit
→ Ctrl+A (전체 선택) → Delete
→ 위 코드 전체 Ctrl+V 붙여넣기
→ Commit changes
