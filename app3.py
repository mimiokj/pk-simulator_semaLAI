import streamlit as st
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="DWJ1691 + Wegovy PK/PD Simulator",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Dark Academic Style
# ============================================================
st.markdown("""
<style>
    /* Base dark theme */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a3040;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #7eb8f7;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a2035 0%, #1e2640 100%);
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .kpi-card:hover { border-color: #4a6090; }
    .kpi-label {
        font-size: 0.72rem;
        color: #8892a4;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .kpi-unit {
        font-size: 0.72rem;
        color: #6b7585;
        margin-top: 4px;
    }
    .kpi-blue  { color: #60a5fa; }
    .kpi-red   { color: #f87171; }
    .kpi-green { color: #34d399; }
    .kpi-amber { color: #fbbf24; }

    /* Section headers */
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

    /* Chart container */
    .chart-container {
        background: #161b27;
        border: 1px solid #2a3040;
        border-radius: 12px;
        padding: 16px;
    }

    /* Table styling */
    .dataframe {
        background-color: #161b27 !important;
        color: #c8d0e0 !important;
    }

    /* Divider */
    hr { border-color: #2a3040 !important; }

    /* Streamlit elements */
    .stSlider > div > div > div { background: #2a3550; }
    .stCheckbox > label { color: #c8d0e0 !important; }
    .stButton > button {
        background: #2166ac;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        padding: 0.5rem;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #1a5490; }

    /* Title area */
    .main-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: -0.01em;
    }
    .main-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 2px;
        margin-bottom: 20px;
    }
    .badge {
        display: inline-block;
        background: #1e3a5f;
        color: #60a5fa;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 20px;
        border: 1px solid #2a4a7a;
        margin-left: 8px;
        vertical-align: middle;
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
# Units: Dose(µg), A1(µg), C = A1/V (µg/L), V(mL→L*1000), CL(L/h), rate(h-1)
# Note: V(L) from Phoenix → V_ug = V * 1000 (mL) to get C in µg/L
#       e.g. V=25L → V_eff=25000 mL, C(µg/L) = A1(µg)/25(L) = A1/V directly
#       because 1 µg / 1 L = 1 µg/L  ✓  (V kept in L, dose in µg)
# ============================================================
DEFAULT = dict(
    # Shared 1-cpt PK
    V        = 25.0,    # Central volume (L)
    Cl       = 3.5,     # Clearance (L/h)
    # Wegovy SC absorption
    Ka       = 5.2,     # Wegovy fast-depot absorption rate (h-1)
    ka_SC    = 1.0,     # SC R-compartment absorption rate (h-1)
    F_SC     = 0.9,     # SC bioavailability
    # DWJ1691 LAI absorption
    Scale_LAI = 0.2,    # LAI dose scaling factor
    F_DR     = 0.2,     # Delayed release fraction
    kdr      = 1.0,     # Transit rate for delayed release (h-1)
    # BW PD — Indirect Response, sigmoidal Imax
    bw_base  = 100.0,   # Baseline body weight (kg)
    Imax     = 0.21,    # Maximum inhibition (0-1)
    IC50     = 55.0,    # IC50 (µg/L)
    Gamma    = 0.5,     # Hill coefficient
    kout     = 0.00039, # BW turnover rate (h-1)
    # GI AE — Simple Emax
    E0_AE    = 0.4833,  # Baseline GI AE (0-1)
    Emax_AE  = 0.2867,  # Max drug-induced GI AE
    EC50_AE  = 32.98,   # EC50 for GI AE (µg/L)
)

# ============================================================
# ODE SYSTEM — Phoenix NLME PML (Taeheon Kim, Ph.D.)
#
# State variables:
#   A1            : Central compartment (µg)
#   FR            : Fast-release LAI depot (µg)
#   DR/DR1/DR2/DR3: 3-transit delayed-release chain (µg)
#   R             : Wegovy SC absorption depot (µg)
#   BW            : Body weight (kg)
#
# Derived:
#   C (µg/L) = A1(µg) / V(L)   ← units consistent ✓
#   F_FR = F_SC - F_DR
#   kin  = kout * Current_Baseline
# ============================================================
def pkpd_ode(y, t, p, dose_fn_sema, dose_fn_lai):
    A1, FR, DR, DR1, DR2, DR3, R, BW = y

    V   = p['V']
    Cl  = p['Cl']
    Ka  = p['Ka']
    ka_SC = p['ka_SC']
    kdr = p['kdr']

    # Concentration (µg/L) = A1(µg) / V(L)
    C = A1 / V

    # Fractions
    F_SC = p['F_SC']
    F_DR = p['F_DR']
    F_FR = F_SC - F_DR          # fast-release fraction

    Scale_LAI = p['Scale_LAI']

    # --- PK ODEs (Phoenix PML translated) ---
    # Central compartment: receives FR (fast LAI), DR3 (delayed LAI), R (SC)
    dA1  = -(Cl * C) + (Ka * FR) + (kdr * DR3) + (ka_SC * R)

    # Fast-release LAI depot
    dFR  = -(FR * Ka)

    # 3-transit delayed-release chain
    dDR  = -(DR * kdr)
    dDR1 = (DR * kdr)  - (DR1 * kdr)
    dDR2 = (DR1 * kdr) - (DR2 * kdr)
    dDR3 = (DR2 * kdr) - (DR3 * kdr)

    # Wegovy SC depot (R compartment)
    dR   = -(R * ka_SC) + dose_fn_sema(t)

    # --- PD: Body Weight (Indirect Response, sigmoidal Imax) ---
    Imax  = p['Imax']
    IC50  = p['IC50']
    Gamma = p['Gamma']
    kout  = p['kout']

    E = (Imax * C**Gamma) / (IC50**Gamma + C**Gamma + 1e-12)
    Current_Baseline = 100.0 - (6.0 * (1.0 - np.exp(-0.0001 * t)))
    kin = kout * Current_Baseline
    dBW = kin * (1.0 - E) - kout * BW

    return [dA1, dFR, dDR, dDR1, dDR2, dDR3, dR, dBW]

# ============================================================
# DOSING HELPERS
# ============================================================
def make_pulsed_fn(times_h, amts, dur=0.5):
    """Bolus dose approximated as short infusion."""
    pairs = list(zip(times_h, amts))
    def fn(t):
        for (dt, amt) in pairs:
            if dt <= t < dt + dur:
                return amt / dur
        return 0.0
    return fn

def build_sema_doses(skip_block=None):
    """Wegovy once-weekly SC, 5-step escalation (4wk each block).
    Dose in µg (e.g. 0.25mg = 250µg).
    """
    levels_mg = [0.25, 0.5, 1.0, 1.7, 2.4]   # mg
    times_h, amts = [], []
    for block in range(5):
        if block == skip_block:
            continue
        dose_ug = levels_mg[block] * 1000 * DEFAULT['F_SC']  # mg→µg, bioavail
        for w in range(4):
            t = (block * 28 + w * 7) * 24    # hours
            times_h.append(t)
            amts.append(dose_ug)
    return times_h, amts

def build_lai_doses(skip_block, dwj_dose_ug):
    """DWJ1691 LAI once-monthly SC injection.
    dwj_dose_ug: dose in µg
    Splits into FR (fast) and DR (delayed) fractions via Scale_LAI.
    """
    if skip_block is None:
        return [], [], [], []

    t_h       = skip_block * 28 * 24
    Scale     = DEFAULT['Scale_LAI']
    F_DR      = DEFAULT['F_DR']
    F_SC      = DEFAULT['F_SC']
    F_FR      = F_SC - F_DR

    amt_FR    = dwj_dose_ug * Scale * F_FR   # fast-release (µg)
    amt_DR    = dwj_dose_ug * Scale * F_DR   # delayed-release (µg)

    return [t_h], [amt_FR], [t_h], [amt_DR]

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
def run_simulation(cohorts_tuple, dwj_dose_mg, sim_weeks,
                   IC50_slider, EC50_AE_slider, kout_slider):
    p = dict(
        V        = DEFAULT['V'],
        Cl       = DEFAULT['Cl'],
        Ka       = DEFAULT['Ka'],
        ka_SC    = DEFAULT['ka_SC'],
        F_SC     = DEFAULT['F_SC'],
        Scale_LAI= DEFAULT['Scale_LAI'],
        F_DR     = DEFAULT['F_DR'],
        kdr      = DEFAULT['kdr'],
        bw_base  = DEFAULT['bw_base'],
        Imax     = DEFAULT['Imax'],
        IC50     = IC50_slider,
        Gamma    = DEFAULT['Gamma'],
        kout     = kout_slider,
        E0_AE    = DEFAULT['E0_AE'],
        Emax_AE  = DEFAULT['Emax_AE'],
        EC50_AE  = EC50_AE_slider,
    )

    # Initial conditions: all depots empty, BW at baseline
    y0 = [0.0,   # A1
          0.0,   # FR
          0.0,   # DR
          0.0,   # DR1
          0.0,   # DR2
          0.0,   # DR3
          0.0,   # R (Wegovy SC depot)
          DEFAULT['bw_base']]  # BW

    t_end = sim_weeks * 7 * 24   # hours
    t_vec = np.linspace(0, t_end, sim_weeks * 7 * 24 + 1)

    results = {}
    for coh in cohorts_tuple:
        skip = COHORT_SKIP[coh]

        # Wegovy SC doses → R compartment via dose_fn
        st_h, sa = build_sema_doses(skip)
        fn_sema  = make_pulsed_fn(st_h, sa)

        # DWJ1691 LAI doses → split into FR and DR at t=0 of injection
        fr_h, fr_a, dr_h, dr_a = build_lai_doses(skip, dwj_dose)
        fn_fr = make_pulsed_fn(fr_h, fr_a)
        fn_dr = make_pulsed_fn(dr_h, dr_a)

        # Wrap into single LAI function used inside ODE for FR/DR injection
        # (FR and DR depots receive their respective fractions at dose time)
        def make_lai_fn(ffr, fdr):
            def fn(t):
                return ffr(t), fdr(t)
            return fn
        fn_lai = make_lai_fn(fn_fr, fn_dr)

        # Custom ODE wrapper to inject FR and DR separately
        def ode_wrap(y, t, p=p, fn_sema=fn_sema,
                     fn_fr=fn_fr, fn_dr=fn_dr):
            A1, FR, DR, DR1, DR2, DR3, R, BW = y
            V     = p['V'];  Cl = p['Cl']
            Ka    = p['Ka']; ka_SC = p['ka_SC']; kdr = p['kdr']
            C     = A1 / V
            F_SC  = p['F_SC']; F_DR = p['F_DR']

            # Dose injections
            dose_R  = fn_sema(t)   # → R depot
            dose_FR = fn_fr(t)     # → FR depot
            dose_DR = fn_dr(t)     # → DR depot

            dA1  = -(Cl * C) + (Ka * FR) + (kdr * DR3) + (ka_SC * R)
            dFR  = -(FR * Ka)  + dose_FR
            dDR  = -(DR * kdr) + dose_DR
            dDR1 = (DR * kdr)  - (DR1 * kdr)
            dDR2 = (DR1 * kdr) - (DR2 * kdr)
            dDR3 = (DR2 * kdr) - (DR3 * kdr)
            dR   = -(R * ka_SC) + dose_R

            Imax  = p['Imax'];  IC50 = p['IC50']; Gamma = p['Gamma']
            kout  = p['kout']
            E = (Imax * C**Gamma) / (IC50**Gamma + C**Gamma + 1e-12)
            Current_Baseline = 100.0 - (6.0 * (1.0 - np.exp(-0.0001 * t)))
            kin = kout * Current_Baseline
            dBW = kin * (1.0 - E) - kout * BW

            return [dA1, dFR, dDR, dDR1, dDR2, dDR3, dR, dBW]

        sol = odeint(ode_wrap, y0, t_vec, mxstep=10000)

        C_ugL  = sol[:, 0] / p['V']                      # µg/L
        BW_arr = sol[:, 7]
        GI     = p['E0_AE'] + p['Emax_AE'] * C_ugL / \
                 (p['EC50_AE'] + C_ugL + 1e-12)

        results[coh] = dict(
            t_weeks = t_vec / (7 * 24),
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
    st.markdown("**PK/PD Simulator** `Demo v1.0`")
    st.markdown("---")

    st.markdown('<div class="section-header">Cohort Selection</div>',
                unsafe_allow_html=True)
    sel = {}
    for coh, col in COHORT_COLORS.items():
        sel[coh] = st.checkbox(coh, value=True,
                               key=f"chk_{coh}")

    st.markdown("---")
    st.markdown('<div class="section-header">DWJ1691 Dose</div>',
                unsafe_allow_html=True)
    dwj_dose = st.slider("Monthly SC dose (µg)", 100, 10000, 1000, 100)

    st.markdown('<div class="section-header">Simulation</div>',
                unsafe_allow_html=True)
    sim_weeks = st.slider("Duration (weeks)", 12, 36, 25, 1)

    st.markdown('<div class="section-header">PD Parameters</div>',
                unsafe_allow_html=True)
    IC50_slider    = st.slider("IC50 — BW (mg/L)",   10, 150,
                                int(DEFAULT['IC50']),  5)
    EC50_AE_slider = st.slider("EC50 — GI AE (mg/L)", 5, 100,
                                int(DEFAULT['EC50_AE']), 5)
    kout_slider    = st.slider("kout × 10⁻⁴ (h⁻¹)",  1, 20,
                                int(DEFAULT['kout']*10000), 1)
    kout_val = kout_slider * 1e-4

    st.markdown("---")
    run = st.button("▶  Run Simulation")

# ============================================================
# MAIN AREA
# ============================================================

# Title
st.markdown("""
<div class="main-title">
  DWJ1691 + Wegovy &nbsp;
  <span class="badge">PK/PD/Safety</span>
  <span class="badge">Demo</span>
</div>
<div class="main-subtitle">
  Integrated pharmacokinetic · pharmacodynamic · safety simulation
</div>
""", unsafe_allow_html=True)

# Run simulation
active_cohorts = [c for c, v in sel.items() if v]

if not active_cohorts:
    st.warning("Please select at least one cohort.")
    st.stop()

with st.spinner("Running ODE simulation..."):
    results = run_simulation(
        tuple(active_cohorts), dwj_dose, sim_weeks,
        IC50_slider, EC50_AE_slider, kout_val
    )

# ---- KPI Cards ----
all_C  = np.concatenate([r['C_ugL']  for r in results.values()])
all_bw = np.concatenate([r['BW_pct'] for r in results.values()])
all_gi = np.concatenate([r['GI_rate'] for r in results.values()])

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">C<sub>max</sub> (combined)</div>
      <div class="kpi-value kpi-blue">{np.max(all_C):.1f}</div>
      <div class="kpi-unit">µg/L</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">IC50 / EC50<sub>AE</sub></div>
      <div class="kpi-value kpi-red">{IC50_slider} / {EC50_AE_slider}</div>
      <div class="kpi-unit">mg/L</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Max BW Loss</div>
      <div class="kpi-value kpi-green">{np.min(all_bw):.1f}%</div>
      <div class="kpi-unit">body weight</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Peak GI AE Rate</div>
      <div class="kpi-value kpi-amber">{np.max(all_gi):.1f}%</div>
      <div class="kpi-unit">adverse event</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---- PK Profile ----
st.markdown('<div class="section-header">PK Profile — Plasma Concentration</div>',
            unsafe_allow_html=True)

fig_pk = go.Figure()
for coh, r in results.items():
    col = COHORT_COLORS[coh]
    tw  = r['t_weeks']
    fig_pk.add_trace(go.Scatter(
        x=tw[::6], y=r['C_ugL'][::6],
        name=coh,
        line=dict(color=col, width=2, dash='solid'),
        hovertemplate="%{y:.2f} µg/L"
    ))

fig_pk.update_layout(
    **PLOT_LAYOUT,
    height=340,
    xaxis_title="Time (Week)",
    yaxis_title="Plasma concentration (µg/L)",
    legend=dict(
        bgcolor="#1a2035", bordercolor="#2a3550", borderwidth=1,
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="left", x=0, font=dict(size=10)
    )
)
st.plotly_chart(fig_pk, use_container_width=True)

# ---- BW + GI side by side ----
col_bw, col_gi = st.columns(2)

with col_bw:
    st.markdown('<div class="section-header">Body Weight Change (%BW)</div>',
                unsafe_allow_html=True)
    fig_bw = go.Figure()
    for coh, r in results.items():
        fig_bw.add_trace(go.Scatter(
            x=r['t_weeks'][::6], y=r['BW_pct'][::6],
            name=coh,
            line=dict(color=COHORT_COLORS[coh], width=2),
            hovertemplate="%{y:.2f}%"
        ))
    fig_bw.update_layout(
        **PLOT_LAYOUT, height=280,
        xaxis_title="Time (Week)",
        yaxis_title="BW change (%)",
        showlegend=False
    )
    st.plotly_chart(fig_bw, use_container_width=True)

with col_gi:
    st.markdown('<div class="section-header">GI Adverse Event Rate</div>',
                unsafe_allow_html=True)
    fig_gi = go.Figure()
    for coh, r in results.items():
        fig_gi.add_trace(go.Scatter(
            x=r['t_weeks'][::6], y=r['GI_rate'][::6],
            name=coh,
            line=dict(color=COHORT_COLORS[coh], width=2),
            hovertemplate="%{y:.1f}%"
        ))
    fig_gi.update_layout(
        **PLOT_LAYOUT, height=280,
        xaxis_title="Time (Week)",
        yaxis_title="GI AE rate (%)",
        yaxis_range=[0, 100],
        showlegend=False
    )
    st.plotly_chart(fig_gi, use_container_width=True)

# ---- PK Summary Table ----
st.markdown('<div class="section-header">PK Summary Table</div>',
            unsafe_allow_html=True)

rows = []
for coh, r in results.items():
    tw = r['t_weeks']
    rows.append({
        "Cohort":           coh,
        "Cmax (µg/L)":     round(float(np.max(r['C_ugL'])), 2),
        "Tmax (wk)":       round(float(tw[np.argmax(r['C_ugL'])]), 1),
        "Clast (µg/L)":    round(float(r['C_ugL'][-1]), 2),
        "Max BW loss (%)": round(float(np.min(r['BW_pct'])), 2),
        "Peak GI AE (%)":  round(float(np.max(r['GI_rate'])), 1),
    })

df = pd.DataFrame(rows)
st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Cohort":          st.column_config.TextColumn(width="large"),
        "Cmax (µg/L)":     st.column_config.NumberColumn(format="%.2f"),
        "Clast (µg/L)":    st.column_config.NumberColumn(format="%.2f"),
        "Max BW loss (%)": st.column_config.NumberColumn(format="%.2f"),
        "Peak GI AE (%)":  st.column_config.NumberColumn(format="%.1f"),
    }
)

# CSV Download
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇ Download Summary CSV",
    csv,
    file_name="pkpd_summary.csv",
    mime="text/csv"
)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#3a4560; font-size:0.75rem; padding:8px 0'>
  PK/PD Simulator · Phoenix NLME Model (Taeheon Kim, Ph.D. · 2026-02-19) ·
  1-Cpt Multiple Absorption (LAI + SC) · Indirect Response BW · Simple Emax GI AE
</div>
""", unsafe_allow_html=True)
