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
    legend=dict(
        bgcolor="#1a2035", bordercolor="#2a3550",
        borderwidth=1, font=dict(size=11),
        orientation="h", yanchor="bottom",
        y=1.02, xanchor="left", x=0
    ),
    margin=dict(l=50, r=20, t=40, b=50),
    hovermode="x unified"
)

COHORT_COLORS = {
    "Reference":               "#94a3b8",
    "Cohort I (W-W-T-W-W)":   "#60a5fa",
    "Cohort II (W-W-W-T-W)":  "#34d399",
    "Cohort III (W-W-W-W-T)": "#f87171",
}

# ============================================================
# DEFAULT PARAMETERS
# ============================================================
DEFAULT = dict(
    sema_ka=0.02, sema_CL=0.066, sema_V1=3.5,
    sema_Q=0.12,  sema_V2=7.0,   sema_F=0.89,
    dwj_ka=0.008, dwj_CL=0.010,  dwj_V1=3.0,
    dwj_Q=0.05,   dwj_V2=6.0,    dwj_F=0.75,
    dwj_kon=0.091, dwj_koff=0.001,
    dwj_kint=0.005, dwj_ksyn=1.0, dwj_kdeg=0.05,
    bw_base=100, bw_kin=0.0001, bw_kout=0.0001,
    bw_Emax_s=0.8, bw_EC50_s=50,
    bw_Emax_d=0.6, bw_EC50_d=20,
    gi_E0=0.05, gi_Emax=0.95, gi_EC50=80, gi_hill=1.5
)

# ============================================================
# ODE SYSTEM
# ============================================================
def pkpd_ode(y, t, p, dose_fn_sema, dose_fn_dwj):
    A_sd, A_sc, A_sp, A_dd, A_dc, A_dp, R_free, RC, BW = y

    # Semaglutide PK
    dA_sd = -p['ka_s'] * A_sd + dose_fn_sema(t)
    C_sc  = A_sc / p['V1_s']
    dA_sc = p['ka_s'] * p['F_s'] * A_sd \
            - (p['CL_s']/p['V1_s'] + p['Q_s']/p['V1_s']) * A_sc \
            + (p['Q_s']/p['V2_s']) * A_sp
    dA_sp = (p['Q_s']/p['V1_s']) * A_sc - (p['Q_s']/p['V2_s']) * A_sp
    C_sema = C_sc * 1000  # ug/L

    # DWJ1691 PK (TMDD simplified)
    dA_dd = -p['ka_d'] * A_dd + dose_fn_dwj(t)
    C_dc  = A_dc / p['V1_d']
    dA_dc = p['ka_d'] * p['F_d'] * A_dd \
            - (p['CL_d']/p['V1_d'] + p['Q_d']/p['V1_d']) * A_dc \
            + (p['Q_d']/p['V2_d']) * A_dp \
            - p['kon'] * C_dc * R_free * p['V1_d'] \
            + p['koff'] * RC * p['V1_d']
    dA_dp = (p['Q_d']/p['V1_d']) * A_dc - (p['Q_d']/p['V2_d']) * A_dp
    C_dwj = C_dc * 1000  # ug/L

    # TMDD
    dR_free = p['ksyn'] - p['kdeg']*R_free \
              - p['kon']*C_dc*R_free + p['koff']*RC
    dRC     = p['kon']*C_dc*R_free - p['koff']*RC - p['kint']*RC

    # PD: Body weight
    IS    = p['Emax_s'] * C_sema / (p['EC50_s'] + C_sema + 1e-10)
    ID    = p['Emax_d'] * C_dwj  / (p['EC50_d'] + C_dwj  + 1e-10)
    Icomb = 1 - (1 - IS) * (1 - ID)
    dBW   = p['kin_bw'] * (1 - Icomb) - p['kout_bw'] * BW

    return [dA_sd, dA_sc, dA_sp, dA_dd, dA_dc, dA_dp, dR_free, dRC, dBW]

# ============================================================
# DOSING HELPERS
# ============================================================
def make_dose_fn(dose_times_h, dose_amt, duration_h=1.0):
    """Returns a continuous infusion-like dose function."""
    def fn(t):
        for dt in dose_times_h:
            if dt <= t < dt + duration_h:
                return dose_amt / duration_h
        return 0.0
    return fn

def sema_schedule(skip_block=None):
    levels = [0.25, 0.5, 1.0, 1.7, 2.4]
    times  = []
    for block in range(5):
        if block == skip_block:
            continue
        for w in range(4):
            times.append((block * 28 + w * 7) * 24)
    return times, levels

def build_sema_doses(skip_block=None):
    levels = [0.25, 0.5, 1.0, 1.7, 2.4]
    times_h, amts = [], []
    for block in range(5):
        if block == skip_block:
            continue
        dose = levels[block] * 1000 * DEFAULT['sema_F']  # mg -> ug
        for w in range(4):
            t = (block * 28 + w * 7) * 24
            times_h.append(t)
            amts.append(dose)
    return times_h, amts

def build_dwj_doses(skip_block, dwj_dose_mg):
    if skip_block is None:
        return [], []
    t = skip_block * 28 * 24
    amt = dwj_dose_mg * 1000 * DEFAULT['dwj_F']
    return [t], [amt]

def make_pulsed_fn(times_h, amts, dur=1.0):
    pairs = list(zip(times_h, amts))
    def fn(t):
        for (dt, amt) in pairs:
            if dt <= t < dt + dur:
                return amt / dur
        return 0.0
    return fn

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
                   ec50_s, ec50_d, gi_ec50):
    p = dict(
        ka_s=DEFAULT['sema_ka'], CL_s=DEFAULT['sema_CL'],
        V1_s=DEFAULT['sema_V1'], Q_s=DEFAULT['sema_Q'],
        V2_s=DEFAULT['sema_V2'], F_s=DEFAULT['sema_F'],
        ka_d=DEFAULT['dwj_ka'],  CL_d=DEFAULT['dwj_CL'],
        V1_d=DEFAULT['dwj_V1'], Q_d=DEFAULT['dwj_Q'],
        V2_d=DEFAULT['dwj_V2'], F_d=DEFAULT['dwj_F'],
        kon=DEFAULT['dwj_kon'], koff=DEFAULT['dwj_koff'],
        kint=DEFAULT['dwj_kint'], ksyn=DEFAULT['dwj_ksyn'],
        kdeg=DEFAULT['dwj_kdeg'],
        kin_bw=DEFAULT['bw_kin'], kout_bw=DEFAULT['bw_kout'],
        Emax_s=DEFAULT['bw_Emax_s'], EC50_s=ec50_s,
        Emax_d=DEFAULT['bw_Emax_d'], EC50_d=ec50_d,
        E0_gi=DEFAULT['gi_E0'], Emax_gi=DEFAULT['gi_Emax'],
        EC50_gi=gi_ec50, hill_gi=DEFAULT['gi_hill']
    )
    R0 = p['ksyn'] / p['kdeg']
    y0 = [0, 0, 0, 0, 0, 0, R0, 0, DEFAULT['bw_base']]

    t_end = sim_weeks * 7 * 24
    t = np.linspace(0, t_end, sim_weeks * 7 * 24 + 1)

    results = {}
    for coh in cohorts_tuple:
        skip = COHORT_SKIP[coh]
        st_h, sa = build_sema_doses(skip)
        dt_h, da = build_dwj_doses(skip, dwj_dose_mg)
        fn_s = make_pulsed_fn(st_h, sa)
        fn_d = make_pulsed_fn(dt_h, da)

        sol = odeint(pkpd_ode, y0, t,
                     args=(p, fn_s, fn_d),
                     mxstep=5000)

        C_sema = sol[:, 1] / p['V1_s'] * 1000
        C_dwj  = sol[:, 4] / p['V1_d'] * 1000
        BW     = sol[:, 8]
        C_pk   = C_sema + 0.5 * C_dwj
        GI     = p['E0_gi'] + (p['Emax_gi'] - p['E0_gi']) * \
                 C_pk**p['hill_gi'] / \
                 (p['EC50_gi']**p['hill_gi'] + C_pk**p['hill_gi'] + 1e-10)

        results[coh] = dict(
            t_weeks = t / (7 * 24),
            C_sema  = C_sema,
            C_dwj   = C_dwj,
            BW_pct  = (BW - DEFAULT['bw_base']) / DEFAULT['bw_base'] * 100,
            GI_rate = GI * 100
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
    dwj_dose = st.slider("Monthly SC dose (mg)", 1, 50, 10, 1)

    st.markdown('<div class="section-header">Simulation</div>',
                unsafe_allow_html=True)
    sim_weeks = st.slider("Duration (weeks)", 12, 36, 25, 1)

    st.markdown("---")
    st.markdown('<div class="section-header">PD Parameters</div>',
                unsafe_allow_html=True)
    ec50_s   = st.slider("Sema EC50 BW (µg/L)",  10, 150, 50, 5)
    ec50_d   = st.slider("DWJ EC50 BW (µg/L)",    5,  80, 20, 5)
    gi_ec50  = st.slider("GI EC50 (µg/L)",        20, 200, 80, 10)

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
        ec50_s, ec50_d, gi_ec50
    )

# ---- KPI Cards ----
all_sema = np.concatenate([r['C_sema'] for r in results.values()])
all_dwj  = np.concatenate([r['C_dwj']  for r in results.values()])
all_bw   = np.concatenate([r['BW_pct'] for r in results.values()])
all_gi   = np.concatenate([r['GI_rate'] for r in results.values()])

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Sema C<sub>max</sub></div>
      <div class="kpi-value kpi-blue">{np.max(all_sema):.1f}</div>
      <div class="kpi-unit">µg/L</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">DWJ1691 C<sub>max</sub></div>
      <div class="kpi-value kpi-red">{np.max(all_dwj):.1f}</div>
      <div class="kpi-unit">µg/L</div>
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
    # Sema solid
    fig_pk.add_trace(go.Scatter(
        x=tw[::6], y=r['C_sema'][::6],
        name=f"{coh} · Sema",
        line=dict(color=col, width=2, dash='solid'),
        hovertemplate="%{y:.1f} µg/L"
    ))
    # DWJ dashed
    fig_pk.add_trace(go.Scatter(
        x=tw[::6], y=r['C_dwj'][::6],
        name=f"{coh} · DWJ1691",
        line=dict(color=col, width=1.5, dash='dot'),
        hovertemplate="%{y:.1f} µg/L"
    ))

fig_pk.update_layout(
    **PLOT_LAYOUT,
    height=340,
    xaxis_title="Time (Week)",
    yaxis_title="Plasma concentration (µg/L)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0, font=dict(size=10))
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
bw0 = DEFAULT['bw_base']
for coh, r in results.items():
    tw = r['t_weeks']
    rows.append({
        "Cohort":            coh,
        "Sema Cmax (µg/L)": round(float(np.max(r['C_sema'])), 2),
        "Sema Tmax (wk)":   round(float(tw[np.argmax(r['C_sema'])]), 1),
        "DWJ Cmax (µg/L)":  round(float(np.max(r['C_dwj'])), 2),
        "DWJ Tmax (wk)":    round(float(tw[np.argmax(r['C_dwj'])]), 1),
        "Max BW loss (%)":  round(float(np.min(r['BW_pct'])), 2),
        "Peak GI AE (%)":   round(float(np.max(r['GI_rate'])), 1),
    })

df = pd.DataFrame(rows)
st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Cohort":           st.column_config.TextColumn(width="large"),
        "Sema Cmax (µg/L)": st.column_config.NumberColumn(format="%.2f"),
        "DWJ Cmax (µg/L)":  st.column_config.NumberColumn(format="%.2f"),
        "Max BW loss (%)":  st.column_config.NumberColumn(format="%.2f"),
        "Peak GI AE (%)":   st.column_config.NumberColumn(format="%.1f"),
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
  PK/PD Simulator · Demo v1.0 · Placeholder parameters —
  replace with NONMEM/Monolix/Phoenix NLME estimates
</div>
""", unsafe_allow_html=True)
