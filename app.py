import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #f0f4f8; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a5f 0%, #1a3050 100%);
    border-right: none;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stCheckbox label { color: #cbd5e1 !important; font-size: 0.85rem; }
.sb-logo { font-size: 1.1rem; font-weight: 700; color: #ffffff !important; letter-spacing: -0.02em; }
.sb-hdr {
    font-size: 0.62rem; font-weight: 700; color: #60a5fa !important;
    text-transform: uppercase; letter-spacing: 0.1em;
    border-bottom: 1px solid #2d4a6e; padding-bottom: 4px; margin: 14px 0 8px 0;
}
.dose-chip {
    display: inline-block; background: rgba(96,165,250,0.15);
    border: 1px solid rgba(96,165,250,0.3); border-radius: 6px;
    padding: 4px 10px; font-size: 0.78rem; color: #93c5fd !important;
    margin-bottom: 5px; width: 100%;
}
.dose-chip b { color: #60a5fa !important; }
.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2166ac 100%);
    border-radius: 14px; padding: 20px 28px; margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(33,102,172,0.25);
}
.main-title { font-size: 1.5rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em; margin-bottom: 4px; }
.main-sub { font-size: 0.82rem; color: #93c5fd; margin-bottom: 10px; }
.badge {
    display: inline-block; background: rgba(255,255,255,0.15); color: #ffffff;
    font-size: 0.65rem; font-weight: 600; padding: 3px 10px; border-radius: 20px;
    margin-right: 5px; border: 1px solid rgba(255,255,255,0.2);
}
.design-strip {
    background: rgba(255,255,255,0.1); border-radius: 8px;
    padding: 8px 14px; font-size: 0.78rem; color: #bfdbfe;
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
}
.design-item { display: flex; align-items: center; gap: 5px; }
.design-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.kpi-card {
    background: #ffffff; border-radius: 12px; padding: 18px 20px; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07); border-top: 3px solid transparent;
}
.kpi-blue { border-top-color: #2166ac; } .kpi-red { border-top-color: #dc2626; }
.kpi-green { border-top-color: #16a34a; } .kpi-orange { border-top-color: #d97706; }
.kpi-label { font-size: 0.65rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.kpi-value { font-size: 1.8rem; font-weight: 700; line-height: 1.0; }
.kpi-unit { font-size: 0.68rem; color: #94a3b8; margin-top: 4px; }
.cv-blue { color: #2166ac; } .cv-red { color: #dc2626; }
.cv-green { color: #16a34a; } .cv-orange { color: #d97706; }
.chart-card {
    background: #ffffff; border-radius: 12px; padding: 18px 20px;
    margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.sec-hdr { font-size: 0.68rem; font-weight: 700; color: #2166ac;
    text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 2px; }
hr { border-color: #2d4a6e !important; }
.stButton > button {
    background: linear-gradient(135deg,#2166ac,#1d4ed8); color: white; border: none;
    border-radius: 8px; font-weight: 600; width: 100%; padding: 0.55rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY THEME
# ============================================================
CHART_BG = dict(
    paper_bgcolor="#ffffff", plot_bgcolor="#fafbfc",
    font=dict(family="Inter, sans-serif", color="#334155", size=12),
    xaxis=dict(gridcolor="#f1f5f9", gridwidth=1, linecolor="#e2e8f0",
               tickcolor="#e2e8f0", zeroline=False,
               title_font=dict(size=11, color="#64748b"),
               tickfont=dict(size=10, color="#94a3b8")),
    yaxis=dict(gridcolor="#f1f5f9", gridwidth=1, linecolor="#e2e8f0",
               tickcolor="#e2e8f0", zeroline=False,
               title_font=dict(size=11, color="#64748b"),
               tickfont=dict(size=10, color="#94a3b8")),
    margin=dict(l=60, r=30, t=30, b=55),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="white", bordercolor="#e2e8f0",
                    font=dict(size=11, color="#334155"))
)

COHORT_COLORS = {
    "Reference":               "#64748b",
    "Cohort I (W-W-T-W-W)":   "#2166ac",
    "Cohort II (W-W-W-T-W)":  "#16a34a",
    "Cohort III (W-W-W-W-T)": "#dc2626",
}
COHORT_DASH = {
    "Reference":               "dash",
    "Cohort I (W-W-T-W-W)":   "solid",
    "Cohort II (W-W-W-T-W)":  "solid",
    "Cohort III (W-W-W-W-T)": "solid",
}

# ============================================================
# STUDY DESIGN
# ============================================================
WEGOVY_LEVELS = [0.25, 0.5, 1.0, 1.7, 2.4]  # mg

COHORT_CONFIG = {
    "Reference":               {"skip_block": None, "dwj_block": None, "dwj_dose": 0.0},
    "Cohort I (W-W-T-W-W)":   {"skip_block": 2,    "dwj_block": 2,    "dwj_dose": WEGOVY_LEVELS[2] * 8},
    "Cohort II (W-W-W-T-W)":  {"skip_block": 3,    "dwj_block": 3,    "dwj_dose": WEGOVY_LEVELS[3] * 8},
    "Cohort III (W-W-W-W-T)": {"skip_block": 4,    "dwj_block": 4,    "dwj_dose": WEGOVY_LEVELS[4] * 8},
}
SIM_WEEKS = 28

# ============================================================
# MODEL PARAMETERS — Phoenix NLME fixef (Taeheon Kim, Ph.D.)
# ============================================================
# 단위 체계:
#   Dose   : mg  → ×1000 → µg (amount 단위)
#   Amount : µg
#   V      : L
#   C      : µg/L  (= Amount / V)
#   CL     : L/h
#   IC50   : µg/L
#   EC50   : µg/L
# ============================================================
P = dict(
    V         = 12.4,      # L        (updated)
    Cl        = 0.0475,    # L/h      (updated)
    Ka        = 0.1026,    # h⁻¹      (updated) FR → A1
    ka_SC     = 0.0296,    # h⁻¹      (updated) R  → A1
    F_SC      = 0.9,       # –        Wegovy bioavailability (fixed)
    Scale_LAI = 0.2459,    # –        (updated)
    F_DR      = 0.429,     # –        (updated) F_FR = 0.9 - 0.429 = 0.471
    kdr       = 0.02,      # h⁻¹      (updated)
    BW0       = 100.0,     # kg       baseline (fixed)
    Imax      = 0.21,      # –
    IC50      = 55.0,      # µg/L
    Gamma     = 0.5,       # –
    kout      = 0.00039,   # h⁻¹
    E0_AE     = 0.4833,    # –
    Emax_AE   = 0.2867,    # –
    EC50_AE   = 32.98,     # µg/L
)

# ============================================================
# ODE — Phoenix PML 직번역
# State: [A1, FR, DR, DR1, DR2, DR3, R, BW]
# A1~DR3, R: µg  /  C = A1/V: µg/L
# ============================================================
def build_ode(p, ev_R, ev_FR, ev_DR):
    def bolus(events, t, dur=0.5):
        val = 0.0
        for (td, amt) in events:
            if td <= t < td + dur:
                val += amt / dur
        return val

    def ode(t, y):
        A1, FR, DR, DR1, DR2, DR3, R, BW = [max(v, 0.0) for v in y]
        C = A1 / p['V']   # µg/L

        dA1  = -(p['Cl'] * C) + (p['ka_SC'] * R) + (FR * p['Ka']) + (DR3 * p['kdr'])
        dFR  = -(FR * p['Ka'])   + bolus(ev_FR, t)
        dDR  = -(DR * p['kdr'])  + bolus(ev_DR, t)
        dDR1 =  (DR  * p['kdr']) - (DR1 * p['kdr'])
        dDR2 =  (DR1 * p['kdr']) - (DR2 * p['kdr'])
        dDR3 =  (DR2 * p['kdr']) - (DR3 * p['kdr'])
        dR   = -(R * p['ka_SC']) + bolus(ev_R, t)

        # BW PD: C(µg/L), IC50(µg/L)
        E    = (p['Imax'] * C**p['Gamma']) / \
               (p['IC50']**p['Gamma'] + C**p['Gamma'] + 1e-15)
        CB   = 100.0 - 6.0 * (1.0 - np.exp(-0.0001 * t))
        dBW  = p['kout'] * CB * (1.0 - E) - p['kout'] * BW

        return [dA1, dFR, dDR, dDR1, dDR2, dDR3, dR, dBW]
    return ode

# ============================================================
# DOSE BUILDERS
# ============================================================
def build_wegovy_events(skip_block=None):
    events = []
    for blk in range(5):
        if blk == skip_block:
            continue
        amt_ug = WEGOVY_LEVELS[blk] * 1000.0 * P['F_SC']  # mg→µg, bioavail
        for w in range(4):
            t_h = (blk * 28 + w * 7) * 24.0
            events.append((t_h, amt_ug))
    return events

def build_dwj_events(dwj_block, dwj_dose_mg):
    if dwj_block is None or dwj_dose_mg <= 0:
        return [], []
    t_h    = dwj_block * 28 * 24.0
    amt_ug = dwj_dose_mg * 1000.0
    F_FR   = P['F_SC'] - P['F_DR']              # 0.471
    amt_FR = amt_ug * P['Scale_LAI'] * F_FR
    amt_DR = amt_ug * P['Scale_LAI'] * P['F_DR']
    return [(t_h, amt_FR)], [(t_h, amt_DR)]

# ============================================================
# SIMULATION
# ============================================================
def simulate_cohort(coh_name, p):
    cfg          = COHORT_CONFIG[coh_name]
    ev_R         = build_wegovy_events(cfg['skip_block'])
    ev_FR, ev_DR = build_dwj_events(cfg['dwj_block'], cfg['dwj_dose'])
    ode          = build_ode(p, ev_R, ev_FR, ev_DR)

    t_end  = SIM_WEEKS * 7 * 24.0        # 4704 h
    t_eval = np.linspace(0, t_end, int(t_end) + 1)
    y0     = [0.0] * 7 + [p['BW0']]

    sol = solve_ivp(ode, [0, t_end], y0, t_eval=t_eval,
                    method='LSODA', rtol=1e-6, atol=1e-10)

    if not sol.success:
        st.error(f"ODE solver failed: {sol.message}")
        return None

    t_h    = sol.t
    t_wk   = t_h / (7.0 * 24.0)                 # hours → weeks
    C_ugL  = np.clip(sol.y[0], 0.0, None) / p['V']
    BW     = np.clip(sol.y[7], 0.0, None)
    BW_pct = (BW - p['BW0']) / p['BW0'] * 100.0
    GI     = np.clip(
        p['E0_AE'] + p['Emax_AE'] * C_ugL / (p['EC50_AE'] + C_ugL + 1e-15),
        0.0, 1.0) * 100.0

    return dict(t_h=t_h, t_wk=t_wk, C_ugL=C_ugL, BW_pct=BW_pct, GI=GI)

@st.cache_data(show_spinner=False)
def run_simulation(active_cohorts):
    return {coh: simulate_cohort(coh, P) for coh in active_cohorts}

def compute_pk_params(t_h, C_ugL):
    Cmax  = float(np.max(C_ugL))
    Tmax  = float(t_h[np.argmax(C_ugL)])
    AUC   = float(trapezoid(C_ugL, t_h))
    Clast = float(C_ugL[-1])
    return Cmax, Tmax, AUC, Clast

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sb-logo">💊 PK/PD Simulator</div>', unsafe_allow_html=True)
    st.markdown('<span style="font-size:0.75rem;color:#64748b">DWJ1691 + Wegovy · v1.0</span>',
                unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#2d4a6e;margin:10px 0">', unsafe_allow_html=True)

    st.markdown('<div class="sb-hdr">Cohort Selection</div>', unsafe_allow_html=True)
    sel = {coh: st.checkbox(coh, value=True, key=f"chk_{coh}")
           for coh in COHORT_COLORS}

    st.markdown('<hr style="border-color:#2d4a6e;margin:10px 0">', unsafe_allow_html=True)
    st.markdown('<div class="sb-hdr">DWJ1691 Dose (8× Wegovy)</div>', unsafe_allow_html=True)
    for pattern, dose, timing in [
        ("W-W-<b>T</b>-W-W", "8.0 mg",  "wk 9"),
        ("W-W-W-<b>T</b>-W", "13.6 mg", "wk 13"),
        ("W-W-W-W-<b>T</b>", "19.2 mg", "wk 17"),
    ]:
        st.markdown(
            f'<div class="dose-chip">{pattern} → <b>{dose}</b> at {timing}</div>',
            unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#2d4a6e;margin:10px 0">', unsafe_allow_html=True)
    st.markdown(
        '<span style="font-size:0.72rem;color:#64748b">'
        '📅 Sim: <b style="color:#93c5fd">28 weeks</b><br>'
        '👨‍🔬 Taeheon Kim, Ph.D.<br>'
        '🔬 Phoenix NLME · 2026-02-19</span>',
        unsafe_allow_html=True)

# ============================================================
# MAIN
# ============================================================
st.markdown("""
<div class="main-header">
  <div class="main-title">DWJ1691 + Wegovy &nbsp;
    <span class="badge">PK/PD/Safety</span>
    <span class="badge">Phoenix NLME</span>
    <span class="badge">28-Week</span>
  </div>
  <div class="main-sub">
    1-Cpt · Multiple Absorption (LAI fast+delayed + Wegovy SC)
    · Indirect Response BW · Simple Emax GI AE
  </div>
  <div class="design-strip">
    <div class="design-item"><div class="design-dot" style="background:#94a3b8"></div>
      <span>Wegovy: 0.25→0.5→1.0→1.7→2.4 mg q1w (4wk each)</span></div>
    <div class="design-item"><div class="design-dot" style="background:#2166ac"></div>
      <span>Cohort I: DWJ <b>8.0 mg</b> @ wk9</span></div>
    <div class="design-item"><div class="design-dot" style="background:#16a34a"></div>
      <span>Cohort II: DWJ <b>13.6 mg</b> @ wk13</span></div>
    <div class="design-item"><div class="design-dot" style="background:#dc2626"></div>
      <span>Cohort III: DWJ <b>19.2 mg</b> @ wk17</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

active = [c for c, v in sel.items() if v]
if not active:
    st.warning("코호트를 한 개 이상 선택해주세요.")
    st.stop()

with st.spinner("🔬 ODE 시뮬레이션 실행 중..."):
    results = run_simulation(tuple(active))

results = {k: v for k, v in results.items() if v is not None}
if not results:
    st.error("시뮬레이션 오류가 발생했습니다.")
    st.stop()

# ---- KPI ----
all_C  = np.concatenate([r['C_ugL']  for r in results.values()])
all_bw = np.concatenate([r['BW_pct'] for r in results.values()])
all_gi = np.concatenate([r['GI']     for r in results.values()])

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card kpi-blue">
      <div class="kpi-label">C<sub>max</sub> (all cohorts)</div>
      <div class="kpi-value cv-blue">{np.max(all_C):.1f}</div>
      <div class="kpi-unit">µg/L</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card kpi-red">
      <div class="kpi-label">DWJ1691 Doses</div>
      <div class="kpi-value cv-red" style="font-size:1.35rem">8 / 13.6 / 19.2</div>
      <div class="kpi-unit">mg · Cohort I / II / III</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card kpi-green">
      <div class="kpi-label">Max BW Loss</div>
      <div class="kpi-value cv-green">{np.min(all_bw):.1f}%</div>
      <div class="kpi-unit">from baseline (BW₀ = 100 kg)</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card kpi-orange">
      <div class="kpi-label">Peak GI AE Rate</div>
      <div class="kpi-value cv-orange">{np.max(all_gi):.1f}%</div>
      <div class="kpi-unit">adverse event</div></div>""", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ---- PK Profile ----
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="sec-hdr">📈 PK Profile — Plasma Concentration (µg/L)</div>',
            unsafe_allow_html=True)
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

fig_pk = go.Figure()
for coh, r in results.items():
    s = max(1, len(r['t_wk']) // 1000)
    fig_pk.add_trace(go.Scatter(
        x=r['t_wk'][::s], y=r['C_ugL'][::s], name=coh,
        line=dict(color=COHORT_COLORS[coh], width=2.5, dash=COHORT_DASH[coh]),
        hovertemplate=f"<b>{coh}</b><br>Week: %{{x:.1f}}<br>Conc: %{{y:.1f}} µg/L<extra></extra>"
    ))

vline_col = {
    "Cohort I (W-W-T-W-W)":   "#2166ac",
    "Cohort II (W-W-W-T-W)":  "#16a34a",
    "Cohort III (W-W-W-W-T)": "#dc2626",
}
for coh in active:
    cfg = COHORT_CONFIG[coh]
    if cfg['dwj_block'] is not None:
        t_v = float(cfg['dwj_block']) * 4.0
        fig_pk.add_vline(
            x=t_v, line_dash="dash",
            line_color=vline_col.get(coh, "#888"),
            line_width=1.2, opacity=0.5,
            annotation_text=f"DWJ {cfg['dwj_dose']:.1f}mg",
            annotation_position="top",
            annotation_font=dict(size=9, color=vline_col.get(coh, "#888"))
        )

fig_pk.update_layout(
    **CHART_BG, height=430,
    xaxis_title="Time (Week)",
    yaxis_title="Plasma concentration (µg/L)",
    legend=dict(
        bgcolor="rgba(255,255,255,0.95)", bordercolor="#e2e8f0", borderwidth=1,
        orientation="h", yanchor="bottom", y=1.01,
        xanchor="left", x=0, font=dict(size=11)
    )
)
st.plotly_chart(fig_pk, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---- BW + GI ----
col_bw, col_gi = st.columns(2)

with col_bw:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">⚖️ Body Weight Change (%BW from baseline)</div>',
                unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    fig_bw = go.Figure()
    for coh, r in results.items():
        s = max(1, len(r['t_wk']) // 1000)
        fig_bw.add_trace(go.Scatter(
            x=r['t_wk'][::s], y=r['BW_pct'][::s], name=coh,
            line=dict(color=COHORT_COLORS[coh], width=2.5, dash=COHORT_DASH[coh]),
            hovertemplate=f"<b>{coh}</b><br>Week: %{{x:.1f}}<br>ΔBW: %{{y:.2f}}%<extra></extra>"
        ))
    fig_bw.add_hline(y=0, line_dash="dot", line_color="#cbd5e1", line_width=1.2)
    fig_bw.update_layout(
        **CHART_BG, height=340,
        xaxis_title="Time (Week)",
        yaxis_title="ΔBW (%) from BW₀ = 100 kg",
        showlegend=False
    )
    st.plotly_chart(fig_bw, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_gi:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">⚠️ GI Adverse Event Rate (%)</div>',
                unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    fig_gi = go.Figure()
    for coh, r in results.items():
        s = max(1, len(r['t_wk']) // 1000)
        fig_gi.add_trace(go.Scatter(
            x=r['t_wk'][::s], y=r['GI'][::s], name=coh,
            line=dict(color=COHORT_COLORS[coh], width=2.5, dash=COHORT_DASH[coh]),
            hovertemplate=f"<b>{coh}</b><br>Week: %{{x:.1f}}<br>GI AE: %{{y:.1f}}%<extra></extra>"
        ))
    fig_gi.update_layout(
        **CHART_BG, height=340,
        xaxis_title="Time (Week)",
        yaxis_title="GI AE rate (%)",
        showlegend=False
    )
    fig_gi.update_yaxes(range=[0, 100])
    st.plotly_chart(fig_gi, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Summary Table ----
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="sec-hdr">📊 Cohort Summary — PK Parameters & PD/Safety Endpoints</div>',
    unsafe_allow_html=True)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

rows = []
for coh, r in results.items():
    cfg = COHORT_CONFIG[coh]
    Cmax, Tmax, AUC, Clast = compute_pk_params(r['t_h'], r['C_ugL'])
    rows.append({
        "Cohort":            coh,
        "DWJ Dose (mg)":    f"{cfg['dwj_dose']:.1f}" if cfg['dwj_dose'] > 0 else "—",
        "Cmax (µg/L)":      round(Cmax,  1),
        "Tmax (h)":         round(Tmax,  1),
        "AUClast (µg·h/L)": round(AUC,   0),
        "Clast (µg/L)":     round(Clast, 2),
        "Max ΔBW (%)":      round(float(np.min(r['BW_pct'])), 2),
        "Peak GI AE (%)":   round(float(np.max(r['GI'])),     1),
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True,
    column_config={
        "Cohort":            st.column_config.TextColumn(width="large"),
        "DWJ Dose (mg)":     st.column_config.TextColumn(width="small"),
        "Cmax (µg/L)":       st.column_config.NumberColumn(format="%.1f"),
        "Tmax (h)":          st.column_config.NumberColumn(format="%.1f"),
        "AUClast (µg·h/L)":  st.column_config.NumberColumn(format="%.0f"),
        "Clast (µg/L)":      st.column_config.NumberColumn(format="%.2f"),
        "Max ΔBW (%)":       st.column_config.NumberColumn(format="%.2f"),
        "Peak GI AE (%)":    st.column_config.NumberColumn(format="%.1f"),
    })

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download Summary CSV", csv,
                   file_name="pkpd_DWJ_28wk.csv", mime="text/csv")
st.markdown('</div>', unsafe_allow_html=True)

# ---- Model Info ----
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
with st.expander("📋 Model Parameters (Phoenix NLME fixef)"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**PK — 1-Cpt Multiple Absorption**")
        st.markdown(f"- V = {P['V']} L")
        st.markdown(f"- CL = {P['Cl']} L/h")
        st.markdown(f"- Ka (FR→A1) = {P['Ka']} h⁻¹")
        st.markdown(f"- ka_SC (R→A1) = {P['ka_SC']} h⁻¹")
        st.markdown(f"- F_SC = {P['F_SC']}  (Wegovy bioavail)")
        st.markdown(f"- Scale_LAI = {P['Scale_LAI']}")
        st.markdown(f"- F_DR = {P['F_DR']}  →  F_FR = {P['F_SC']-P['F_DR']:.3f}")
        st.markdown(f"- kdr = {P['kdr']} h⁻¹")
    with c2:
        st.markdown("**BW PD — Indirect Response**")
        st.markdown(f"- Imax = {P['Imax']}")
        st.markdown(f"- IC50 = {P['IC50']} µg/L")
        st.markdown(f"- Gamma = {P['Gamma']}")
        st.markdown(f"- kout = {P['kout']} h⁻¹")
        st.markdown(f"- BW₀ = {P['BW0']} kg (fixed)")
        st.markdown("- CB(t) = 100 − 6×(1−e^(−0.0001t))")
    with c3:
        st.markdown("**GI AE — Simple Emax**")
        st.markdown(f"- E₀ = {P['E0_AE']}")
        st.markdown(f"- Emax = {P['Emax_AE']}")
        st.markdown(f"- EC50 = {P['EC50_AE']} µg/L")
        st.markdown("---")
        st.markdown("**Unit Convention**")
        st.markdown("- Dose(mg) × 1000 → Amount(µg)")
        st.markdown("- C = Amount(µg) / V(L) = µg/L")
        st.markdown(f"- Wegovy R: dose×1000×{P['F_SC']}")
        st.markdown(f"- DWJ FR: dose×1000×{P['Scale_LAI']}×{P['F_SC']-P['F_DR']:.3f}")
        st.markdown(f"- DWJ DR: dose×1000×{P['Scale_LAI']}×{P['F_DR']}")

st.markdown("""
<div style='text-align:center;color:#94a3b8;font-size:0.72rem;
            padding:10px 0;margin-top:12px;background:#ffffff;
            border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,0.05)'>
  Phoenix NLME · Taeheon Kim, Ph.D. · 2026-02-19 · 28-week simulation
</div>
""", unsafe_allow_html=True)
