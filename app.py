import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="DWJ1691 + Wegovy PK/PD Simulator",
    page_icon="💊", layout="wide",
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
.sb-logo { font-size: 1.1rem; font-weight: 700; color: #ffffff !important; }
.sb-hdr {
    font-size: 0.62rem; font-weight: 700; color: #60a5fa !important;
    text-transform: uppercase; letter-spacing: 0.1em;
    border-bottom: 1px solid #2d4a6e; padding-bottom: 4px; margin: 14px 0 8px 0;
}
.dose-chip {
    display: inline-block; background: rgba(96,165,250,0.15);
    border: 1px solid rgba(96,165,250,0.3); border-radius: 6px;
    padding: 5px 10px; font-size: 0.78rem; color: #93c5fd !important;
    margin-bottom: 5px; width: 100%;
}
.dose-chip b { color: #60a5fa !important; }
.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2166ac 100%);
    border-radius: 14px; padding: 20px 28px; margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(33,102,172,0.25);
}
.main-title { font-size: 1.5rem; font-weight: 700; color: #ffffff; margin-bottom: 4px; }
.main-sub   { font-size: 0.82rem; color: #93c5fd; margin-bottom: 10px; }
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
.design-dot  { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.kpi-card {
    background: #ffffff; border-radius: 12px; padding: 18px 20px; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07); border-top: 3px solid transparent;
}
.kpi-blue   { border-top-color: #2166ac; } .kpi-red    { border-top-color: #dc2626; }
.kpi-green  { border-top-color: #16a34a; } .kpi-orange { border-top-color: #d97706; }
.kpi-label  { font-size: 0.65rem; color: #94a3b8; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.kpi-value  { font-size: 1.8rem; font-weight: 700; line-height: 1.0; }
.kpi-unit   { font-size: 0.68rem; color: #94a3b8; margin-top: 4px; }
.cv-blue  { color: #2166ac; } .cv-red    { color: #dc2626; }
.cv-green { color: #16a34a; } .cv-orange { color: #d97706; }
.chart-card {
    background: #ffffff; border-radius: 12px; padding: 18px 20px;
    margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.sec-hdr { font-size: 0.68rem; font-weight: 700; color: #2166ac;
    text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 2px; }
hr { border-color: #2d4a6e !important; }
.stButton > button {
    background: linear-gradient(135deg,#2166ac,#1d4ed8); color: white;
    border: none; border-radius: 8px; font-weight: 600; width: 100%; padding: 0.55rem;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──
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

HOURS_PER_WEEK = 168.0

# ============================================================
# MODEL PARAMETERS — Phoenix NLME fixef (Taeheon Kim, Ph.D.)
# Units: Amount(µg), V(L), C=µg/L, CL(L/h), IC50(µg/L)
# ============================================================
P = dict(
    V=12.4, Cl=0.0475,
    Ka=0.1026, ka_SC=0.0296, F_SC=0.9,
    Scale_LAI=0.2459, F_DR=0.429, kdr=0.02,
    BW0=100.0, Imax=0.21, IC50=55.0, Gamma=0.5, kout=0.00039,
    E0_AE=0.4833, Emax_AE=0.2867, EC50_AE=32.98,
)

# ============================================================
# DOSING SCHEDULE (엑셀 표 기준, µg, hours)
# ============================================================
WEGOVY_BLOCKS = [
    (250,  [0,   168,  336,  504 ]),
    (500,  [672, 840,  1008, 1176]),
    (1000, [1344,1512, 1680, 1848]),
    (1700, [2016,2184, 2352, 2520]),
    (2400, [2688,2856, 3024, 3192]),
]

DWJ_CONFIG = {
    "Reference":               None,
    "Cohort I (W-W-T-W-W)":   {"t": 1344.0, "dose": 8000.0,  "skip": 2},
    "Cohort II (W-W-W-T-W)":  {"t": 2016.0, "dose": 13600.0, "skip": 3},
    "Cohort III (W-W-W-W-T)": {"t": 2688.0, "dose": 19200.0, "skip": 4},
}

def build_dose_events(coh_name, p):
    cfg  = DWJ_CONFIG[coh_name]
    skip = cfg["skip"] if cfg else None

    # Wegovy → R depot (bioavail = F_SC)
    ev_R = []
    for bi, (dose_ug, times) in enumerate(WEGOVY_BLOCKS):
        if bi == skip:
            continue
        for t in times:
            ev_R.append((float(t), dose_ug * p['F_SC']))

    # DWJ1691 → FR/DR depot
    ev_FR, ev_DR = [], []
    if cfg:
        t_h  = cfg["t"]
        d_ug = cfg["dose"]
        F_FR = p['F_SC'] - p['F_DR']
        ev_FR = [(t_h, d_ug * p['Scale_LAI'] * F_FR)]
        ev_DR = [(t_h, d_ug * p['Scale_LAI'] * p['F_DR'])]

    return ev_R, ev_FR, ev_DR

# ============================================================
# ODE — Phoenix PML 직번역
# ============================================================
def make_ode(p):
    def ode(t, y):
        A1,FR,DR,DR1,DR2,DR3,R,BW = [max(v,0.0) for v in y]
        C    = A1 / p['V']
        dA1  = -(p['Cl']*C) + (p['ka_SC']*R) + (FR*p['Ka']) + (DR3*p['kdr'])
        dFR  = -(FR  * p['Ka'])
        dDR  = -(DR  * p['kdr'])
        dDR1 =  (DR  * p['kdr']) - (DR1 * p['kdr'])
        dDR2 =  (DR1 * p['kdr']) - (DR2 * p['kdr'])
        dDR3 =  (DR2 * p['kdr']) - (DR3 * p['kdr'])
        dR   = -(R * p['ka_SC'])
        E    = (p['Imax']*C**p['Gamma']) / (p['IC50']**p['Gamma'] + C**p['Gamma'] + 1e-15)
        CB   = 100.0 - 6.0*(1.0 - np.exp(-0.0001*t))
        dBW  = p['kout']*CB*(1.0-E) - p['kout']*BW
        return [dA1,dFR,dDR,dDR1,dDR2,dDR3,dR,dBW]
    return ode

# ============================================================
# SIMULATION — Event-based (Phoenix 방식과 동일)
# ============================================================
def simulate_cohort(coh_name, p, t_end=4032.0):
    ev_R, ev_FR, ev_DR = build_dose_events(coh_name, p)
    ode_fn = make_ode(p)

    # 투여 이벤트를 딕셔너리로 집계
    all_events = {}
    for t_h, amt in ev_R:
        all_events.setdefault(t_h, [0.0, 0.0, 0.0])
        all_events[t_h][0] += amt   # R depot
    for t_h, amt in ev_FR:
        all_events.setdefault(t_h, [0.0, 0.0, 0.0])
        all_events[t_h][1] += amt   # FR depot
    for t_h, amt in ev_DR:
        all_events.setdefault(t_h, [0.0, 0.0, 0.0])
        all_events[t_h][2] += amt   # DR depot

    breakpoints = sorted(set([0.0] + list(all_events.keys()) + [t_end]))

    y        = np.array([0.0]*7 + [p['BW0']])
    all_t    = []
    all_y    = []

    for i in range(len(breakpoints) - 1):
        t0 = breakpoints[i]
        t1 = breakpoints[i + 1]

        # 투여 이벤트 즉시 적용
        if t0 in all_events:
            R_add, FR_add, DR_add = all_events[t0]
            y[6] += R_add    # R
            y[1] += FR_add   # FR
            y[2] += DR_add   # DR

        n_pts  = max(2, int(t1 - t0) + 1)
        t_eval = np.linspace(t0, t1, n_pts)

        sol = solve_ivp(
            ode_fn, [t0, t1], y.copy(),
            t_eval=t_eval,
            method='LSODA',
            rtol=1e-7, atol=1e-10
        )
        if not sol.success:
            return None

        all_t.extend(sol.t[:-1].tolist())
        all_y.append(sol.y[:, :-1])
        y = sol.y[:, -1].copy()

    all_t.append(t_end)
    all_y.append(y.reshape(-1, 1))

    t_arr  = np.array(all_t)
    y_arr  = np.hstack(all_y)

    C_ugL   = np.clip(y_arr[0], 0.0, None) / p['V']
    BW      = np.clip(y_arr[7], 0.0, None)
    BW_pct  = (BW - p['BW0']) / p['BW0'] * 100.0
    GI_drug = np.clip(
        p['Emax_AE'] * C_ugL / (p['EC50_AE'] + C_ugL + 1e-15),
        0.0, 1.0)
    GI_total = np.clip(p['E0_AE'] + GI_drug, 0.0, 1.0)

    return {
        "t_h":       t_arr,
        "C_ugL":     C_ugL,
        "BW_pct":    BW_pct,
        "GI_total":  GI_total * 100.0,
        "GI_drug":   GI_drug  * 100.0,
    }

@st.cache_data(show_spinner=False)
def run_simulation(active_cohorts, _ver):
    return {coh: simulate_cohort(coh, P) for coh in active_cohorts}

def pk_params(t_h, C_ugL):
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
    st.markdown('<span style="font-size:0.75rem;color:#94a3b8">DWJ1691 + Wegovy · v2.1</span>',
                unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#2d4a6e;margin:10px 0">', unsafe_allow_html=True)

    st.markdown('<div class="sb-hdr">Cohort Selection</div>', unsafe_allow_html=True)
    sel = {coh: st.checkbox(coh, value=True, key=f"chk_{coh}")
           for coh in COHORT_COLORS}

    st.markdown('<hr style="border-color:#2d4a6e;margin:10px 0">', unsafe_allow_html=True)
    st.markdown('<div class="sb-hdr">DWJ1691 Dosing (8× Wegovy)</div>', unsafe_allow_html=True)
    for pat, dose, t_info in [
        ("W-W-<b>T</b>-W-W", "8,000 µg",  "h1344 (wk8)"),
        ("W-W-W-<b>T</b>-W", "13,600 µg", "h2016 (wk12)"),
        ("W-W-W-W-<b>T</b>", "19,200 µg", "h2688 (wk16)"),
    ]:
        st.markdown(
            f'<div class="dose-chip">{pat}<br>'
            f'<b>{dose}</b> @ {t_info}</div>',
            unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#2d4a6e;margin:10px 0">', unsafe_allow_html=True)
    st.markdown(
        '<span style="font-size:0.72rem;color:#94a3b8">'
        '📅 Sim: 4032h (dosing + washout)<br>'
        '✅ Phoenix NLME validated<br>'
        '👨‍🔬 Taeheon Kim, Ph.D.<br>'
        '🔬 2026-02-19</span>',
        unsafe_allow_html=True)

# ============================================================
# MAIN
# ============================================================
st.markdown("""
<div class="main-header">
  <div class="main-title">DWJ1691 + Wegovy &nbsp;
    <span class="badge">PK/PD/Safety</span>
    <span class="badge">Phoenix NLME</span>
    <span class="badge">Validated ✓</span>
  </div>
  <div class="main-sub">
    1-Cpt · Multiple Absorption (LAI fast+delayed + Wegovy SC)
    · Indirect Response BW · Simple Emax GI AE
  </div>
  <div class="design-strip">
    <div class="design-item"><div class="design-dot" style="background:#94a3b8"></div>
      <span>Wegovy: 250→500→1000→1700→2400 µg SC q1w (4doses each)</span></div>
    <div class="design-item"><div class="design-dot" style="background:#2166ac"></div>
      <span>Cohort I: DWJ <b>8,000 µg</b> @ h1344</span></div>
    <div class="design-item"><div class="design-dot" style="background:#16a34a"></div>
      <span>Cohort II: DWJ <b>13,600 µg</b> @ h2016</span></div>
    <div class="design-item"><div class="design-dot" style="background:#dc2626"></div>
      <span>Cohort III: DWJ <b>19,200 µg</b> @ h2688</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

active = [c for c, v in sel.items() if v]
if not active:
    st.warning("코호트를 한 개 이상 선택해주세요.")
    st.stop()

with st.spinner("🔬 ODE 시뮬레이션 실행 중 (Phoenix validated)..."):
    results = run_simulation(tuple(active), _ver="v2.1")

results = {k: v for k, v in results.items() if v is not None}
if not results:
    st.error("시뮬레이션 오류가 발생했습니다.")
    st.stop()

all_C  = np.concatenate([r['C_ugL']   for r in results.values()])
all_bw = np.concatenate([r['BW_pct']  for r in results.values()])
all_gi = np.concatenate([r['GI_total']for r in results.values()])

# ── KPI ──
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card kpi-blue">
      <div class="kpi-label">C<sub>max</sub> (all cohorts)</div>
      <div class="kpi-value cv-blue">{np.max(all_C):.1f}</div>
      <div class="kpi-unit">µg/L</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card kpi-red">
      <div class="kpi-label">DWJ1691 Doses</div>
      <div class="kpi-value cv-red" style="font-size:1.1rem">8k/13.6k/19.2k</div>
      <div class="kpi-unit">µg · Cohort I/II/III</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card kpi-green">
      <div class="kpi-label">Max BW Loss</div>
      <div class="kpi-value cv-green">{np.min(all_bw):.1f}%</div>
      <div class="kpi-unit">from BW₀ = 100 kg</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card kpi-orange">
      <div class="kpi-label">Peak GI AE (Total)</div>
      <div class="kpi-value cv-orange">{np.max(all_gi):.1f}%</div>
      <div class="kpi-unit">adverse event rate</div></div>""", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── PK Chart ──
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="sec-hdr">📈 PK Profile — Plasma Concentration (µg/L)</div>',
            unsafe_allow_html=True)
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
fig_pk = go.Figure()
for coh, r in results.items():
    t_wk = r['t_h'] / HOURS_PER_WEEK
    s = max(1, len(t_wk) // 2000)
    fig_pk.add_trace(go.Scatter(
        x=t_wk[::s], y=r['C_ugL'][::s], name=coh,
        line=dict(color=COHORT_COLORS[coh], width=2, dash=COHORT_DASH[coh]),
        hovertemplate=f"<b>{coh}</b><br>Week:%{{x:.1f}}<br>Conc:%{{y:.1f}} µg/L<extra></extra>"
    ))
vline_info = {
    "Cohort I (W-W-T-W-W)":   (1344/HOURS_PER_WEEK, "#2166ac", "8,000µg"),
    "Cohort II (W-W-W-T-W)":  (2016/HOURS_PER_WEEK, "#16a34a", "13,600µg"),
    "Cohort III (W-W-W-W-T)": (2688/HOURS_PER_WEEK, "#dc2626", "19,200µg"),
}
for coh in active:
    if coh in vline_info:
        t_v, col, lbl = vline_info[coh]
        fig_pk.add_vline(x=t_v, line_dash="dash", line_color=col,
                         line_width=1, opacity=0.4,
                         annotation_text=f"DWJ {lbl}",
                         annotation_position="top",
                         annotation_font=dict(size=9, color=col))
fig_pk.update_layout(
    **CHART_BG, height=430,
    xaxis_title="Time (Week)",
    yaxis_title="Plasma concentration (µg/L)",
    legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#e2e8f0",
                borderwidth=1, orientation="h", yanchor="bottom", y=1.01,
                xanchor="left", x=0, font=dict(size=11))
)
st.plotly_chart(fig_pk, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── BW + GI ──
col_bw, col_gi = st.columns(2)
with col_bw:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">⚖️ Body Weight Change (%BW from baseline)</div>',
                unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    fig_bw = go.Figure()
    for coh, r in results.items():
        t_wk = r['t_h'] / HOURS_PER_WEEK
        s = max(1, len(t_wk) // 2000)
        fig_bw.add_trace(go.Scatter(
            x=t_wk[::s], y=r['BW_pct'][::s], name=coh,
            line=dict(color=COHORT_COLORS[coh], width=2, dash=COHORT_DASH[coh]),
            hovertemplate=f"<b>{coh}</b><br>Week:%{{x:.1f}}<br>ΔBW:%{{y:.2f}}%<extra></extra>"
        ))
    fig_bw.add_hline(y=0, line_dash="dot", line_color="#cbd5e1", line_width=1)
    fig_bw.update_layout(**CHART_BG, height=340,
        xaxis_title="Time (Week)", yaxis_title="ΔBW (%) from BW₀=100kg",
        showlegend=False)
    st.plotly_chart(fig_bw, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_gi:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">⚠️ GI Adverse Event Rate — Total (%)</div>',
                unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    fig_gi = go.Figure()
    for coh, r in results.items():
        t_wk = r['t_h'] / HOURS_PER_WEEK
        s = max(1, len(t_wk) // 2000)
        fig_gi.add_trace(go.Scatter(
            x=t_wk[::s], y=r['GI_total'][::s], name=coh,
            line=dict(color=COHORT_COLORS[coh], width=2, dash=COHORT_DASH[coh]),
            hovertemplate=f"<b>{coh}</b><br>Week:%{{x:.1f}}<br>GI Total:%{{y:.1f}}%<extra></extra>"
        ))
    fig_gi.update_layout(**CHART_BG, height=340,
        xaxis_title="Time (Week)", yaxis_title="GI AE Total rate (%)",
        showlegend=False)
    fig_gi.update_yaxes(range=[0, 100])
    st.plotly_chart(fig_gi, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Summary Table ──
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="sec-hdr">📊 Cohort Summary — PK / PD / Safety</div>',
            unsafe_allow_html=True)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

DWJ_LABELS = {
    "Reference":               "—",
    "Cohort I (W-W-T-W-W)":   "8,000",
    "Cohort II (W-W-W-T-W)":  "13,600",
    "Cohort III (W-W-W-W-T)": "19,200",
}
rows = []
for coh, r in results.items():
    Cmax, Tmax, AUC, Clast = pk_params(r['t_h'], r['C_ugL'])
    rows.append({
        "Cohort":             coh,
        "DWJ (µg)":          DWJ_LABELS[coh],
        "Cmax (µg/L)":       round(Cmax,  1),
        "Tmax (h)":          round(Tmax,  1),
        "AUClast (µg·h/L)":  round(AUC,   0),
        "Clast (µg/L)":      round(Clast, 2),
        "Max ΔBW (%)":       round(float(np.min(r['BW_pct'])),   2),
        "Peak GI Total (%)": round(float(np.max(r['GI_total'])), 1),
        "Peak GI Drug (%)":  round(float(np.max(r['GI_drug'])),  1),
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True,
    column_config={
        "Cohort":             st.column_config.TextColumn(width="large"),
        "DWJ (µg)":           st.column_config.TextColumn(width="small"),
        "Cmax (µg/L)":        st.column_config.NumberColumn(format="%.1f"),
        "Tmax (h)":           st.column_config.NumberColumn(format="%.1f"),
        "AUClast (µg·h/L)":   st.column_config.NumberColumn(format="%.0f"),
        "Clast (µg/L)":       st.column_config.NumberColumn(format="%.2f"),
        "Max ΔBW (%)":        st.column_config.NumberColumn(format="%.2f"),
        "Peak GI Total (%)":  st.column_config.NumberColumn(format="%.1f"),
        "Peak GI Drug (%)":   st.column_config.NumberColumn(format="%.1f"),
    })

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download Summary CSV", csv,
                   file_name="pkpd_DWJ_validated.csv", mime="text/csv")
st.markdown('</div>', unsafe_allow_html=True)

# ── Model Info ──
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
with st.expander("📋 Model Parameters (Phoenix NLME fixef) — Validated"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**PK — 1-Cpt Multiple Absorption**")
        st.markdown(f"- V = {P['V']} L")
        st.markdown(f"- CL = {P['Cl']} L/h")
        st.markdown(f"- Ka (FR→A1) = {P['Ka']} h⁻¹")
        st.markdown(f"- ka_SC (R→A1) = {P['ka_SC']} h⁻¹")
        st.markdown(f"- F_SC = {P['F_SC']}")
        st.markdown(f"- Scale_LAI = {P['Scale_LAI']}")
        st.markdown(f"- F_DR = {P['F_DR']} → F_FR = {P['F_SC']-P['F_DR']:.3f}")
        st.markdown(f"- kdr = {P['kdr']} h⁻¹")
    with c2:
        st.markdown("**BW PD — Indirect Response**")
        st.markdown(f"- Imax = {P['Imax']}, IC50 = {P['IC50']} µg/L")
        st.markdown(f"- Gamma = {P['Gamma']}, kout = {P['kout']} h⁻¹")
        st.markdown(f"- CB(t) = 100−6·(1−e^(−0.0001t))")
        st.markdown(f"- BW₀ = {P['BW0']} kg (fixed)")
        st.markdown("**GI AE — Simple Emax**")
        st.markdown(f"- E₀={P['E0_AE']}, Emax={P['Emax_AE']}")
        st.markdown(f"- EC50={P['EC50_AE']} µg/L")
    with c3:
        st.markdown("**✅ Phoenix Validation**")
        st.markdown("- t=1h:    0.5282 µg/L ✓")
        st.markdown("- t=80h:  13.389  µg/L ✓")
        st.markdown("- t=168h: 10.807  µg/L ✓")
        st.markdown("- t=240h: 21.639  µg/L ✓")
        st.markdown("- t=1344h: 43.97  µg/L ✓")
        st.markdown("- t=2688h: 150.72 µg/L ✓")
        st.markdown("---")
        st.markdown("**Dosing (µg)**")
        st.markdown("- Wegovy: 250→500→1000→1700→2400")
        st.markdown("- Cohort I: 8,000 @ h1344")
        st.markdown("- Cohort II: 13,600 @ h2016")
        st.markdown("- Cohort III: 19,200 @ h2688")

st.markdown("""
<div style='text-align:center;color:#94a3b8;font-size:0.72rem;
            padding:10px 0;margin-top:12px;background:#ffffff;
            border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,0.05)'>
  ✅ Phoenix NLME Validated · Taeheon Kim, Ph.D. · 2026-02-19 · v2.1
</div>
""", unsafe_allow_html=True)
