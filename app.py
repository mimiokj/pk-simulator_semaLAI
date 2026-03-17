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
    padding: 4px 10px; font-size: 0.78rem; color: #93c5fd !important;
    margin-bottom: 5px; width: 100%;
}
.dose-chip b { color: #60a5fa !important; }
.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2166ac 100%);
    border-radius: 14px; padding: 20px 28px; margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(33,102,172,0.25);
}
.main-title { font-size: 1.5rem; font-weight: 700; color: #ffffff; margin-bottom: 4px; }
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
.kpi-blue   { border-top-color: #2166ac; }
.kpi-red    { border-top-color: #dc2626; }
.kpi-green  { border-top-color: #16a34a; }
.kpi-orange { border-top-color: #d97706; }
.kpi-label { font-size: 0.65rem; color: #94a3b8; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.kpi-value { font-size: 1.8rem; font-weight: 700; line-height: 1.0; }
.kpi-unit  { font-size: 0.68rem; color: #94a3b8; margin-top: 4px; }
.cv-blue { color: #2166ac; } .cv-red    { color: #dc2626; }
.cv-green{ color: #16a34a; } .cv-orange { color: #d97706; }
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

SIM_WEEKS = 28
HOURS_PER_WEEK = 168.0  # 7 × 24

# ============================================================
# MODEL PARAMETERS — Phoenix NLME fixef (Taeheon Kim, Ph.D.)
# ============================================================
# 단위 체계 (엑셀 표 기준):
#   Amount : µg  (dose 표의 숫자 그대로)
#   V      : L
#   C      : µg/L  (= Amount / V)
#   CL     : L/h
#   IC50   : µg/L
# ============================================================
P = dict(
    V         = 12.4,
    Cl        = 0.0475,
    Ka        = 0.1026,    # h⁻¹  FR depot → A1 (DWJ LAI fast)
    ka_SC     = 0.0296,    # h⁻¹  R depot  → A1 (Wegovy SC)
    F_SC      = 0.9,
    Scale_LAI = 0.2459,
    F_DR      = 0.429,     # F_FR = 0.9 - 0.429 = 0.471
    kdr       = 0.02,
    BW0       = 100.0,
    Imax      = 0.21,
    IC50      = 55.0,      # µg/L
    Gamma     = 0.5,
    kout      = 0.00039,
    E0_AE     = 0.4833,
    Emax_AE   = 0.2867,
    EC50_AE   = 32.98,     # µg/L
)

# ============================================================
# DOSING SCHEDULE — 엑셀 표 기준 (단위: µg, hours)
# ============================================================
# Wegovy 용량 단계 (µg)
WEGOVY_DOSE_UG = [250, 500, 1000, 1700, 2400]

def build_all_events(cohort_name):
    """
    엑셀 표를 그대로 구현:
    - Wegovy SC: 168h 간격, 각 단계 4회씩 (0~3192h)
    - DWJ1691 LAI: 코호트별 1회

    반환:
      ev_R  : Wegovy → R depot [(t_h, amt_ug), ...]
      ev_FR : DWJ fast → FR depot
      ev_DR : DWJ delayed → DR depot
    """

    # ── Wegovy 투여 스케줄 (Reference 기준 전체) ──
    # 블록 0: 250µg × 4회 (0, 168, 336, 504h)
    # 블록 1: 500µg × 4회 (672, 840, 1008, 1176h)
    # 블록 2: 1000µg × 4회 (1344, 1512, 1680, 1848h)
    # 블록 3: 1700µg × 4회 (2016, 2184, 2352, 2520h)
    # 블록 4: 2400µg × 4회 (2688, 2856, 3024, 3192h)

    # DWJ1691 투여 시점 및 용량
    dwj_schedule = {
        "Reference":               None,
        "Cohort I (W-W-T-W-W)":   {"t_h": 1344.0, "dose_ug": 8000.0},
        "Cohort II (W-W-W-T-W)":  {"t_h": 2016.0, "dose_ug": 13600.0},
        "Cohort III (W-W-W-W-T)": {"t_h": 2688.0, "dose_ug": 19200.0},
    }

    # Wegovy 스킵할 블록 결정
    # Cohort I: 블록 2 (1344~1848h) Wegovy 없음 → DWJ로 대체
    # Cohort II: 블록 3 (2016~2520h) Wegovy 없음
    # Cohort III: 블록 4 (2688~3192h) Wegovy 없음
    skip_block = {
        "Reference":               None,
        "Cohort I (W-W-T-W-W)":   2,
        "Cohort II (W-W-W-T-W)":  3,
        "Cohort III (W-W-W-W-T)": 4,
    }[cohort_name]

    # ── Wegovy R depot 이벤트 구성 ──
    ev_R = []
    for blk, dose_ug in enumerate(WEGOVY_DOSE_UG):
        if blk == skip_block:
            continue   # 해당 블록은 DWJ1691으로 대체
        for rep in range(4):
            t_h = (blk * 4 + rep) * HOURS_PER_WEEK
            # bioavail=F_SC 적용
            ev_R.append((t_h, dose_ug * P['F_SC']))

    # ── DWJ1691 FR/DR depot 이벤트 구성 ──
    ev_FR, ev_DR = [], []
    dwj = dwj_schedule[cohort_name]
    if dwj is not None:
        t_h    = dwj['t_h']
        d_ug   = dwj['dose_ug']
        F_FR   = P['F_SC'] - P['F_DR']      # 0.471
        ev_FR  = [(t_h, d_ug * P['Scale_LAI'] * F_FR)]
        ev_DR  = [(t_h, d_ug * P['Scale_LAI'] * P['F_DR'])]

    return ev_R, ev_FR, ev_DR

# ============================================================
# ODE — Phoenix PML 직번역
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

        dA1  = -(p['Cl']*C) + (p['ka_SC']*R) + (FR*p['Ka']) + (DR3*p['kdr'])
        dFR  = -(FR*p['Ka'])   + bolus(ev_FR, t)
        dDR  = -(DR*p['kdr'])  + bolus(ev_DR, t)
        dDR1 =  (DR*p['kdr'])  - (DR1*p['kdr'])
        dDR2 =  (DR1*p['kdr']) - (DR2*p['kdr'])
        dDR3 =  (DR2*p['kdr']) - (DR3*p['kdr'])
        dR   = -(R*p['ka_SC']) + bolus(ev_R, t)

        E   = (p['Imax']*C**p['Gamma']) / (p['IC50']**p['Gamma'] + C**p['Gamma'] + 1e-15)
        CB  = 100.0 - 6.0*(1.0 - np.exp(-0.0001*t))
        dBW = p['kout']*CB*(1.0-E) - p['kout']*BW

        return [dA1, dFR, dDR, dDR1, dDR2, dDR3, dR, dBW]
    return ode

# ============================================================
# SIMULATION
# ============================================================
def simulate_cohort(coh_name, p):
    ev_R, ev_FR, ev_DR = build_all_events(coh_name)
    ode   = build_ode(p, ev_R, ev_FR, ev_DR)

    t_end  = SIM_WEEKS * HOURS_PER_WEEK      # 4704 h
    # 0.25h 해상도 → weekly oscillation 정확 포착
    n_pts  = int(t_end / 0.25) + 1
    t_eval = np.linspace(0, t_end, n_pts)
    y0     = [0.0]*7 + [p['BW0']]

    sol = solve_ivp(ode, [0, t_end], y0, t_eval=t_eval,
                    method='LSODA', rtol=1e-7, atol=1e-10)
    if not sol.success:
        return None

    t_h    = sol.t
    C_ugL  = np.clip(sol.y[0], 0.0, None) / p['V']
    BW     = np.clip(sol.y[7], 0.0, None)
    BW_pct = (BW - p['BW0']) / p['BW0'] * 100.0
    GI     = np.clip(
        p['E0_AE'] + p['Emax_AE']*C_ugL / (p['EC50_AE'] + C_ugL + 1e-15),
        0.0, 1.0) * 100.0

    return {"t_h": t_h, "C_ugL": C_ugL, "BW_pct": BW_pct, "GI": GI}

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
    st.markdown('<span style="font-size:0.75rem;color:#64748b">DWJ1691 + Wegovy · v1.3</span>',
                unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#2d4a6e;margin:10px 0">', unsafe_allow_html=True)

    st.markdown('<div class="sb-hdr">Cohort Selection</div>', unsafe_allow_html=True)
    sel = {coh: st.checkbox(coh, value=True, key=f"chk_{coh}")
           for coh in COHORT_COLORS}

    st.markdown('<hr style="border-color:#2d4a6e;margin:10px 0">', unsafe_allow_html=True)
    st.markdown('<div class="sb-hdr">DWJ1691 Dose (8× Wegovy)</div>', unsafe_allow_html=True)
    for pattern, dose, timing in [
        ("W-W-<b>T</b>-W-W", "8,000 µg",  "h 1344 (wk 8)"),
        ("W-W-W-<b>T</b>-W", "13,600 µg", "h 2016 (wk 12)"),
        ("W-W-W-W-<b>T</b>", "19,200 µg", "h 2688 (wk 16)"),
    ]:
        st.markdown(
            f'<div class="dose-chip">{pattern} → <b>{dose}</b><br>'
            f'<span style="font-size:0.7rem;opacity:0.8">at {timing}</span></div>',
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
      <span>Wegovy: 250→500→1000→1700→2400 µg q1w (4doses each)</span></div>
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

with st.spinner("🔬 ODE 시뮬레이션 실행 중..."):
    results = run_simulation(tuple(active), _ver="v1.3")

results = {k: v for k, v in results.items() if v is not None}
if not results:
    st.error("시뮬레이션 오류가 발생했습니다.")
    st.stop()

all_C  = np.concatenate([r['C_ugL']  for r in results.values()])
all_bw = np.concatenate([r['BW_pct'] for r in results.values()])
all_gi = np.concatenate([r['GI']     for r in results.values()])

# ---- KPI ----
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card kpi-blue">
      <div class="kpi-label">C<sub>max</sub> (all cohorts)</div>
      <div class="kpi-value cv-blue">{np.max(all_C):.1f}</div>
      <div class="kpi-unit">µg/L</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card kpi-red">
      <div class="kpi-label">DWJ1691 Doses</div>
      <div class="kpi-value cv-red" style="font-size:1.2rem">8k / 13.6k / 19.2k</div>
      <div class="kpi-unit">µg · Cohort I / II / III</div></div>""", unsafe_allow_html=True)
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

# ---- PK Chart ----
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
        hovertemplate=f"<b>{coh}</b><br>Week: %{{x:.2f}}<br>Conc: %{{y:.1f}} µg/L<extra></extra>"
    ))

vline_info = {
    "Cohort I (W-W-T-W-W)":   (1344.0/HOURS_PER_WEEK, "#2166ac", "8,000µg"),
    "Cohort II (W-W-W-T-W)":  (2016.0/HOURS_PER_WEEK, "#16a34a", "13,600µg"),
    "Cohort III (W-W-W-W-T)": (2688.0/HOURS_PER_WEEK, "#dc2626", "19,200µg"),
}
for coh in active:
    if coh in vline_info:
        t_v, col, lbl = vline_info[coh]
        fig_pk.add_vline(
            x=t_v, line_dash="dash", line_color=col,
            line_width=1, opacity=0.4,
            annotation_text=f"DWJ {lbl}",
            annotation_position="top",
            annotation_font=dict(size=9, color=col)
        )

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

# ---- BW + GI ----
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
            hovertemplate=f"<b>{coh}</b><br>Week: %{{x:.1f}}<br>ΔBW: %{{y:.2f}}%<extra></extra>"
        ))
    fig_bw.add_hline(y=0, line_dash="dot", line_color="#cbd5e1", line_width=1)
    fig_bw.update_layout(**CHART_BG, height=340,
        xaxis_title="Time (Week)", yaxis_title="ΔBW (%) from BW₀=100 kg",
        showlegend=False)
    st.plotly_chart(fig_bw, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_gi:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">⚠️ GI Adverse Event Rate (%)</div>',
                unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    fig_gi = go.Figure()
    for coh, r in results.items():
        t_wk = r['t_h'] / HOURS_PER_WEEK
        s = max(1, len(t_wk) // 2000)
        fig_gi.add_trace(go.Scatter(
            x=t_wk[::s], y=r['GI'][::s], name=coh,
            line=dict(color=COHORT_COLORS[coh], width=2, dash=COHORT_DASH[coh]),
            hovertemplate=f"<b>{coh}</b><br>Week: %{{x:.1f}}<br>GI AE: %{{y:.1f}}%<extra></extra>"
        ))
    fig_gi.update_layout(**CHART_BG, height=340,
        xaxis_title="Time (Week)", yaxis_title="GI AE rate (%)",
        showlegend=False)
    fig_gi.update_yaxes(range=[0, 100])
    st.plotly_chart(fig_gi, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Summary Table ----
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="sec-hdr">📊 Cohort Summary — PK Parameters & PD/Safety Endpoints</div>',
            unsafe_allow_html=True)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

dwj_doses_display = {
    "Reference":               "—",
    "Cohort I (W-W-T-W-W)":   "8,000",
    "Cohort II (W-W-W-T-W)":  "13,600",
    "Cohort III (W-W-W-W-T)": "19,200",
}

rows = []
for coh, r in results.items():
    Cmax, Tmax, AUC, Clast = pk_params(r['t_h'], r['C_ugL'])
    rows.append({
        "Cohort":            coh,
        "DWJ Dose (µg)":    dwj_doses_display.get(coh, "—"),
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
        "DWJ Dose (µg)":     st.column_config.TextColumn(width="small"),
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
with st.expander("📋 Model Parameters & Dosing Schedule"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**PK — 1-Cpt Multiple Absorption**")
        st.markdown(f"- V = {P['V']} L")
        st.markdown(f"- CL = {P['Cl']} L/h")
        st.markdown(f"- Ka (FR→A1) = {P['Ka']} h⁻¹")
        st.markdown(f"- ka_SC (R→A1) = {P['ka_SC']} h⁻¹")
        st.markdown(f"- F_SC = {P['F_SC']}")
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
        st.markdown("**GI AE — Simple Emax**")
        st.markdown(f"- E₀={P['E0_AE']}, Emax={P['Emax_AE']}, EC50={P['EC50_AE']} µg/L")
    with c3:
        st.markdown("**Dosing Schedule (엑셀 기준)**")
        st.markdown("*Wegovy SC q1w (µg):*")
        st.markdown("- 250 × 4회 (h 0~504)")
        st.markdown("- 500 × 4회 (h 672~1176)")
        st.markdown("- 1000 × 4회 (h 1344~1848)")
        st.markdown("- 1700 × 4회 (h 2016~2520)")
        st.markdown("- 2400 × 4회 (h 2688~3192)")
        st.markdown("*DWJ1691 (single SC):*")
        st.markdown("- Cohort I: **8,000 µg** @ h1344")
        st.markdown("- Cohort II: **13,600 µg** @ h2016")
        st.markdown("- Cohort III: **19,200 µg** @ h2688")

st.markdown("""
<div style='text-align:center;color:#94a3b8;font-size:0.72rem;
            padding:10px 0;margin-top:12px;background:#ffffff;
            border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,0.05)'>
  Phoenix NLME · Taeheon Kim, Ph.D. · 2026-02-19 · 28-week simulation
</div>
""", unsafe_allow_html=True)
