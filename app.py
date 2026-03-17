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

# ── Plotly theme ──────────────────────────────────────────────
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

# ── Constants ─────────────────────────────────────────────────
SIM_WEEKS      = 28
HOURS_PER_WEEK = 168.0   # 7 × 24

# ============================================================
# MODEL PARAMETERS — Phoenix NLME fixef (Taeheon Kim, Ph.D.)
# ============================================================
# Unit convention:
#   Amount : µg   (dose 표의 숫자 그대로)
#   V      : L
#   C      = A1 / V  →  µg/L
#   CL     : L/h
#   IC50, EC50_AE : µg/L
# ============================================================
P = dict(
    V         = 12.4,      # L
    Cl        = 0.0475,    # L/h
    Ka        = 0.1026,    # h⁻¹   FR depot → A1  (DWJ1691 fast-release)
    ka_SC     = 0.0296,    # h⁻¹   R depot  → A1  (Wegovy SC)
    F_SC      = 0.9,       # –     Wegovy bioavailability
    Scale_LAI = 0.2459,    # –     DWJ1691 LAI overall scaling
    F_DR      = 0.429,     # –     delayed fraction  (F_FR = 0.9-0.429 = 0.471)
    kdr       = 0.02,      # h⁻¹   3-transit delayed rate
    BW0       = 100.0,     # kg
    Imax      = 0.21,
    IC50      = 55.0,      # µg/L
    Gamma     = 0.5,
    kout      = 0.00039,   # h⁻¹
    E0_AE     = 0.4833,
    Emax_AE   = 0.2867,
    EC50_AE   = 32.98,     # µg/L
)

# ============================================================
# DOSING SCHEDULE — 엑셀 표 기준 (단위: µg, hours)
# ============================================================
#
#  ▶ Wegovy (R depot):
#    블록0: 250µg × 4 → h 0, 168, 336, 504
#    블록1: 500µg × 4 → h 672, 840, 1008, 1176
#    블록2: 1000µg× 4 → h 1344, 1512, 1680, 1848
#    블록3: 1700µg× 4 → h 2016, 2184, 2352, 2520
#    블록4: 2400µg× 4 → h 2688, 2856, 3024, 3192
#
#  ▶ DWJ1691 (FR/DR depot, R은 0):
#    Cohort I  : 8000µg  @ h1344  (블록2 대체)
#    Cohort II : 13600µg @ h2016  (블록3 대체)
#    Cohort III: 19200µg @ h2688  (블록4 대체)
#
#  ※ 동일 시간 블록에서 Wegovy와 DWJ1691은 상호 배타적
#     (R에 amount가 있으면 FR/DR은 0, 그 반대도 마찬가지)
# ============================================================

# Wegovy 블록별 (dose_ug, [t_h_list])
WEGOVY_BLOCKS = [
    (250,  [0,   168,  336,  504 ]),   # 블록 0
    (500,  [672, 840,  1008, 1176]),   # 블록 1
    (1000, [1344,1512, 1680, 1848]),   # 블록 2
    (1700, [2016,2184, 2352, 2520]),   # 블록 3
    (2400, [2688,2856, 3024, 3192]),   # 블록 4
]

# DWJ1691 코호트별 설정
DWJ_INFO = {
    "Reference":               None,
    "Cohort I (W-W-T-W-W)":   {"t_h": 1344.0, "dose_ug": 8000.0,  "skip_blk": 2},
    "Cohort II (W-W-W-T-W)":  {"t_h": 2016.0, "dose_ug": 13600.0, "skip_blk": 3},
    "Cohort III (W-W-W-W-T)": {"t_h": 2688.0, "dose_ug": 19200.0, "skip_blk": 4},
}

def build_dose_events(cohort_name, p):
    """
    엑셀 dosing schedule → ODE 이벤트 리스트 변환

    반환:
      ev_R  : Wegovy → R depot  [(t_h, amt_ug), ...]
      ev_FR : DWJ    → FR depot [(t_h, amt_ug)]  or []
      ev_DR : DWJ    → DR depot [(t_h, amt_ug)]  or []

    규칙:
      · R depot  : dosepoint(R,  bioavail = F_SC)
                   → 주입량 = dose_ug × F_SC
      · FR depot : dosepoint(FR, bioavail = Scale_LAI × F_FR)
                   → 주입량 = dose_ug × Scale_LAI × F_FR
      · DR depot : dosepoint(DR, bioavail = Scale_LAI × F_DR)
                   → 주입량 = dose_ug × Scale_LAI × F_DR
      · 동일 블록에서 R과 FR/DR은 상호 배타적
    """
    dwj = DWJ_INFO[cohort_name]
    skip_blk = dwj["skip_blk"] if dwj else None

    # ── Wegovy R depot 이벤트 ──
    ev_R = []
    for blk_idx, (dose_ug, times) in enumerate(WEGOVY_BLOCKS):
        if blk_idx == skip_blk:
            # 이 블록은 DWJ1691으로 대체 → R에 투여 없음
            continue
        amt_R = dose_ug * p['F_SC']   # bioavail 적용
        for t_h in times:
            ev_R.append((t_h, amt_R))

    # ── DWJ1691 FR/DR depot 이벤트 ──
    ev_FR, ev_DR = [], []
    if dwj is not None:
        t_h    = dwj["t_h"]
        d_ug   = dwj["dose_ug"]
        F_FR   = p['F_SC'] - p['F_DR']          # 0.471
        amt_FR = d_ug * p['Scale_LAI'] * F_FR   # µg into FR
        amt_DR = d_ug * p['Scale_LAI'] * p['F_DR']  # µg into DR
        ev_FR  = [(t_h, amt_FR)]
        ev_DR  = [(t_h, amt_DR)]

    return ev_R, ev_FR, ev_DR

# ============================================================
# ODE — Phoenix PML deriv 블록 직번역
# ============================================================
# State: [A1, FR, DR, DR1, DR2, DR3, R, BW]
#
# PML:
#   deriv(A1)  = -(Cl×C) + (ka_SC×R) + (FR×Ka) + (DR3×kdr)
#   deriv(FR)  = -(FR×Ka)
#   deriv(DR)  = -(DR×kdr)
#   deriv(DR1) = (DR×kdr)  - (DR1×kdr)
#   deriv(DR2) = (DR1×kdr) - (DR2×kdr)
#   deriv(DR3) = (DR2×kdr) - (DR3×kdr)
#   deriv(R)   = -(R×ka_SC)
#
# PD:
#   E(C) = Imax×C^γ / (IC50^γ + C^γ)
#   CB(t) = 100 - 6×(1 - exp(-0.0001t))
#   kin(t) = kout × CB(t)
#   dBW/dt = kin(t)×(1-E(C)) - kout×BW
#
# GI AE:
#   AE_drug = Emax_AE×C / (EC50_AE + C)
#   AE_Total = E0_AE + AE_drug
# ============================================================
def build_ode(p, ev_R, ev_FR, ev_DR):
    def bolus(events, t, dur=0.5):
        """볼러스를 짧은 infusion으로 근사 (수치 안정성)"""
        val = 0.0
        for (td, amt) in events:
            if td <= t < td + dur:
                val += amt / dur
        return val

    def ode(t, y):
        A1, FR, DR, DR1, DR2, DR3, R, BW = [max(v, 0.0) for v in y]

        C = A1 / p['V']   # µg/L

        # ── PK ──
        dA1  = -(p['Cl'] * C) \
               + (p['ka_SC'] * R) \
               + (FR * p['Ka']) \
               + (DR3 * p['kdr'])
        dFR  = -(FR  * p['Ka'])   + bolus(ev_FR, t)
        dDR  = -(DR  * p['kdr']) + bolus(ev_DR, t)
        dDR1 =  (DR  * p['kdr']) - (DR1 * p['kdr'])
        dDR2 =  (DR1 * p['kdr']) - (DR2 * p['kdr'])
        dDR3 =  (DR2 * p['kdr']) - (DR3 * p['kdr'])
        dR   = -(R * p['ka_SC']) + bolus(ev_R, t)

        # ── BW PD: E(C) = Imax×C^γ / (IC50^γ + C^γ) ──
        E   = (p['Imax'] * C**p['Gamma']) / \
              (p['IC50']**p['Gamma'] + C**p['Gamma'] + 1e-15)
        CB  = 100.0 - 6.0 * (1.0 - np.exp(-0.0001 * t))
        kin = p['kout'] * CB
        dBW = kin * (1.0 - E) - p['kout'] * BW

        return [dA1, dFR, dDR, dDR1, dDR2, dDR3, dR, dBW]
    return ode

# ============================================================
# SIMULATION
# ============================================================
def simulate_cohort(coh_name, p):
    ev_R, ev_FR, ev_DR = build_dose_events(coh_name, p)
    ode    = build_ode(p, ev_R, ev_FR, ev_DR)

    t_end  = SIM_WEEKS * HOURS_PER_WEEK        # 4704 h
    n_pts  = int(t_end / 0.25) + 1            # 0.25h 해상도
    t_eval = np.linspace(0, t_end, n_pts)
    y0     = [0.0] * 7 + [p['BW0']]

    sol = solve_ivp(
        ode, [0, t_end], y0,
        t_eval=t_eval,
        method='LSODA',
        rtol=1e-7, atol=1e-10,
        dense_output=False
    )
    if not sol.success:
        return None

    t_h    = sol.t
    C_ugL  = np.clip(sol.y[0], 0.0, None) / p['V']   # µg/L
    BW     = np.clip(sol.y[7], 0.0, None)
    BW_pct = (BW - p['BW0']) / p['BW0'] * 100.0       # % from BW0=100

    # GI AE: AE_Total = E0_AE + Emax_AE×C/(EC50_AE+C)
    GI = np.clip(
        p['E0_AE'] + p['Emax_AE'] * C_ugL / (p['EC50_AE'] + C_ugL + 1e-15),
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
    st.markdown(
        '<span style="font-size:0.75rem;color:#94a3b8">'
        'DWJ1691 + Wegovy · v1.4</span>',
        unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#2d4a6e;margin:10px 0">', unsafe_allow_html=True)

    st.markdown('<div class="sb-hdr">Cohort Selection</div>', unsafe_allow_html=True)
    sel = {coh: st.checkbox(coh, value=True, key=f"chk_{coh}")
           for coh in COHORT_COLORS}

    st.markdown('<hr style="border-color:#2d4a6e;margin:10px 0">', unsafe_allow_html=True)
    st.markdown('<div class="sb-hdr">DWJ1691 Dosing</div>', unsafe_allow_html=True)
    for pat, dose, t_info in [
        ("W-W-<b>T</b>-W-W", "8,000 µg",  "h1344 · wk8"),
        ("W-W-W-<b>T</b>-W", "13,600 µg", "h2016 · wk12"),
        ("W-W-W-W-<b>T</b>", "19,200 µg", "h2688 · wk16"),
    ]:
        st.markdown(
            f'<div class="dose-chip">{pat}<br>'
            f'<b>{dose}</b> &nbsp;@ {t_info}</div>',
            unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#2d4a6e;margin:10px 0">', unsafe_allow_html=True)
    st.markdown(
        '<span style="font-size:0.72rem;color:#94a3b8">'
        '📅 28-week simulation<br>'
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
    <div class="design-item">
      <div class="design-dot" style="background:#94a3b8"></div>
      <span>Wegovy: 250→500→1000→1700→2400 µg  q1w × 4doses/block</span>
    </div>
    <div class="design-item">
      <div class="design-dot" style="background:#2166ac"></div>
      <span>Cohort I &nbsp;: DWJ <b>8,000 µg</b> @ h1344 (replaces 1000µg block)</span>
    </div>
    <div class="design-item">
      <div class="design-dot" style="background:#16a34a"></div>
      <span>Cohort II : DWJ <b>13,600 µg</b> @ h2016 (replaces 1700µg block)</span>
    </div>
    <div class="design-item">
      <div class="design-dot" style="background:#dc2626"></div>
      <span>Cohort III: DWJ <b>19,200 µg</b> @ h2688 (replaces 2400µg block)</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

active = [c for c, v in sel.items() if v]
if not active:
    st.warning("코호트를 한 개 이상 선택해주세요.")
    st.stop()

with st.spinner("🔬 ODE 시뮬레이션 실행 중..."):
    results = run_simulation(tuple(active), _ver="v1.4")

results = {k: v for k, v in results.items() if v is not None}
if not results:
    st.error("시뮬레이션 오류가 발생했습니다.")
    st.stop()

all_C  = np.concatenate([r['C_ugL']  for r in results.values()])
all_bw = np.concatenate([r['BW_pct'] for r in results.values()])
all_gi = np.concatenate([r['GI']     for r in results.values()])

# ── KPI ──────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card kpi-blue">
      <div class="kpi-label">C<sub>max</sub> (all cohorts)</div>
      <div class="kpi-value cv-blue">{np.max(all_C):.1f}</div>
      <div class="kpi-unit">µg/L</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card kpi-red">
      <div class="kpi-label">DWJ1691 Doses</div>
      <div class="kpi-value cv-red" style="font-size:1.15rem">
        8k / 13.6k / 19.2k</div>
      <div class="kpi-unit">µg · Cohort I / II / III</div></div>""",
                unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card kpi-green">
      <div class="kpi-label">Max BW Loss</div>
      <div class="kpi-value cv-green">{np.min(all_bw):.1f}%</div>
      <div class="kpi-unit">from BW₀ = 100 kg</div></div>""",
                unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card kpi-orange">
      <div class="kpi-label">Peak GI AE Rate</div>
      <div class="kpi-value cv-orange">{np.max(all_gi):.1f}%</div>
      <div class="kpi-unit">adverse event</div></div>""",
                unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── PK Profile ────────────────────────────────────────────────
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="sec-hdr">📈 PK Profile — Plasma Concentration (µg/L)</div>',
    unsafe_allow_html=True)
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

fig_pk = go.Figure()
for coh, r in results.items():
    t_wk = r['t_h'] / HOURS_PER_WEEK
    s    = max(1, len(t_wk) // 2000)
    fig_pk.add_trace(go.Scatter(
        x=t_wk[::s], y=r['C_ugL'][::s], name=coh,
        line=dict(color=COHORT_COLORS[coh], width=2, dash=COHORT_DASH[coh]),
        hovertemplate=(f"<b>{coh}</b><br>"
                       "Week: %{x:.2f}<br>"
                       "Conc: %{y:.1f} µg/L<extra></extra>")
    ))

# DWJ1691 투여 시점 수직선
vline_info = {
    "Cohort I (W-W-T-W-W)":   (1344/HOURS_PER_WEEK, "#2166ac", "8,000µg"),
    "Cohort II (W-W-W-T-W)":  (2016/HOURS_PER_WEEK, "#16a34a", "13,600µg"),
    "Cohort III (W-W-W-W-T)": (2688/HOURS_PER_WEEK, "#dc2626", "19,200µg"),
}
for coh in active:
    if coh in vline_info:
        t_v, col, lbl = vline_info[coh]
        fig_pk.add_vline(
            x=t_v, line_dash="dash", line_color=col,
            line_width=1, opacity=0.45,
            annotation_text=f"DWJ {lbl}",
            annotation_position="top",
            annotation_font=dict(size=9, color=col)
        )

fig_pk.update_layout(
    **CHART_BG, height=430,
    xaxis_title="Time (Week)",
    yaxis_title="Plasma concentration (µg/L)",
    legend=dict(
        bgcolor="rgba(255,255,255,0.95)", bordercolor="#e2e8f0",
        borderwidth=1, orientation="h",
        yanchor="bottom", y=1.01, xanchor="left", x=0,
        font=dict(size=11)
    )
)
st.plotly_chart(fig_pk, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── BW + GI ──────────────────────────────────────────────────
col_bw, col_gi = st.columns(2)

with col_bw:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-hdr">⚖️ Body Weight Change (%BW from baseline)</div>',
        unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    fig_bw = go.Figure()
    for coh, r in results.items():
        t_wk = r['t_h'] / HOURS_PER_WEEK
        s    = max(1, len(t_wk) // 2000)
        fig_bw.add_trace(go.Scatter(
            x=t_wk[::s], y=r['BW_pct'][::s], name=coh,
            line=dict(color=COHORT_COLORS[coh], width=2, dash=COHORT_DASH[coh]),
            hovertemplate=(f"<b>{coh}</b><br>"
                           "Week: %{x:.1f}<br>"
                           "ΔBW: %{y:.2f}%<extra></extra>")
        ))
    fig_bw.add_hline(y=0, line_dash="dot", line_color="#cbd5e1", line_width=1)
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
    st.markdown(
        '<div class="sec-hdr">⚠️ GI Adverse Event Rate (%)</div>',
        unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    fig_gi = go.Figure()
    for coh, r in results.items():
        t_wk = r['t_h'] / HOURS_PER_WEEK
        s    = max(1, len(t_wk) // 2000)
        fig_gi.add_trace(go.Scatter(
            x=t_wk[::s], y=r['GI'][::s], name=coh,
            line=dict(color=COHORT_COLORS[coh], width=2, dash=COHORT_DASH[coh]),
            hovertemplate=(f"<b>{coh}</b><br>"
                           "Week: %{x:.1f}<br>"
                           "GI AE: %{y:.1f}%<extra></extra>")
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

# ── Summary Table ─────────────────────────────────────────────
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="sec-hdr">'
    '📊 Cohort Summary — PK Parameters & PD/Safety Endpoints'
    '</div>',
    unsafe_allow_html=True)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

DWJ_DOSE_DISPLAY = {
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
        "DWJ Dose (µg)":    DWJ_DOSE_DISPLAY[coh],
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

# ── Model Info ────────────────────────────────────────────────
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
with st.expander("📋 Model Parameters & Dosing Schedule"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**PK — 1-Cpt Multiple Absorption**")
        st.markdown(f"- V = {P['V']} L")
        st.markdown(f"- CL = {P['Cl']} L/h")
        st.markdown(f"- Ka  (FR→A1) = {P['Ka']} h⁻¹")
        st.markdown(f"- ka_SC (R→A1) = {P['ka_SC']} h⁻¹")
        st.markdown(f"- F_SC = {P['F_SC']} (Wegovy bioavail)")
        st.markdown(f"- Scale_LAI = {P['Scale_LAI']}")
        st.markdown(f"- F_DR = {P['F_DR']}  → F_FR = {P['F_SC']-P['F_DR']:.3f}")
        st.markdown(f"- kdr = {P['kdr']} h⁻¹")
    with c2:
        st.markdown("**BW PD — Indirect Response**")
        st.markdown(f"- E(C) = Imax·Cᵞ / (IC50ᵞ + Cᵞ)")
        st.markdown(f"- Imax = {P['Imax']},  IC50 = {P['IC50']} µg/L")
        st.markdown(f"- Gamma = {P['Gamma']},  kout = {P['kout']} h⁻¹")
        st.markdown(f"- CB(t) = 100−6·(1−e^(−0.0001t))")
        st.markdown(f"- kin(t) = kout × CB(t)")
        st.markdown(f"- BW₀ = {P['BW0']} kg (fixed)")
        st.markdown("**GI AE — Simple Emax**")
        st.markdown(f"- E₀={P['E0_AE']}, Emax={P['Emax_AE']}")
        st.markdown(f"- EC50={P['EC50_AE']} µg/L")
    with c3:
        st.markdown("**Dosing Schedule (µg)**")
        st.markdown("*Wegovy SC q1w:*")
        for blk_i, (d, ts) in enumerate(WEGOVY_BLOCKS):
            st.markdown(f"- Blk{blk_i}: **{d}µg** × 4  (h{ts[0]}~{ts[-1]})")
        st.markdown("*DWJ1691 single SC:*")
        st.markdown("- Cohort I : **8,000µg** @ h1344")
        st.markdown("- Cohort II: **13,600µg** @ h2016")
        st.markdown("- Cohort III: **19,200µg** @ h2688")
        st.markdown("*R ↔ FR/DR 상호 배타적 (동시 투여 없음)*")

st.markdown("""
<div style='text-align:center;color:#94a3b8;font-size:0.72rem;
            padding:10px 0;margin-top:12px;background:#ffffff;
            border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,0.05)'>
  Phoenix NLME · Taeheon Kim, Ph.D. · 2026-02-19 · 28-week simulation · v1.4
</div>
""", unsafe_allow_html=True)
