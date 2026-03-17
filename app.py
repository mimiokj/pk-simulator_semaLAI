import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
import plotly.graph_objects as go
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
# CSS — Light Clinical Theme
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #f5f7fa; }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] label { color: #334155 !important; font-size: 0.88rem; }
    .kpi-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 16px 18px; text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06); margin-bottom: 8px;
    }
    .kpi-label {
        font-size: 0.68rem; color: #94a3b8; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 5px;
    }
    .kpi-value { font-size: 1.65rem; font-weight: 700; line-height: 1.1; }
    .kpi-unit  { font-size: 0.68rem; color: #94a3b8; margin-top: 3px; }
    .kpi-blue  { color: #2166ac; }
    .kpi-green { color: #16a34a; }
    .kpi-amber { color: #d97706; }
    .kpi-red   { color: #dc2626; }
    .sec-hdr {
        font-size: 0.68rem; font-weight: 700; color: #2166ac;
        text-transform: uppercase; letter-spacing: 0.08em;
        border-bottom: 2px solid #dbeafe;
        padding-bottom: 5px; margin: 14px 0 10px 0;
    }
    .main-title { font-size: 1.45rem; font-weight: 700; color: #1e293b; margin-bottom: 2px; }
    .main-sub   { font-size: 0.82rem; color: #64748b; margin-bottom: 18px; }
    .badge {
        display: inline-block; background: #dbeafe; color: #1d4ed8;
        font-size: 0.65rem; font-weight: 700; padding: 2px 10px;
        border-radius: 20px; margin-left: 6px; vertical-align: middle;
    }
    .badge-green {
        display: inline-block; background: #dcfce7; color: #15803d;
        font-size: 0.65rem; font-weight: 700; padding: 2px 10px;
        border-radius: 20px; margin-left: 6px; vertical-align: middle;
    }
    .dose-info {
        background: #f0f9ff; border: 1px solid #bae6fd;
        border-radius: 8px; padding: 10px 14px;
        font-size: 0.8rem; color: #0369a1; margin-bottom: 12px;
    }
    .dose-info b { color: #0c4a6e; }
    hr { border-color: #e2e8f0 !important; }
    .stButton > button {
        background: #2166ac; color: white; border: none;
        border-radius: 8px; font-weight: 600;
        width: 100%; padding: 0.5rem; transition: background 0.2s;
    }
    .stButton > button:hover { background: #1a5490; }
    .sb-hdr {
        font-size: 0.68rem; font-weight: 700; color: #2166ac;
        text-transform: uppercase; letter-spacing: 0.07em;
        border-bottom: 1px solid #dbeafe;
        padding-bottom: 4px; margin: 12px 0 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY LIGHT THEME
# ============================================================
BASE_LAYOUT = dict(
    paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
    font=dict(family="Inter, sans-serif", color="#334155", size=12),
    xaxis=dict(gridcolor="#e2e8f0", gridwidth=0.8,
               linecolor="#cbd5e1", tickcolor="#cbd5e1",
               title_font=dict(size=11, color="#64748b")),
    yaxis=dict(gridcolor="#e2e8f0", gridwidth=0.8,
               linecolor="#cbd5e1", tickcolor="#cbd5e1",
               title_font=dict(size=11, color="#64748b")),
    margin=dict(l=55, r=20, t=50, b=50),
    hovermode="x unified"
)

COHORT_COLORS = {
    "Reference":               "#888888",
    "Cohort I (W-W-T-W-W)":   "#2166ac",
    "Cohort II (W-W-W-T-W)":  "#16a34a",
    "Cohort III (W-W-W-W-T)": "#dc2626",
}

# ============================================================
# STUDY DESIGN
# ============================================================
# Wegovy titration: 4주씩, 주1회 SC
WEGOVY_LEVELS = [0.25, 0.5, 1.0, 1.7, 2.4]   # mg, block 0~4

# Cohort별 DWJ1691 투여 블록 및 8배 용량 설정
# skip_block: 해당 블록의 Wegovy 투여 생략 (DWJ1691으로 대체)
# dwj_dose  : 해당 블록 Wegovy 용량의 8배
COHORT_CONFIG = {
    "Reference": {
        "skip_block": None,
        "dwj_block":  None,
        "dwj_dose":   0.0,
        "label":      "Wegovy only (titration)"
    },
    "Cohort I (W-W-T-W-W)": {
        "skip_block": 2,
        "dwj_block":  2,
        "dwj_dose":   WEGOVY_LEVELS[2] * 8,   # 1.0 × 8 = 8.0 mg
        "label":      f"DWJ {WEGOVY_LEVELS[2]*8:.1f} mg at wk9"
    },
    "Cohort II (W-W-W-T-W)": {
        "skip_block": 3,
        "dwj_block":  3,
        "dwj_dose":   WEGOVY_LEVELS[3] * 8,   # 1.7 × 8 = 13.6 mg
        "label":      f"DWJ {WEGOVY_LEVELS[3]*8:.1f} mg at wk13"
    },
    "Cohort III (W-W-W-W-T)": {
        "skip_block": 4,
        "dwj_block":  4,
        "dwj_dose":   WEGOVY_LEVELS[4] * 8,   # 2.4 × 8 = 19.2 mg
        "label":      f"DWJ {WEGOVY_LEVELS[4]*8:.1f} mg at wk17"
    },
}

SIM_WEEKS = 28   # 고정

# ============================================================
# MODEL PARAMETERS — Phoenix NLME fixef (Taeheon Kim, Ph.D.)
# C = A1/V (mg/L)  →  display: ×1000 = µg/L
# IC50, EC50_AE: mg/L (C와 동일 단위)
# ============================================================
P = dict(
    V         = 25.0,
    Cl        = 3.5,
    Ka        = 5.2,       # h⁻¹  FR depot → A1
    ka_SC     = 1.0,       # h⁻¹  R depot  → A1  (Wegovy)
    F_SC      = 0.9,       # Wegovy SC bioavailability
    Scale_LAI = 0.2,       # DWJ1691 LAI overall scaling
    F_DR      = 0.2,       # delayed-release fraction
    # F_FR = F_SC - F_DR = 0.7 (fast-release fraction)
    kdr       = 1.0,       # h⁻¹  3-transit delayed rate
    BW0       = 100.0,     # kg   baseline (fixed)
    Imax      = 0.21,
    IC50      = 0.055,     # mg/L (= 55 µg/L)
    Gamma     = 0.5,
    kout      = 0.00039,   # h⁻¹
    E0_AE     = 0.4833,
    Emax_AE   = 0.2867,
    EC50_AE   = 0.03298,   # mg/L (= 32.98 µg/L)
)

# ============================================================
# ODE — Phoenix PML 직번역
# State: [A1, FR, DR, DR1, DR2, DR3, R, BW]
#   A1        : central (mg)
#   FR        : fast-release LAI depot (mg)
#   DR/1/2/3  : 3-transit delayed-release (mg)
#   R         : Wegovy SC depot (mg)
#   BW        : body weight (kg)
# ============================================================
def build_ode(p, doses_R, doses_FR, doses_DR):
    def bolus_rate(dlist, t, dur=0.5):
        val = 0.0
        for (td, amt) in dlist:
            if td <= t < td + dur:
                val += amt / dur
        return val

    def ode(t, y):
        A1, FR, DR, DR1, DR2, DR3, R, BW = [max(v, 0.0) for v in y]
        C = A1 / p['V']   # mg/L

        # PK (PML deriv 블록 직번역)
        dA1  = -(p['Cl'] * C) + (p['ka_SC'] * R) + (FR * p['Ka']) + (DR3 * p['kdr'])
        dFR  = -(FR * p['Ka'])   + bolus_rate(doses_FR, t)
        dDR  = -(DR * p['kdr'])  + bolus_rate(doses_DR, t)
        dDR1 =  (DR  * p['kdr']) - (DR1 * p['kdr'])
        dDR2 =  (DR1 * p['kdr']) - (DR2 * p['kdr'])
        dDR3 =  (DR2 * p['kdr']) - (DR3 * p['kdr'])
        dR   = -(R * p['ka_SC']) + bolus_rate(doses_R, t)

        # BW PD — Indirect Response (C in mg/L, IC50 in mg/L)
        C_s  = max(C, 0.0)
        E    = (p['Imax'] * C_s**p['Gamma']) / \
               (p['IC50']**p['Gamma'] + C_s**p['Gamma'] + 1e-15)
        CB   = 100.0 - 6.0 * (1.0 - np.exp(-0.0001 * t))
        kin  = p['kout'] * CB
        dBW  = kin * (1.0 - E) - p['kout'] * BW

        return [dA1, dFR, dDR, dDR1, dDR2, dDR3, dR, dBW]
    return ode

# ============================================================
# DOSE BUILDERS
# ============================================================
def build_wegovy_doses(skip_block=None):
    """
    Wegovy SC 주1회, 4주 블록 × 5단계
    dosepoint(R, bioavail=F_SC) → R depot에 dose × F_SC 주입
    """
    out = []
    for blk in range(5):
        if blk == skip_block:
            continue
        amt = WEGOVY_LEVELS[blk] * P['F_SC']
        for w in range(4):
            t_h = (blk * 28 + w * 7) * 24.0
            out.append((t_h, amt))
    return out

def build_dwj_doses(dwj_block, dwj_dose_mg):
    """
    DWJ1691 LAI 1회 SC 주입
    dosepoint(FR, bioavail = Scale_LAI × F_FR) → FR depot
    dosepoint(DR, bioavail = Scale_LAI × F_DR) → DR depot
    F_FR = F_SC - F_DR = 0.7,  F_DR = 0.2
    """
    if dwj_block is None or dwj_dose_mg == 0:
        return [], []
    t_h   = dwj_block * 28 * 24.0
    F_FR  = P['F_SC'] - P['F_DR']          # 0.7
    F_DR  = P['F_DR']                       # 0.2
    Scale = P['Scale_LAI']                  # 0.2

    # 각 depot에 들어가는 실제 amount (mg)
    amt_FR = dwj_dose_mg * Scale * F_FR     # e.g. 8×0.2×0.7 = 1.12 mg
    amt_DR = dwj_dose_mg * Scale * F_DR     # e.g. 8×0.2×0.2 = 0.32 mg

    doses_FR = [(t_h, amt_FR)]
    doses_DR = [(t_h, amt_DR)]
    return doses_FR, doses_DR

# ============================================================
# SIMULATION
# ============================================================
def run_one(coh_name, p):
    cfg    = COHORT_CONFIG[coh_name]
    skip   = cfg['skip_block']
    dwj_blk= cfg['dwj_block']
    dwj_mg = cfg['dwj_dose']

    doses_R        = build_wegovy_doses(skip)
    doses_FR, doses_DR = build_dwj_doses(dwj_blk, dwj_mg)
    ode    = build_ode(p, doses_R, doses_FR, doses_DR)

    t_end  = SIM_WEEKS * 7 * 24.0
    t_eval = np.linspace(0, t_end, int(t_end) + 1)
    y0     = [0.0] * 7 + [p['BW0']]

    sol = solve_ivp(ode, [0, t_end], y0, t_eval=t_eval,
                    method='LSODA', rtol=1e-6, atol=1e-10)

    t_h    = sol.t
    C_mgL  = np.clip(sol.y[0], 0, None) / p['V']
    C_ugL  = C_mgL * 1000.0                       # µg/L (display)
    BW     = np.clip(sol.y[7], 0, None)

    # %BW change: (BW(t) - BW0) / BW0 × 100
    BW_pct = (BW - p['BW0']) / p['BW0'] * 100.0

    # GI AE: AE_drug = Emax_AE×C / (EC50_AE+C), C in mg/L
    GI = np.clip(
        p['E0_AE'] + p['Emax_AE'] * C_mgL / (p['EC50_AE'] + C_mgL + 1e-15),
        0, 1) * 100.0

    return dict(t_h=t_h, C_mgL=C_mgL, C_ugL=C_ugL,
                BW_pct=BW_pct, GI=GI)

@st.cache_data(show_spinner=False)
def run_all(cohorts):
    return {coh: run_one(coh, P) for coh in cohorts}

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
    st.markdown("## 💊 PK/PD Simulator")
    st.markdown("<span style='color:#64748b;font-size:0.8rem'>DWJ1691 + Wegovy · v1.0</span>",
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sb-hdr">Cohort Selection</div>',
                unsafe_allow_html=True)
    sel = {coh: st.checkbox(coh, value=True, key=f"chk_{coh}")
           for coh in COHORT_COLORS}

    st.markdown("---")

    # 코호트별 DWJ1691 용량 정보 표시
    st.markdown('<div class="sb-hdr">DWJ1691 Dose (8× Wegovy)</div>',
                unsafe_allow_html=True)
    for coh, cfg in COHORT_CONFIG.items():
        if cfg['dwj_dose'] > 0:
            blk  = cfg['dwj_block']
            dose = cfg['dwj_dose']
            t_wk = blk * 4 + 1
            st.markdown(
                f"<span style='font-size:0.78rem;color:#334155'>"
                f"**{coh.split('(')[1].rstrip(')')}** → "
                f"<span style='color:#2166ac;font-weight:700'>{dose:.1f} mg</span>"
                f" at wk{t_wk}</span>",
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown(
        "<span style='color:#94a3b8;font-size:0.72rem'>"
        "Sim duration: <b>28 weeks</b><br>"
        "Modeler: Taeheon Kim, Ph.D.<br>"
        "Phoenix NLME · 2026-02-19</span>",
        unsafe_allow_html=True
    )

# ============================================================
# MAIN
# ============================================================
st.markdown("""
<div class="main-title">
  DWJ1691 + Wegovy
  <span class="badge">PK/PD/Safety</span>
  <span class="badge-green">Phoenix NLME</span>
</div>
<div class="main-sub">
  1-Cpt · Multiple Absorption (LAI fast+delayed + Wegovy SC)
  · Indirect Response BW · Simple Emax GI AE · 28-week simulation
</div>
""", unsafe_allow_html=True)

# 도징 설계 요약 카드
st.markdown("""
<div class="dose-info">
  <b>Study Design</b> &nbsp;|&nbsp;
  Wegovy titration: 0.25→0.5→1.0→1.7→2.4 mg SC q1w (4wk each) &nbsp;|&nbsp;
  DWJ1691: <b>8× Wegovy dose</b> at corresponding block &nbsp;|&nbsp;
  Cohort I: <b>8.0 mg</b> (wk9) &nbsp;·&nbsp;
  Cohort II: <b>13.6 mg</b> (wk13) &nbsp;·&nbsp;
  Cohort III: <b>19.2 mg</b> (wk17)
</div>
""", unsafe_allow_html=True)

active = [c for c, v in sel.items() if v]
if not active:
    st.warning("코호트를 한 개 이상 선택해주세요.")
    st.stop()

with st.spinner("ODE 시뮬레이션 실행 중..."):
    results = run_all(tuple(active))

# ---- KPI ----
all_C  = np.concatenate([r['C_ugL']  for r in results.values()])
all_bw = np.concatenate([r['BW_pct'] for r in results.values()])
all_gi = np.concatenate([r['GI']     for r in results.values()])

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">C<sub>max</sub> (all cohorts)</div>
      <div class="kpi-value kpi-blue">{np.max(all_C):.3f}</div>
      <div class="kpi-unit">µg/L</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">DWJ1691 Doses</div>
      <div class="kpi-value kpi-red">8 / 13.6 / 19.2</div>
      <div class="kpi-unit">mg · Cohort I / II / III</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Max BW Loss</div>
      <div class="kpi-value kpi-green">{np.min(all_bw):.2f}%</div>
      <div class="kpi-unit">body weight</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Peak GI AE</div>
      <div class="kpi-value kpi-amber">{np.max(all_gi):.1f}%</div>
      <div class="kpi-unit">adverse event rate</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---- PK Chart ----
st.markdown('<div class="sec-hdr">PK Profile — Plasma Concentration (µg/L)</div>',
            unsafe_allow_html=True)
fig_pk = go.Figure()
for coh, r in results.items():
    s = max(1, len(r['t_h']) // 600)
    fig_pk.add_trace(go.Scatter(
        x=r['t_h'][::s], y=r['C_ugL'][::s], name=coh,
        line=dict(color=COHORT_COLORS[coh], width=2),
        hovertemplate="%{x:.0f} h — %{y:.4f} µg/L"
    ))
# DWJ1691 투여 시점 수직선 표시
for coh in active:
    cfg = COHORT_CONFIG[coh]
    if cfg['dwj_block'] is not None:
        t_dose = cfg['dwj_block'] * 28 * 24.0
        fig_pk.add_vline(
            x=t_dose, line_dash="dash",
            line_color=COHORT_COLORS[coh], line_width=1, opacity=0.5,
            annotation_text=f"{cfg['dwj_dose']:.1f}mg",
            annotation_font_size=9,
            annotation_font_color=COHORT_COLORS[coh]
        )
fig_pk.update_layout(
    **BASE_LAYOUT, height=360,
    xaxis_title="Time (h)",
    yaxis_title="Plasma concentration (µg/L)",
    legend=dict(bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#e2e8f0", borderwidth=1,
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0, font=dict(size=10))
)
st.plotly_chart(fig_pk, use_container_width=True)

# ---- BW + GI ----
col_bw, col_gi = st.columns(2)
with col_bw:
    st.markdown('<div class="sec-hdr">Body Weight Change (%BW from baseline)</div>',
                unsafe_allow_html=True)
    fig_bw = go.Figure()
    for coh, r in results.items():
        s = max(1, len(r['t_h']) // 600)
        fig_bw.add_trace(go.Scatter(
            x=r['t_h'][::s], y=r['BW_pct'][::s], name=coh,
            line=dict(color=COHORT_COLORS[coh], width=2),
            hovertemplate="%{x:.0f} h — %{y:.2f}%"
        ))
    # 기준선 (0%)
    fig_bw.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig_bw.update_layout(**BASE_LAYOUT, height=310,
        xaxis_title="Time (h)",
        yaxis_title="ΔBW (% from BW₀=100 kg)",
        showlegend=False)
    st.plotly_chart(fig_bw, use_container_width=True)

with col_gi:
    st.markdown('<div class="sec-hdr">GI Adverse Event Rate (%)</div>',
                unsafe_allow_html=True)
    fig_gi = go.Figure()
    for coh, r in results.items():
        s = max(1, len(r['t_h']) // 600)
        fig_gi.add_trace(go.Scatter(
            x=r['t_h'][::s], y=r['GI'][::s], name=coh,
            line=dict(color=COHORT_COLORS[coh], width=2),
            hovertemplate="%{x:.0f} h — %{y:.1f}%"
        ))
    fig_gi.update_layout(**BASE_LAYOUT, height=310,
        xaxis_title="Time (h)", yaxis_title="GI AE rate (%)",
        yaxis=dict(**BASE_LAYOUT['yaxis'], range=[0, 100]),
        showlegend=False)
    st.plotly_chart(fig_gi, use_container_width=True)

# ---- Summary Table ----
st.markdown('<div class="sec-hdr">Cohort Summary — PK Parameters & PD/Safety Endpoints</div>',
            unsafe_allow_html=True)
rows = []
for coh, r in results.items():
    cfg = COHORT_CONFIG[coh]
    Cmax, Tmax, AUC, Clast = pk_params(r['t_h'], r['C_ugL'])
    rows.append({
        "Cohort":            coh,
        "DWJ Dose (mg)":    cfg['dwj_dose'] if cfg['dwj_dose'] > 0 else "—",
        "Cmax (µg/L)":      round(Cmax,  4),
        "Tmax (h)":         round(Tmax,  1),
        "AUClast (µg·h/L)": round(AUC,   1),
        "Clast (µg/L)":     round(Clast, 4),
        "Max ΔBW (%)":      round(float(np.min(r['BW_pct'])), 2),
        "Peak GI AE (%)":   round(float(np.max(r['GI'])),     1),
    })
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True,
    column_config={
        "Cohort":            st.column_config.TextColumn(width="large"),
        "DWJ Dose (mg)":     st.column_config.TextColumn(),
        "Cmax (µg/L)":       st.column_config.NumberColumn(format="%.4f"),
        "Tmax (h)":          st.column_config.NumberColumn(format="%.1f"),
        "AUClast (µg·h/L)":  st.column_config.NumberColumn(format="%.1f"),
        "Clast (µg/L)":      st.column_config.NumberColumn(format="%.4f"),
        "Max ΔBW (%)":       st.column_config.NumberColumn(format="%.2f"),
        "Peak GI AE (%)":    st.column_config.NumberColumn(format="%.1f"),
    })

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download Summary CSV", csv,
                   file_name="pkpd_summary_28wk.csv", mime="text/csv")

# ---- Model Info ----
st.markdown("---")
with st.expander("📋 Model Parameters & Study Design"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**PK — 1-Cpt Multiple Absorption**")
        st.markdown(f"- V = {P['V']} L")
        st.markdown(f"- CL = {P['Cl']} L/h")
        st.markdown(f"- Ka (FR→A1) = {P['Ka']} h⁻¹")
        st.markdown(f"- ka_SC (R→A1) = {P['ka_SC']} h⁻¹")
        st.markdown(f"- F_SC = {P['F_SC']}  (Wegovy bioavail)")
        st.markdown(f"- Scale_LAI = {P['Scale_LAI']}")
        st.markdown(f"- F_DR = {P['F_DR']}  →  F_FR = {P['F_SC']-P['F_DR']}")
        st.markdown(f"- kdr = {P['kdr']} h⁻¹")
    with c2:
        st.markdown("**BW PD — Indirect Response**")
        st.markdown(f"- Imax = {P['Imax']}")
        st.markdown(f"- IC50 = {P['IC50']} mg/L (= {P['IC50']*1000:.1f} µg/L)")
        st.markdown(f"- Gamma = {P['Gamma']}")
        st.markdown(f"- kout = {P['kout']} h⁻¹")
        st.markdown(f"- BW₀ = {P['BW0']} kg (fixed)")
        st.markdown("- CB = 100 − 6×(1−e^(−0.0001t))")
    with c3:
        st.markdown("**GI AE — Simple Emax**")
        st.markdown(f"- E0 = {P['E0_AE']}")
        st.markdown(f"- Emax = {P['Emax_AE']}")
        st.markdown(f"- EC50 = {P['EC50_AE']} mg/L (= {P['EC50_AE']*1000:.2f} µg/L)")
        st.markdown("---")
        st.markdown("**Cohort Doses (8× Wegovy)**")
        for coh, cfg in COHORT_CONFIG.items():
            if cfg['dwj_dose'] > 0:
                st.markdown(f"- {coh.split('(')[1].rstrip(')')}: **{cfg['dwj_dose']:.1f} mg**")

st.markdown("""
<div style='text-align:center;color:#94a3b8;font-size:0.72rem;
            padding:8px 0;margin-top:8px;background:#f1f5f9;border-radius:8px;'>
  Phoenix NLME · Taeheon Kim, Ph.D. · 2026-02-19 · 28-week simulation
</div>
""", unsafe_allow_html=True)
