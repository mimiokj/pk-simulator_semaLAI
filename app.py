import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid
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
# CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a3040;
    }
    .kpi-card {
        background: #161b27; border: 1px solid #2a3550;
        border-radius: 10px; padding: 14px 16px;
        text-align: center; margin-bottom: 8px;
    }
    .kpi-label {
        font-size: 0.68rem; color: #8892a4; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 5px;
    }
    .kpi-value { font-size: 1.6rem; font-weight: 700; line-height: 1.1; }
    .kpi-unit  { font-size: 0.68rem; color: #6b7585; margin-top: 3px; }
    .kpi-blue  { color: #60a5fa; }
    .kpi-green { color: #34d399; }
    .kpi-amber { color: #fbbf24; }
    .kpi-red   { color: #f87171; }
    .sec-hdr {
        font-size: 0.68rem; font-weight: 700; color: #7eb8f7;
        text-transform: uppercase; letter-spacing: 0.08em;
        border-bottom: 1px solid #2a3550;
        padding-bottom: 5px; margin: 10px 0 8px 0;
    }
    .main-title { font-size: 1.35rem; font-weight: 700; color: #e2e8f0; }
    .main-sub   { font-size: 0.82rem; color: #64748b; margin-bottom: 16px; }
    .badge {
        display: inline-block; background: #1e3a5f; color: #60a5fa;
        font-size: 0.65rem; font-weight: 600; padding: 2px 9px;
        border-radius: 20px; border: 1px solid #2a4a7a; margin-left: 6px;
    }
    hr { border-color: #2a3040 !important; }
    .stButton > button {
        background: #2166ac; color: white; border: none;
        border-radius: 8px; font-weight: 600;
        width: 100%; padding: 0.5rem;
    }
    .stButton > button:hover { background: #1a5490; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY THEME
# ============================================================
BASE_LAYOUT = dict(
    paper_bgcolor="#161b27", plot_bgcolor="#0f1117",
    font=dict(family="Inter, sans-serif", color="#c8d0e0", size=12),
    xaxis=dict(gridcolor="#1e2640", gridwidth=0.5,
               linecolor="#2a3550", tickcolor="#2a3550",
               title_font=dict(size=11, color="#8892a4")),
    yaxis=dict(gridcolor="#1e2640", gridwidth=0.5,
               linecolor="#2a3550", tickcolor="#2a3550",
               title_font=dict(size=11, color="#8892a4")),
    margin=dict(l=55, r=20, t=50, b=50),
    hovermode="x unified"
)

COHORT_COLORS = {
    "Reference":               "#94a3b8",
    "Cohort I (W-W-T-W-W)":   "#60a5fa",
    "Cohort II (W-W-W-T-W)":  "#34d399",
    "Cohort III (W-W-W-W-T)": "#f87171",
}

COHORT_SKIP = {
    "Reference":               None,
    "Cohort I (W-W-T-W-W)":   2,
    "Cohort II (W-W-W-T-W)":  3,
    "Cohort III (W-W-W-W-T)": 4,
}

# ============================================================
# MODEL PARAMETERS — Phoenix NLME fixef (Taeheon Kim, Ph.D.)
# ============================================================
# PML unit convention:
#   Dose  : mg
#   Amount: mg  (A1, FR, DR, R 모두)
#   V     : L
#   C     = A1 / V  → mg/L
#   Cl    : L/h
#   IC50, EC50_AE : mg/L  (C와 동일 단위, PML 원본 기준)
#
# Display:
#   C_display = C * 1000  → µg/L  (화면 표시용만)
# ============================================================
P = dict(
    # PK
    V         = 25.0,      # L
    Cl        = 3.5,       # L/h
    Ka        = 5.2,       # h⁻¹  FR depot → A1
    ka_SC     = 1.0,       # h⁻¹  R depot  → A1
    F_SC      = 0.9,       # –    Wegovy SC bioavailability
    Scale_LAI = 0.2,       # –    DWJ1691 LAI scaling
    F_DR      = 0.2,       # –    delayed-release fraction
    # F_FR = F_SC - F_DR = 0.7 (fast-release fraction)
    kdr       = 1.0,       # h⁻¹  3-transit delayed rate

    # BW PD (Indirect Response, sigmoidal Imax)
    # C 단위: mg/L  → IC50도 mg/L
    BW0       = 100.0,     # kg
    Imax      = 0.21,
    IC50      = 0.055,     # mg/L  (= 55 µg/L ÷ 1000)
    Gamma     = 0.5,
    kout      = 0.00039,   # h⁻¹

    # GI AE (Simple Emax)
    # C 단위: mg/L  → EC50_AE도 mg/L
    E0_AE     = 0.4833,
    Emax_AE   = 0.2867,
    EC50_AE   = 0.03298,   # mg/L  (= 32.98 µg/L ÷ 1000)
)

# ============================================================
# ODE — PML → Python 직번역
# ============================================================
# State vector: [A1, FR, DR, DR1, DR2, DR3, R, BW]
#
# PML dosepoint 처리:
#   dosepoint(FR, bioavail = Scale_LAI * F_FR)
#     → DWJ1691 dose * Scale_LAI * F_FR 가 FR에 주입
#   dosepoint(DR, bioavail = Scale_LAI * F_DR)
#     → DWJ1691 dose * Scale_LAI * F_DR 가 DR에 주입
#   dosepoint(R,  bioavail = F_SC)
#     → Wegovy dose * F_SC 가 R에 주입
# ============================================================
def build_ode(p, doses_R, doses_FR, doses_DR):
    """
    doses_R  : [(time_h, raw_dose_mg), ...]  Wegovy SC
    doses_FR : [(time_h, raw_dose_mg), ...]  DWJ1691 LAI
    doses_DR : [(time_h, raw_dose_mg), ...]  DWJ1691 LAI (same time, same dose)
    bioavail 은 이미 dose 계산 시 적용됨
    """
    def bolus_rate(dlist, t, dur=0.5):
        """볼러스를 짧은 infusion으로 근사 (수치 안정성)"""
        val = 0.0
        for (td, amt) in dlist:
            if td <= t < td + dur:
                val += amt / dur
        return val

    def ode(t, y):
        A1, FR, DR, DR1, DR2, DR3, R, BW = [max(v, 0.0) for v in y]

        # C = A1 / V  (mg/L) — PML 원본과 동일
        C = A1 / p['V']

        # ── PK (PML deriv 블록 직번역) ──
        dA1  = -(p['Cl'] * C) + (p['ka_SC'] * R) + (FR * p['Ka']) + (DR3 * p['kdr'])
        dFR  = -(FR * p['Ka'])   + bolus_rate(doses_FR, t)
        dDR  = -(DR * p['kdr'])  + bolus_rate(doses_DR, t)
        dDR1 = (DR  * p['kdr']) - (DR1 * p['kdr'])
        dDR2 = (DR1 * p['kdr']) - (DR2 * p['kdr'])
        dDR3 = (DR2 * p['kdr']) - (DR3 * p['kdr'])
        dR   = -(R * p['ka_SC']) + bolus_rate(doses_R, t)

        # ── BW PD (Indirect Response) ──
        # E = (Imax * C^Gamma) / (IC50^Gamma + C^Gamma)
        # C, IC50 모두 mg/L
        C_safe = max(C, 0.0)
        E   = (p['Imax'] * C_safe**p['Gamma']) / \
              (p['IC50']**p['Gamma'] + C_safe**p['Gamma'] + 1e-15)
        CB  = 100.0 - 6.0 * (1.0 - np.exp(-0.0001 * t))
        kin = p['kout'] * CB
        dBW = kin * (1.0 - E) - p['kout'] * BW

        return [dA1, dFR, dDR, dDR1, dDR2, dDR3, dR, dBW]

    return ode

# ============================================================
# DOSE SCHEDULE BUILDERS
# ============================================================
def build_wegovy_doses(skip_block=None):
    """
    Wegovy SC: 주1회, 4주 블록 × 5단계 용량 escalation
    dosepoint(R, bioavail = F_SC)
    → R depot에 들어가는 양 = raw_dose * F_SC
    """
    levels = [0.25, 0.5, 1.0, 1.7, 2.4]   # mg (raw dose)
    F_SC   = P['F_SC']
    out = []
    for blk in range(5):
        if blk == skip_block:
            continue
        for w in range(4):
            t_h = (blk * 28 + w * 7) * 24.0
            out.append((t_h, levels[blk] * F_SC))   # bioavail 적용
    return out

def build_dwj_doses(skip_block, dose_mg):
    """
    DWJ1691 LAI: 월1회 SC
    dosepoint(FR, bioavail = Scale_LAI * F_FR)
    dosepoint(DR, bioavail = Scale_LAI * F_DR)
    F_FR = F_SC - F_DR
    """
    if skip_block is None:
        return [], []
    t_h     = skip_block * 28 * 24.0
    F_FR    = P['F_SC'] - P['F_DR']     # 0.9 - 0.2 = 0.7
    F_DR    = P['F_DR']                  # 0.2
    Scale   = P['Scale_LAI']             # 0.2

    doses_FR = [(t_h, dose_mg * Scale * F_FR)]   # FR depot 주입량
    doses_DR = [(t_h, dose_mg * Scale * F_DR)]   # DR depot 주입량
    return doses_FR, doses_DR

# ============================================================
# SIMULATION
# ============================================================
def run_one(coh, dwj_mg, sim_weeks, p):
    skip          = COHORT_SKIP[coh]
    doses_R       = build_wegovy_doses(skip)
    doses_FR, doses_DR = build_dwj_doses(skip, dwj_mg)

    ode    = build_ode(p, doses_R, doses_FR, doses_DR)
    t_end  = sim_weeks * 7 * 24.0
    t_eval = np.linspace(0, t_end, int(t_end) + 1)
    y0     = [0.0] * 7 + [p['BW0']]

    sol = solve_ivp(
        ode, [0, t_end], y0,
        t_eval=t_eval,
        method='LSODA',
        rtol=1e-6, atol=1e-10,
        dense_output=False
    )

    t_h    = sol.t
    C_mgL  = np.clip(sol.y[0], 0, None) / p['V']     # mg/L
    C_ugL  = C_mgL * 1000.0                           # µg/L (display)
    BW     = np.clip(sol.y[7], 0, None)
    BW_pct = (BW - p['BW0']) / p['BW0'] * 100.0

    # GI AE: AE_drug = (Emax_AE * C) / (EC50_AE + C), C in mg/L
    GI = np.clip(
        p['E0_AE'] + p['Emax_AE'] * C_mgL / (p['EC50_AE'] + C_mgL + 1e-15),
        0, 1) * 100.0

    return dict(t_h=t_h, C_mgL=C_mgL, C_ugL=C_ugL,
                BW_pct=BW_pct, GI=GI)

@st.cache_data(show_spinner=False)
def run_all(cohorts, dwj_mg, sim_weeks):
    return {coh: run_one(coh, dwj_mg, sim_weeks, P)
            for coh in cohorts}

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
    st.markdown("### 💊 DWJ1691 + Wegovy")
    st.markdown("**PK/PD Simulator** `v1.0`")
    st.markdown("---")

    st.markdown('<div class="sec-hdr">Cohort Selection</div>',
                unsafe_allow_html=True)
    sel = {coh: st.checkbox(coh, value=True, key=f"chk_{coh}")
           for coh in COHORT_COLORS}

    st.markdown("---")
    st.markdown('<div class="sec-hdr">DWJ1691 Dose</div>',
                unsafe_allow_html=True)
    dwj_dose = st.slider("SC dose (mg)", 1, 30, 10, 1)

    st.markdown('<div class="sec-hdr">Simulation Duration</div>',
                unsafe_allow_html=True)
    sim_weeks = st.slider("Weeks", 12, 36, 25, 1)
    st.markdown("---")

# ============================================================
# MAIN
# ============================================================
st.markdown("""
<div class="main-title">DWJ1691 + Wegovy
  <span class="badge">PK/PD/Safety</span>
  <span class="badge">Phoenix NLME</span>
</div>
<div class="main-sub">
  1-Cpt · Multiple Absorption (LAI fast+delayed + Wegovy SC)
  · Indirect Response BW · Simple Emax GI AE<br>
  <b style="color:#60a5fa">Modeler: Taeheon Kim, Ph.D. · 2026-02-19</b>
</div>
""", unsafe_allow_html=True)

active = [c for c, v in sel.items() if v]
if not active:
    st.warning("코호트를 한 개 이상 선택해주세요.")
    st.stop()

with st.spinner("ODE 시뮬레이션 실행 중..."):
    results = run_all(tuple(active), dwj_dose, sim_weeks)

# ---- KPI ----
all_C  = np.concatenate([r['C_ugL']  for r in results.values()])
all_bw = np.concatenate([r['BW_pct'] for r in results.values()])
all_gi = np.concatenate([r['GI']     for r in results.values()])

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">C<sub>max</sub> (all cohorts)</div>
      <div class="kpi-value kpi-blue">{np.max(all_C):.2f}</div>
      <div class="kpi-unit">µg/L</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">DWJ1691 Dose</div>
      <div class="kpi-value kpi-red">{dwj_dose}</div>
      <div class="kpi-unit">mg SC monthly</div></div>""", unsafe_allow_html=True)
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
        hovertemplate="%{x:.0f} h — %{y:.3f} µg/L"
    ))
fig_pk.update_layout(
    **BASE_LAYOUT, height=340,
    xaxis_title="Time (h)", yaxis_title="Plasma concentration (µg/L)",
    legend=dict(bgcolor="#1a2035", bordercolor="#2a3550", borderwidth=1,
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0, font=dict(size=10))
)
st.plotly_chart(fig_pk, use_container_width=True)

# ---- BW + GI ----
col_bw, col_gi = st.columns(2)
with col_bw:
    st.markdown('<div class="sec-hdr">Body Weight Change (%BW)</div>',
                unsafe_allow_html=True)
    fig_bw = go.Figure()
    for coh, r in results.items():
        s = max(1, len(r['t_h']) // 600)
        fig_bw.add_trace(go.Scatter(
            x=r['t_h'][::s], y=r['BW_pct'][::s], name=coh,
            line=dict(color=COHORT_COLORS[coh], width=2),
            hovertemplate="%{x:.0f} h — %{y:.2f}%"
        ))
    fig_bw.update_layout(**BASE_LAYOUT, height=300,
        xaxis_title="Time (h)", yaxis_title="BW change (%)",
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
    fig_gi.update_layout(**BASE_LAYOUT, height=300,
        xaxis_title="Time (h)", yaxis_title="GI AE rate (%)",
        yaxis=dict(**BASE_LAYOUT['yaxis'], range=[0, 100]),
        showlegend=False)
    st.plotly_chart(fig_gi, use_container_width=True)

# ---- Summary Table ----
st.markdown('<div class="sec-hdr">Cohort Summary — PK Parameters & PD/Safety Endpoints</div>',
            unsafe_allow_html=True)
rows = []
for coh, r in results.items():
    Cmax, Tmax, AUC, Clast = pk_params(r['t_h'], r['C_ugL'])
    rows.append({
        "Cohort":            coh,
        "Cmax (µg/L)":      round(Cmax,  3),
        "Tmax (h)":         round(Tmax,  1),
        "AUClast (µg·h/L)": round(AUC,   1),
        "Clast (µg/L)":     round(Clast, 3),
        "Max ΔBW (%)":      round(float(np.min(r['BW_pct'])), 2),
        "Peak GI AE (%)":   round(float(np.max(r['GI'])),     1),
    })
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True,
    column_config={
        "Cohort":            st.column_config.TextColumn(width="large"),
        "Cmax (µg/L)":       st.column_config.NumberColumn(format="%.3f"),
        "Tmax (h)":          st.column_config.NumberColumn(format="%.1f"),
        "AUClast (µg·h/L)":  st.column_config.NumberColumn(format="%.1f"),
        "Clast (µg/L)":      st.column_config.NumberColumn(format="%.3f"),
        "Max ΔBW (%)":       st.column_config.NumberColumn(format="%.2f"),
        "Peak GI AE (%)":    st.column_config.NumberColumn(format="%.1f"),
    })

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download Summary CSV", csv,
                   file_name=f"pkpd_DWJ{dwj_dose}mg.csv",
                   mime="text/csv")

# ---- Model Info ----
st.markdown("---")
with st.expander("📋 Model Parameters (Phoenix NLME fixef)"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**PK — 1-Cpt Multiple Absorption**")
        st.markdown(f"- V = {P['V']} L")
        st.markdown(f"- CL = {P['Cl']} L/h")
        st.markdown(f"- Ka (FR→A1) = {P['Ka']} h⁻¹")
        st.markdown(f"- ka_SC (R→A1) = {P['ka_SC']} h⁻¹")
        st.markdown(f"- F_SC = {P['F_SC']}")
        st.markdown(f"- Scale_LAI = {P['Scale_LAI']}")
        st.markdown(f"- F_DR = {P['F_DR']}  →  F_FR = {P['F_SC']-P['F_DR']}")
        st.markdown(f"- kdr = {P['kdr']} h⁻¹")
    with c2:
        st.markdown("**BW PD — Indirect Response**")
        st.markdown(f"- Imax = {P['Imax']}")
        st.markdown(f"- IC50 = {P['IC50']} mg/L (= {P['IC50']*1000:.1f} µg/L)")
        st.markdown(f"- Gamma = {P['Gamma']}")
        st.markdown(f"- kout = {P['kout']} h⁻¹")
        st.markdown(f"- BW₀ = {P['BW0']} kg")
        st.markdown(f"- Current_Baseline = 100 − 6×(1−e^(−0.0001t))")
    with c3:
        st.markdown("**GI AE — Simple Emax**")
        st.markdown(f"- E0 = {P['E0_AE']}")
        st.markdown(f"- Emax = {P['Emax_AE']}")
        st.markdown(f"- EC50 = {P['EC50_AE']} mg/L (= {P['EC50_AE']*1000:.2f} µg/L)")
        st.markdown("---")
        st.markdown("**Dose Units**")
        st.markdown("- Wegovy: mg SC q1w")
        st.markdown("- DWJ1691: mg SC q4w")
        st.markdown("- C = A1/V → mg/L (×1000 = µg/L 표시)")

st.markdown("""
<div style='text-align:center;color:#2a3560;font-size:0.72rem;padding:6px 0;margin-top:8px'>
  Phoenix NLME · Taeheon Kim, Ph.D. · 2026-02-19
</div>
""", unsafe_allow_html=True)
