import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="DWJ1691 Clinical PK/PD Simulator",
    page_icon="💊", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #f0f4f8; }

/* ── 사이드바 ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a5f 0%, #1a3050 100%) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stCheckbox label { color: #cbd5e1 !important; font-size: 0.9rem !important; }
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSlider p { color: #bfdbfe !important; font-size: 0.85rem !important; }
[data-testid="stSidebar"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stTickBarMax"] { color: #93c5fd !important; }

/* ── 메인 영역 슬라이더 레이블 ── */
.stSlider > label, .stSlider > div > label { color: #1e293b !important; font-weight: 600 !important; }

/* ── 탭 ── */
.stTabs [data-baseweb="tab"] { font-size: 0.92rem; font-weight: 600; color: #334155 !important; }
.stTabs [aria-selected="true"] { color: #1e40af !important; }

.sb-logo { font-size: 1.1rem; font-weight: 700; color: #ffffff !important; }
.sb-hdr {
    font-size: 0.68rem; font-weight: 700; color: #60a5fa !important;
    text-transform: uppercase; letter-spacing: 0.08em;
    border-bottom: 1px solid #2d4a6e; padding-bottom: 4px; margin: 14px 0 8px 0;
}
.dose-box { background: rgba(96,165,250,0.12); border-radius: 8px; padding: 10px 12px; margin-top: 6px; }
.dose-box-title { color: #93c5fd !important; font-weight: 700; font-size: 0.85rem; margin-bottom: 7px; }
.dose-row { color: #bfdbfe !important; font-size: 0.83rem; padding: 3px 0; }
.dose-val { color: #7dd3fc !important; font-weight: 700; }
.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2166ac 100%);
    border-radius: 14px; padding: 22px 28px; margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(33,102,172,0.25);
}
.main-title { font-size: 1.5rem; font-weight: 700; color: #ffffff; margin-bottom: 3px; }
.main-title-ko { font-size: 0.95rem; color: #93c5fd; margin-bottom: 8px; font-weight: 500; }
.main-sub { font-size: 0.82rem; color: #7eb8f7; margin-bottom: 10px; }
.badge {
    display: inline-block; background: rgba(255,255,255,0.15); color: #ffffff;
    font-size: 0.68rem; font-weight: 600; padding: 3px 10px; border-radius: 20px;
    margin-right: 4px; border: 1px solid rgba(255,255,255,0.2);
}
.badge-yellow {
    display: inline-block; background: rgba(251,191,36,0.2); color: #fde68a;
    font-size: 0.68rem; font-weight: 600; padding: 3px 10px; border-radius: 20px;
    margin-right: 4px; border: 1px solid rgba(251,191,36,0.3);
}
.design-strip {
    background: rgba(255,255,255,0.08); border-radius: 8px;
    padding: 8px 14px; font-size: 0.80rem; color: #bfdbfe;
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
}
.design-item { display: flex; align-items: center; gap: 5px; }
.design-dot  { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.kpi-card {
    background: #ffffff; border-radius: 12px; padding: 16px 18px; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07); border-top: 3px solid transparent;
}
.kpi-blue   { border-top-color: #2166ac; } .kpi-red    { border-top-color: #dc2626; }
.kpi-green  { border-top-color: #16a34a; } .kpi-orange { border-top-color: #d97706; }
.kpi-purple { border-top-color: #7c3aed; }
.kpi-label  { font-size: 0.70rem; color: #334155; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
.kpi-label-ko { font-size: 0.65rem; color: #64748b; margin-bottom: 5px; }
.kpi-value  { font-size: 1.75rem; font-weight: 700; line-height: 1.0; }
.kpi-unit   { font-size: 0.68rem; color: #64748b; margin-top: 4px; }
.cv-blue  { color: #2166ac; } .cv-red    { color: #dc2626; }
.cv-green { color: #16a34a; } .cv-orange { color: #d97706; }
.cv-purple{ color: #7c3aed; }
.chart-card {
    background: #ffffff; border-radius: 12px; padding: 18px 22px;
    margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.sec-hdr {
    font-size: 0.78rem; font-weight: 700; color: #1e40af;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 6px;
}
.window-badge {
    display: inline-block; background: #fef3c7; color: #92400e;
    font-size: 0.68rem; font-weight: 600; padding: 2px 9px;
    border-radius: 4px; margin-left: 8px; border: 1px solid #fcd34d;
}
.custom-info {
    background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
    padding: 12px 16px; font-size: 0.84rem; color: #1e40af; margin-bottom: 12px;
}
hr { border-color: #2d4a6e !important; }
.stButton > button {
    background: linear-gradient(135deg,#2166ac,#1d4ed8); color: white;
    border: none; border-radius: 8px; font-weight: 600; width: 100%; padding: 0.5rem;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly 공통 테마 ──
# 축 제목을 확실히 검정으로 고정하는 핵심:
# update_layout 대신 fig.update_xaxes / fig.update_yaxes 로 별도 강제
AXIS_CFG = dict(
    gridcolor="#e2e8f0", gridwidth=1,
    linecolor="#475569", tickcolor="#475569", linewidth=1.5,
    zeroline=False, showline=True,
    title_font=dict(size=14, color="#111827", family="Inter, sans-serif"),
    tickfont=dict(size=12, color="#374151", family="Inter, sans-serif"),
)

def make_chart_bg():
    return dict(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fafbfc",
        font=dict(family="Inter, sans-serif", color="#1e293b", size=13),
        margin=dict(l=70, r=30, t=50, b=65),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor="#e2e8f0",
                        font=dict(size=12, color="#1e293b"))
    )

def apply_axes(fig, x_title, y_title):
    """축 제목/스타일을 update_xaxes/yaxes로 강제 적용 — 덮어씌움 방지"""
    fig.update_xaxes(title_text=x_title, **AXIS_CFG)
    fig.update_yaxes(title_text=y_title, **AXIS_CFG)

LEGEND_STYLE = dict(
    bgcolor="rgba(255,255,255,0.95)",
    bordercolor="#cbd5e1", borderwidth=1,
    orientation="h", yanchor="bottom", y=1.02,
    xanchor="left", x=0,
    font=dict(size=12, color="#1e293b")
)

SHORT_NAMES = {
    "Reference (Wegovy)":      "Wegovy (Reference)",
    "Cohort I (W-W-T-W-W)":   "Cohort I",
    "Cohort II (W-W-W-T-W)":  "Cohort II",
    "Cohort III (W-W-W-W-T)": "Cohort III",
}
COHORT_COLORS = {
    "Reference (Wegovy)":      "#64748b",
    "Cohort I (W-W-T-W-W)":   "#2166ac",
    "Cohort II (W-W-W-T-W)":  "#16a34a",
    "Cohort III (W-W-W-W-T)": "#dc2626",
}
COHORT_DASH = {
    "Reference (Wegovy)":      "dash",
    "Cohort I (W-W-T-W-W)":   "solid",
    "Cohort II (W-W-W-T-W)":  "solid",
    "Cohort III (W-W-W-W-T)": "solid",
}

HOURS_PER_WEEK = 168.0

P = dict(
    V=12.4, Cl=0.0475,
    Ka=0.1026, ka_SC=0.0296, F_SC=0.9,
    Scale_LAI=0.2459, F_DR=0.429, kdr=0.02,
    BW0=100.0, Imax=0.25, IC50=55.0, Gamma=0.5, kout=0.00039,
    Emax_PBO=0.0, k_pbo=0.0001,
    E0_AE=0.4833, Emax_AE=0.2867, EC50_AE=32.98,
)

DEFAULT_WEGOVY_BLOCKS = [
    (250,  [0,   168,  336,  504 ]),
    (500,  [672, 840,  1008, 1176]),
    (1000, [1344,1512, 1680, 1848]),
    (1700, [2016,2184, 2352, 2520]),
    (2400, [2688,2856, 3024, 3192]),
]

COHORT_META = {
    "Reference (Wegovy)":      {"skip": None, "blk": None, "wegovy_dose_ug": 0},
    "Cohort I (W-W-T-W-W)":   {"skip": 2,    "blk": 2,    "wegovy_dose_ug": 1000},
    "Cohort II (W-W-W-T-W)":  {"skip": 3,    "blk": 3,    "wegovy_dose_ug": 1700},
    "Cohort III (W-W-W-W-T)": {"skip": 4,    "blk": 4,    "wegovy_dose_ug": 2400},
}

COHORT_WINDOWS = {
    "Cohort I (W-W-T-W-W)":   {"t_dose": 1344.0, "wegovy_label": "1.0 mg 구간"},
    "Cohort II (W-W-W-T-W)":  {"t_dose": 2016.0, "wegovy_label": "1.7 mg 구간"},
    "Cohort III (W-W-W-W-T)": {"t_dose": 2688.0, "wegovy_label": "2.4 mg 구간"},
}

def make_ode(p):
    def ode(t, y):
        A1,FR,DR,DR1,DR2,DR3,R,BW = [max(v,0.0) for v in y]
        C    = A1/p['V']
        dA1  = -(p['Cl']*C)+(p['ka_SC']*R)+(FR*p['Ka'])+(DR3*p['kdr'])
        dFR  = -(FR*p['Ka'])
        dDR  = -(DR*p['kdr'])
        dDR1 =  (DR*p['kdr'])-(DR1*p['kdr'])
        dDR2 =  (DR1*p['kdr'])-(DR2*p['kdr'])
        dDR3 =  (DR2*p['kdr'])-(DR3*p['kdr'])
        dR   = -(R*p['ka_SC'])
        E_drug = (p['Imax']*C**p['Gamma'])/(p['IC50']**p['Gamma']+C**p['Gamma']+1e-15)
        E_pbo  = p['Emax_PBO']*(1.0-np.exp(-p['k_pbo']*t))
        dBW    = p['kout']*100.0*(1.0-E_drug-E_pbo)-p['kout']*BW
        return [dA1,dFR,dDR,dDR1,dDR2,dDR3,dR,dBW]
    return ode

def run_ode(all_events, p, t_end):
    ode_fn = make_ode(p)
    bps    = sorted(set([0.0]+list(all_events.keys())+[float(t_end)]))
    y      = np.array([0.0]*7+[p['BW0']])
    all_t, all_y = [], []
    for i in range(len(bps)-1):
        t0,t1 = bps[i],bps[i+1]
        if t0 in all_events:
            R_a,FR_a,DR_a = all_events[t0]
            y[6]+=R_a; y[1]+=FR_a; y[2]+=DR_a
        n    = max(2,int(t1-t0)+1)
        t_ev = np.linspace(t0,t1,n)
        sol  = solve_ivp(ode_fn,[t0,t1],y.copy(),t_eval=t_ev,
                         method='LSODA',rtol=1e-7,atol=1e-10)
        if not sol.success: return None
        all_t.extend(sol.t[:-1].tolist())
        all_y.append(sol.y[:,:-1])
        y = sol.y[:,-1].copy()
    all_t.append(float(t_end)); all_y.append(y.reshape(-1,1))
    t_arr  = np.array(all_t); y_arr = np.hstack(all_y)
    C_ugL  = np.clip(y_arr[0],0,None)/p['V']
    BW     = np.clip(y_arr[7],0,None)
    BW_pct = (BW-p['BW0'])/p['BW0']*100.0
    GI_d   = np.clip(p['Emax_AE']*C_ugL/(p['EC50_AE']+C_ugL+1e-15),0,1)
    GI_t   = np.clip(p['E0_AE']+GI_d,0,1)
    return {"t_h":t_arr,"C_ugL":C_ugL,"BW_pct":BW_pct,
            "GI_total":GI_t*100,"GI_drug":GI_d*100}

def build_standard_events(coh_name, multiplier, p):
    meta = COHORT_META[coh_name]; skip = meta["skip"]
    all_events = {}
    for bi,(dose_ug,times) in enumerate(DEFAULT_WEGOVY_BLOCKS):
        if bi==skip: continue
        for t in times:
            all_events.setdefault(float(t),[0.0,0.0,0.0])
            all_events[float(t)][0] += dose_ug*p['F_SC']
    if meta["blk"] is not None:
        t_h  = float(DEFAULT_WEGOVY_BLOCKS[meta["blk"]][1][0])
        d_ug = meta["wegovy_dose_ug"]*multiplier
        F_FR = p['F_SC']-p['F_DR']
        all_events.setdefault(t_h,[0.0,0.0,0.0])
        all_events[t_h][1] += d_ug*p['Scale_LAI']*F_FR
        all_events[t_h][2] += d_ug*p['Scale_LAI']*p['F_DR']
    return all_events

@st.cache_data(show_spinner=False)
def run_standard(active_cohorts, multiplier, t_end, _ver):
    return {coh: run_ode(build_standard_events(coh, multiplier, P), P, t_end)
            for coh in active_cohorts}

def calc_window(t_h, C_ugL, BW_pct, GI_total, t_start, tau):
    mask = (t_h>=t_start)&(t_h<=t_start+tau)
    t_w,C_w = t_h[mask],C_ugL[mask]
    BW_w,GI_w = BW_pct[mask],GI_total[mask]
    if len(C_w)==0: return None
    AUCtau = float(trapezoid(C_w,t_w))
    return {
        "Cmax":    float(np.max(C_w)),
        "Tmax":    float(t_w[np.argmax(C_w)]),
        "AUCtau":  AUCtau,
        "Cavg":    AUCtau/tau,
        "Clast":   float(C_w[-1]),
        "dBW":     float(BW_w[-1]-BW_w[0]) if len(BW_w)>0 else 0.0,
        "peak_GI": float(np.max(GI_w)) if len(GI_w)>0 else 0.0,
    }

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sb-logo">💊 DWJ1691 Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<span style="font-size:0.78rem;color:#93c5fd;">'
        'Minipig PK → Human 예측 · Phoenix NLME</span>',
        unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#2d4a6e;margin:10px 0">', unsafe_allow_html=True)

    st.markdown('<div class="sb-hdr">코호트 선택 / COHORT</div>', unsafe_allow_html=True)
    sel = {coh: st.checkbox(SHORT_NAMES[coh], value=True, key=f"chk_{coh}")
           for coh in COHORT_COLORS}

    st.markdown('<hr style="border-color:#2d4a6e;margin:10px 0">', unsafe_allow_html=True)
    st.markdown('<div class="sb-hdr">시험약 용량 배수 / DOSE MULTIPLIER</div>',
                unsafe_allow_html=True)
    multiplier = st.slider("대조약(Wegovy) 대비 배수", 1, 16, 8, 1,
                           label_visibility="visible")

    def fmt_dose(ug):
        return f"{ug*multiplier/1000:.1f} mg ({ug*multiplier:,} µg)"

    st.markdown(
        f'<div class="dose-box">'
        f'<div class="dose-box-title">실제 투여 용량 ({multiplier}×)</div>'
        f'<div class="dose-row">Cohort I &nbsp;&nbsp;: <span class="dose-val">{fmt_dose(1000)}</span></div>'
        f'<div class="dose-row">Cohort II &nbsp;: <span class="dose-val">{fmt_dose(1700)}</span></div>'
        f'<div class="dose-row">Cohort III: <span class="dose-val">{fmt_dose(2400)}</span></div>'
        f'</div>',
        unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#2d4a6e;margin:10px 0">', unsafe_allow_html=True)
    st.markdown('<div class="sb-hdr">안전성 평가 기준 (GI AE 위험 허용 범위)</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<span style="font-size:0.76rem;color:#93c5fd;">'
        'Cmax 배수 기준: 시험약 / 대조약</span>',
        unsafe_allow_html=True)
    ni_margin = st.slider("허용 Cmax 배수", 0.5, 5.0, 2.0, 0.25,
                          label_visibility="visible")

    st.markdown('<hr style="border-color:#2d4a6e;margin:10px 0">', unsafe_allow_html=True)
    st.markdown(
        '<span style="font-size:0.76rem;color:#93c5fd;">'
        '📅 Phoenix NLME validated<br>'
        '🐷 Minipig PK 기반 인체 예측<br>'
        '👨‍🔬 Taeheon Kim, Ph.D. · 2026-02-19</span>',
        unsafe_allow_html=True)

# ============================================================
# MAIN TABS
# ============================================================
tab_std, tab_custom = st.tabs([
    "📊 표준 시뮬레이션 (Standard)",
    "⚙️ 커스텀 설계 (Custom Schedule)"
])

with tab_std:
    col_obs1, col_obs2 = st.columns([3,1])
    with col_obs1:
        sim_weeks = st.slider("전체 시뮬레이션 기간 (weeks)", 12, 36, 24, 1, key="std_sim_wk")
    with col_obs2:
        obs_weeks = st.slider("관찰 구간 τ (weeks)", 2, 8, 4, 1, key="std_obs_wk")
    tau_h = obs_weeks * HOURS_PER_WEEK
    t_end = sim_weeks * HOURS_PER_WEEK

    st.markdown(f"""
    <div class="main-header">
      <div class="main-title">DWJ1691 임상 1상 PK/PD 시뮬레이터
        <span class="badge">Phoenix NLME</span>
        <span class="badge">Validated ✓</span>
        <span class="badge-yellow">🐷 Minipig→Human</span>
      </div>
      <div class="main-title-ko">미니피그 PK 데이터 기반 인체 약동학 예측 · 위고비 병용 임상 설계 시뮬레이션</div>
      <div class="main-sub">
        1-Cpt · Multiple Absorption (LAI fast+delayed + Wegovy SC)
        · Indirect Response BW · Simple Emax GI AE
      </div>
      <div class="design-strip">
        <div class="design-item"><div class="design-dot" style="background:#94a3b8"></div>
          <span>Wegovy: 250→500→1000→1700→2400 µg q1w (4doses/block)</span></div>
        <div class="design-item"><div class="design-dot" style="background:#2166ac"></div>
          <span>Cohort I: <b>{1000*multiplier/1000:.1f}mg</b> @ h1344</span></div>
        <div class="design-item"><div class="design-dot" style="background:#16a34a"></div>
          <span>Cohort II: <b>{1700*multiplier/1000:.1f}mg</b> @ h2016</span></div>
        <div class="design-item"><div class="design-dot" style="background:#dc2626"></div>
          <span>Cohort III: <b>{2400*multiplier/1000:.1f}mg</b> @ h2688</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    active = [c for c,v in sel.items() if v]
    if not active:
        st.warning("코호트를 한 개 이상 선택해주세요.")
        st.stop()

    with st.spinner("🔬 ODE 시뮬레이션 실행 중..."):
        results = run_standard(tuple(active), multiplier, t_end, _ver="v9.0")
    results = {k:v for k,v in results.items() if v is not None}
    if not results:
        st.error("시뮬레이션 오류가 발생했습니다."); st.stop()

    ref_r = results.get("Reference (Wegovy)")

    # window 계산
    test_window, ref_window = {}, {}
    for coh in ["Cohort I (W-W-T-W-W)","Cohort II (W-W-W-T-W)","Cohort III (W-W-W-W-T)"]:
        t_s = COHORT_WINDOWS[coh]["t_dose"]
        if coh in results:
            test_window[coh] = calc_window(
                results[coh]['t_h'],results[coh]['C_ugL'],
                results[coh]['BW_pct'],results[coh]['GI_total'],t_s,tau_h)
        if ref_r is not None:
            ref_window[coh] = calc_window(
                ref_r['t_h'],ref_r['C_ugL'],
                ref_r['BW_pct'],ref_r['GI_total'],t_s,tau_h)

    all_C  = np.concatenate([r['C_ugL']   for r in results.values()])
    all_bw = np.concatenate([r['BW_pct']  for r in results.values()])
    all_gi = np.concatenate([r['GI_total']for r in results.values()])

    test_cmax = max((test_window[c]['Cmax'] for c in test_window if test_window[c]), default=0)
    first_test = next((c for c in ["Cohort I (W-W-T-W-W)","Cohort II (W-W-W-T-W)","Cohort III (W-W-W-W-T)"]
                       if c in ref_window and ref_window[c]), None)
    ref_cmax_kpi = ref_window[first_test]['Cmax'] if first_test else None
    cmax_ratio = test_cmax/ref_cmax_kpi if ref_cmax_kpi and test_cmax else 0
    ni_ok = 0 < cmax_ratio <= ni_margin

    # KPI
    k1,k2,k3,k4,k5 = st.columns(5)
    with k1:
        st.markdown(f"""<div class="kpi-card kpi-blue">
          <div class="kpi-label">전체 C<sub>max</sub></div>
          <div class="kpi-label-ko">전체 최고 혈중 농도</div>
          <div class="kpi-value cv-blue">{np.max(all_C):.1f}</div>
          <div class="kpi-unit">µg/L</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card kpi-purple">
          <div class="kpi-label">시험약 τ C<sub>max</sub></div>
          <div class="kpi-label-ko">투여 후 {obs_weeks}주 관찰구간</div>
          <div class="kpi-value cv-purple">{test_cmax:.1f}</div>
          <div class="kpi-unit">µg/L</div></div>""", unsafe_allow_html=True)
    with k3:
        ratio_disp = f"{cmax_ratio:.2f}×" if cmax_ratio>0 else "—"
        ni_icon    = "✅" if ni_ok else ("⚠️" if cmax_ratio>0 else "—")
        ratio_col  = "cv-green" if ni_ok else "cv-red"
        card_col   = "green" if ni_ok else "red"
        st.markdown(f"""<div class="kpi-card kpi-{card_col}">
          <div class="kpi-label">vs. Wegovy C<sub>max</sub></div>
          <div class="kpi-label-ko">허용 배수 기준 {ni_margin}×</div>
          <div class="kpi-value {ratio_col}">{ratio_disp}</div>
          <div class="kpi-unit">{ni_icon} 안전성 허용범위</div></div>""",
                    unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card kpi-green">
          <div class="kpi-label">Max BW Loss</div>
          <div class="kpi-label-ko">최대 체중 감소율</div>
          <div class="kpi-value cv-green">{np.min(all_bw):.1f}%</div>
          <div class="kpi-unit">BW₀ = 100 kg 기준</div></div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class="kpi-card kpi-orange">
          <div class="kpi-label">Peak GI AE</div>
          <div class="kpi-label-ko">최고 위장관 이상반응률</div>
          <div class="kpi-value cv-orange">{np.max(all_gi):.1f}%</div>
          <div class="kpi-unit">Total AE rate</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── PK Chart ──
    shade_colors = {
        "Cohort I (W-W-T-W-W)":   "rgba(33,102,172,0.08)",
        "Cohort II (W-W-W-T-W)":  "rgba(22,163,74,0.08)",
        "Cohort III (W-W-W-W-T)": "rgba(220,38,38,0.08)",
    }
    vline_info = {
        "Cohort I (W-W-T-W-W)":   (1344/HOURS_PER_WEEK,"#2166ac",f"{1000*multiplier/1000:.1f}mg"),
        "Cohort II (W-W-W-T-W)":  (2016/HOURS_PER_WEEK,"#16a34a",f"{1700*multiplier/1000:.1f}mg"),
        "Cohort III (W-W-W-W-T)": (2688/HOURS_PER_WEEK,"#dc2626",f"{2400*multiplier/1000:.1f}mg"),
    }

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sec-hdr">📈 PK Profile — 혈중 약물 농도 (µg/L)'
        f'<span class="window-badge">음영 = 시험약 관찰 구간 (Duration = {int(tau_h)}h)</span></div>',
        unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    fig_pk = go.Figure()
    for coh in active:
        if coh in shade_colors:
            t_s = COHORT_WINDOWS[coh]["t_dose"]/HOURS_PER_WEEK
            t_e = (COHORT_WINDOWS[coh]["t_dose"]+tau_h)/HOURS_PER_WEEK
            fig_pk.add_vrect(x0=t_s,x1=t_e,
                fillcolor=shade_colors[coh],line_width=0,
                annotation_text=f"{obs_weeks}wk obs",
                annotation_position="top left",
                annotation_font=dict(size=9,color=COHORT_COLORS[coh]))
    for coh,r in results.items():
        t_wk = r['t_h']/HOURS_PER_WEEK; s = max(1,len(t_wk)//2000)
        fig_pk.add_trace(go.Scatter(
            x=t_wk[::s],y=r['C_ugL'][::s],name=SHORT_NAMES[coh],
            line=dict(color=COHORT_COLORS[coh],width=2.5,dash=COHORT_DASH[coh]),
            hovertemplate=f"<b>{SHORT_NAMES[coh]}</b><br>Week:%{{x:.1f}}<br>Conc:%{{y:.1f}} µg/L<extra></extra>"
        ))
    for coh in active:
        if coh in vline_info:
            t_v,col,lbl = vline_info[coh]
            fig_pk.add_vline(x=t_v,line_dash="dash",line_color=col,
                             line_width=1.2,opacity=0.5,
                             annotation_text=f"DWJ {lbl}",annotation_position="top",
                             annotation_font=dict(size=10,color=col))
    fig_pk.update_layout(**make_chart_bg(), height=440, legend=LEGEND_STYLE)
    apply_axes(fig_pk, "Time (Week)", "Plasma concentration (µg/L)")
    st.plotly_chart(fig_pk, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── BW + GI ──
    col_bw,col_gi = st.columns(2)
    with col_bw:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">⚖️ 체중 변화 (Body Weight Change %BW)</div>',
                    unsafe_allow_html=True)
        fig_bw = go.Figure()
        for coh,r in results.items():
            t_wk = r['t_h']/HOURS_PER_WEEK; s = max(1,len(t_wk)//2000)
            fig_bw.add_trace(go.Scatter(
                x=t_wk[::s],y=r['BW_pct'][::s],name=SHORT_NAMES[coh],
                line=dict(color=COHORT_COLORS[coh],width=2.5,dash=COHORT_DASH[coh]),
                hovertemplate=f"<b>{SHORT_NAMES[coh]}</b><br>Week:%{{x:.1f}}<br>ΔBW:%{{y:.2f}}%<extra></extra>"
            ))
        fig_bw.add_hline(y=0,line_dash="dot",line_color="#94a3b8",line_width=1.2)
        fig_bw.update_layout(**make_chart_bg(),height=330,legend=LEGEND_STYLE)
        apply_axes(fig_bw, "Time (Week)", "ΔBW (%) from BW₀=100kg")
        st.plotly_chart(fig_bw, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_gi:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">⚠️ 위장관 이상반응률 (GI Adverse Event %)</div>',
                    unsafe_allow_html=True)
        fig_gi = go.Figure()
        for coh,r in results.items():
            t_wk = r['t_h']/HOURS_PER_WEEK; s = max(1,len(t_wk)//2000)
            fig_gi.add_trace(go.Scatter(
                x=t_wk[::s],y=r['GI_total'][::s],name=SHORT_NAMES[coh],
                line=dict(color=COHORT_COLORS[coh],width=2.5,dash=COHORT_DASH[coh]),
                hovertemplate=f"<b>{SHORT_NAMES[coh]}</b><br>Week:%{{x:.1f}}<br>GI AE:%{{y:.1f}}%<extra></extra>"
            ))
        fig_gi.update_layout(**make_chart_bg(),height=330,legend=LEGEND_STYLE)
        apply_axes(fig_gi, "Time (Week)", "GI AE Total (%)")
        fig_gi.update_yaxes(range=[0,100])
        st.plotly_chart(fig_gi, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 결과 테이블 ──
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sec-hdr">📊 관찰 구간 결과 요약'
        f'<span class="window-badge">Duration = {int(tau_h)}h ({obs_weeks}주)</span></div>',
        unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    coh_pairs = [
        ("Cohort I (W-W-T-W-W)",   "Wegovy (1.0 mg 구간)"),
        ("Cohort II (W-W-W-T-W)",  "Wegovy (1.7 mg 구간)"),
        ("Cohort III (W-W-W-W-T)", "Wegovy (2.4 mg 구간)"),
    ]
    rows = []
    for coh, ref_label in coh_pairs:
        dose_ug = COHORT_META[coh]['wegovy_dose_ug']*multiplier
        dose_mg = dose_ug/1000
        rw = ref_window.get(coh)
        if "Reference (Wegovy)" in results and rw:
            rows.append({
                "구분":            ref_label,
                "시험약 용량":     "— (대조약)",
                "Cmax (µg/L)":    round(rw['Cmax'],  1),
                "Tmax (h)":       round(rw['Tmax'],  1),
                "AUCτ (µg·h/L)": round(rw['AUCtau'],0),
                "Cavg (µg/L)":    round(rw['Cavg'],  3),
                "ΔBW τ (%)":      round(rw['dBW'],   2),
                "Peak GI (%)":    round(rw['peak_GI'],1),
                "안전성 판정":     "—",
            })
        if coh in results and coh in test_window and test_window[coh]:
            tw = test_window[coh]
            ni_sym = "✅" if (rw and tw['Cmax']/rw['Cmax']<=ni_margin) else "⚠️"
            rows.append({
                "구분":            SHORT_NAMES[coh],
                "시험약 용량":     f"{dose_mg:.1f}mg ({dose_ug:,}µg)",
                "Cmax (µg/L)":    round(tw['Cmax'],  1),
                "Tmax (h)":       round(tw['Tmax'],  1),
                "AUCτ (µg·h/L)": round(tw['AUCtau'],0),
                "Cavg (µg/L)":    round(tw['Cavg'],  3),
                "ΔBW τ (%)":      round(tw['dBW'],   2),
                "Peak GI (%)":    round(tw['peak_GI'],1),
                "안전성 판정":     ni_sym,
            })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True,
            column_config={
                "구분":             st.column_config.TextColumn(width="medium"),
                "시험약 용량":      st.column_config.TextColumn(width="medium"),
                "Cmax (µg/L)":      st.column_config.NumberColumn(format="%.1f"),
                "Tmax (h)":         st.column_config.NumberColumn(format="%.1f"),
                "AUCτ (µg·h/L)":   st.column_config.NumberColumn(format="%.0f"),
                "Cavg (µg/L)":      st.column_config.NumberColumn(format="%.3f"),
                "ΔBW τ (%)":        st.column_config.NumberColumn(format="%.2f"),
                "Peak GI (%)":      st.column_config.NumberColumn(format="%.1f"),
                "안전성 판정":       st.column_config.TextColumn(width="small"),
            })
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ 결과 다운로드 (CSV)", csv,
                           file_name=f"pkpd_{multiplier}x_{obs_weeks}wk.csv",
                           mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 2 ──
with tab_custom:
    st.markdown("""
    <div class="custom-info">
    ⚙️ <b>커스텀 투약 스케줄 설계</b> — 투약 시점, 용량, 약물 종류를 자유롭게 설정합니다.<br>
    <b>Wegovy SC</b> → R depot &nbsp;|&nbsp; <b>DWJ1691 LAI</b> → FR+DR depot
    </div>
    """, unsafe_allow_html=True)

    col_cs1,col_cs2 = st.columns([2,1])
    with col_cs1:
        st.markdown("#### 📋 투약 스케줄 입력")
        default_schedule = pd.DataFrame([
            {"시간 (h)": 0,    "용량 (µg)": 250,  "약물": "Wegovy SC"},
            {"시간 (h)": 168,  "용량 (µg)": 250,  "약물": "Wegovy SC"},
            {"시간 (h)": 336,  "용량 (µg)": 250,  "약물": "Wegovy SC"},
            {"시간 (h)": 504,  "용량 (µg)": 250,  "약물": "Wegovy SC"},
            {"시간 (h)": 672,  "용량 (µg)": 500,  "약물": "Wegovy SC"},
            {"시간 (h)": 840,  "용량 (µg)": 500,  "약물": "Wegovy SC"},
            {"시간 (h)": 1008, "용량 (µg)": 500,  "약물": "Wegovy SC"},
            {"시간 (h)": 1176, "용량 (µg)": 500,  "약물": "Wegovy SC"},
            {"시간 (h)": 1344, "용량 (µg)": 8000, "약물": "DWJ1691 LAI"},
            {"시간 (h)": 1512, "용량 (µg)": 1000, "약물": "Wegovy SC"},
            {"시간 (h)": 1680, "용량 (µg)": 1000, "약물": "Wegovy SC"},
            {"시간 (h)": 1848, "용량 (µg)": 1000, "약물": "Wegovy SC"},
        ])
        edited_df = st.data_editor(
            default_schedule, num_rows="dynamic", use_container_width=True,
            column_config={
                "시간 (h)":  st.column_config.NumberColumn("시간 (h)", min_value=0, step=168),
                "용량 (µg)": st.column_config.NumberColumn("용량 (µg)", min_value=0, step=100),
                "약물":      st.column_config.SelectboxColumn("약물 종류",
                    options=["Wegovy SC","DWJ1691 LAI"], required=True),
            }, key="custom_schedule")

    with col_cs2:
        st.markdown("#### ⚙️ 시뮬레이션 설정")
        cs_sim_wk  = st.slider("시뮬레이션 기간 (weeks)", 8, 40, 16, 1, key="cs_sim")
        cs_obs_t   = st.number_input("관찰 시작 시간 (h)", value=1344, step=168, key="cs_obs_t")
        cs_obs_dur = st.slider("관찰 기간 τ (weeks)", 2, 8, 4, 1, key="cs_obs_dur")
        cs_label   = st.text_input("시뮬레이션 이름", value="Custom Design", key="cs_label")
        run_btn    = st.button("▶ 커스텀 시뮬레이션 실행", key="run_custom_btn")

    if run_btn or "custom_result" in st.session_state:
        if run_btn:
            ev = {}
            for _,row in edited_df.iterrows():
                t_h = float(row["시간 (h)"]); d_ug = float(row["용량 (µg)"])
                ev.setdefault(t_h,[0.0,0.0,0.0])
                if row["약물"]=="Wegovy SC":
                    ev[t_h][0] += d_ug*P['F_SC']
                else:
                    F_FR = P['F_SC']-P['F_DR']
                    ev[t_h][1] += d_ug*P['Scale_LAI']*F_FR
                    ev[t_h][2] += d_ug*P['Scale_LAI']*P['F_DR']
            with st.spinner("🔬 커스텀 시뮬레이션 실행 중..."):
                cr = run_ode(ev, P, cs_sim_wk*HOURS_PER_WEEK)
            st.session_state.update({
                "custom_result":cr,"custom_events":ev,
                "cs_obs_t":cs_obs_t,"cs_tau":cs_obs_dur*HOURS_PER_WEEK,
                "cs_label":cs_label,
            })

        cr    = st.session_state.get("custom_result")
        c_obs = st.session_state.get("cs_obs_t", cs_obs_t)
        c_tau = st.session_state.get("cs_tau", cs_obs_dur*HOURS_PER_WEEK)
        c_lbl = st.session_state.get("cs_label", cs_label)

        if cr:
            st.markdown(f"---\n#### 📈 결과: **{c_lbl}**")
            t_wk_c = cr['t_h']/HOURS_PER_WEEK; s_c = max(1,len(t_wk_c)//2000)

            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="sec-hdr">📈 PK Profile (µg/L)'
                f'<span class="window-badge">Duration = {int(c_tau)}h</span></div>',
                unsafe_allow_html=True)
            fig_cpk = go.Figure()
            fig_cpk.add_vrect(x0=c_obs/HOURS_PER_WEEK,x1=(c_obs+c_tau)/HOURS_PER_WEEK,
                fillcolor="rgba(124,58,237,0.08)",line_width=0,
                annotation_text="관찰구간",annotation_font=dict(size=9,color="#7c3aed"))
            fig_cpk.add_trace(go.Scatter(
                x=t_wk_c[::s_c],y=cr['C_ugL'][::s_c],name=c_lbl,
                line=dict(color="#7c3aed",width=2.5),
                hovertemplate="Week:%{x:.1f}<br>Conc:%{y:.1f} µg/L<extra></extra>"))
            ev_cs = st.session_state.get("custom_events",{})
            for t_ev,amts in sorted(ev_cs.items()):
                col_ev = "#2166ac" if amts[0]>0 else "#dc2626"
                fig_cpk.add_vline(x=t_ev/HOURS_PER_WEEK,line_dash="dot",
                                  line_color=col_ev,line_width=1,opacity=0.4,
                                  annotation_text="W" if amts[0]>0 else "DWJ",
                                  annotation_font=dict(size=8,color=col_ev))
            fig_cpk.update_layout(**make_chart_bg(),height=380,legend=LEGEND_STYLE)
            apply_axes(fig_cpk,"Time (Week)","Plasma concentration (µg/L)")
            st.plotly_chart(fig_cpk, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            cc1,cc2 = st.columns(2)
            with cc1:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown('<div class="sec-hdr">⚖️ 체중 변화 (%BW)</div>', unsafe_allow_html=True)
                fig_cbw = go.Figure()
                fig_cbw.add_trace(go.Scatter(x=t_wk_c[::s_c],y=cr['BW_pct'][::s_c],
                    name=c_lbl,line=dict(color="#7c3aed",width=2.5),
                    hovertemplate="Week:%{x:.1f}<br>ΔBW:%{y:.2f}%<extra></extra>"))
                fig_cbw.add_hline(y=0,line_dash="dot",line_color="#94a3b8",line_width=1.2)
                fig_cbw.update_layout(**make_chart_bg(),height=300,showlegend=False)
                apply_axes(fig_cbw,"Time (Week)","ΔBW (%)")
                st.plotly_chart(fig_cbw, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with cc2:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown('<div class="sec-hdr">⚠️ GI 이상반응률 (%)</div>', unsafe_allow_html=True)
                fig_cgi = go.Figure()
                fig_cgi.add_trace(go.Scatter(x=t_wk_c[::s_c],y=cr['GI_total'][::s_c],
                    name=c_lbl,line=dict(color="#7c3aed",width=2.5),
                    hovertemplate="Week:%{x:.1f}<br>GI AE:%{y:.1f}%<extra></extra>"))
                fig_cgi.update_layout(**make_chart_bg(),height=300,showlegend=False)
                apply_axes(fig_cgi,"Time (Week)","GI AE Total (%)")
                fig_cgi.update_yaxes(range=[0,100])
                st.plotly_chart(fig_cgi, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            wp_c = calc_window(cr['t_h'],cr['C_ugL'],cr['BW_pct'],cr['GI_total'],c_obs,c_tau)
            if wp_c:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="sec-hdr">📊 관찰 구간 PK/PD 결과'
                    f'<span class="window-badge">t={c_obs:.0f}h ~ {c_obs+c_tau:.0f}h</span></div>',
                    unsafe_allow_html=True)
                r1,r2,r3,r4,r5 = st.columns(5)
                r1.metric("Cmax (µg/L)",    f"{wp_c['Cmax']:.2f}")
                r2.metric("Tmax (h)",        f"{wp_c['Tmax']:.1f}")
                r3.metric("AUCτ (µg·h/L)", f"{wp_c['AUCtau']:.0f}")
                r4.metric("Cavg (µg/L)",    f"{wp_c['Cavg']:.4f}")
                r5.metric("ΔBW τ (%)",      f"{wp_c['dBW']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)

# Model Info
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
with st.expander("📋 모델 파라미터 (Phoenix NLME fixef)"):
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**PK — 1-Cpt Multiple Absorption**")
        st.markdown(f"- V={P['V']}L, CL={P['Cl']}L/h")
        st.markdown(f"- Ka(FR→A1)={P['Ka']} h⁻¹, ka_SC(R→A1)={P['ka_SC']} h⁻¹")
        st.markdown(f"- F_SC={P['F_SC']}, Scale_LAI={P['Scale_LAI']}")
        st.markdown(f"- F_DR={P['F_DR']} → F_FR={P['F_SC']-P['F_DR']:.3f}, kdr={P['kdr']} h⁻¹")
    with c2:
        st.markdown("**BW PD — Indirect Response**")
        st.markdown("- kin=kout×100, E_drug=Imax·Cᵞ/(IC50ᵞ+Cᵞ)")
        st.markdown(f"- Imax={P['Imax']}, IC50={P['IC50']} µg/L, kout={P['kout']} h⁻¹")
    with c3:
        st.markdown("**GI AE / Validation**")
        st.markdown(f"- E₀={P['E0_AE']}, Emax={P['Emax_AE']}, EC50={P['EC50_AE']} µg/L")
        st.markdown("- t=1h: 0.5282 ✓ | t=1344h: 43.97 ✓ | t=2688h: 150.72 ✓")

st.markdown(f"""
<div style='text-align:center;color:#475569;font-size:0.78rem;
            padding:12px 0;margin-top:12px;background:#ffffff;
            border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,0.05)'>
  🐷 Minipig PK → Human 예측 · Phoenix NLME Validated ·
  Taeheon Kim, Ph.D. · 2026-02-19 · v9.0 · 현재 배수: {multiplier}×
</div>
""", unsafe_allow_html=True)
