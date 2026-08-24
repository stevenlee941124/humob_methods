"""
===============================================================================
HuMob 2026 Diffusion Exp - Baseline Visual Dashboard v3
===============================================================================
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import plotly.graph_objects as go

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
from exponential_baseline import compute_full_baseline, EXCLUDED_DATES, FLOW_THRESHOLD

OD_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DATES_PKL = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
CLASS_CSV = PACKAGE_ROOT / 'data' / 'processed' / 'grid_final_classification.csv'

st.set_page_config(
    page_title="HuMob 2026: Baseline 視覺化儀表板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 全年日期索引 ───────────────────────────────────────────────────
start_dt_global = datetime(2023, 11, 1)
CAL_DTS   = [start_dt_global + timedelta(days=i) for i in range(366)]
CAL_DATES = [d.strftime('%Y%m%d') for d in CAL_DTS]
CAL_DATE_TO_IDX = {d: i for i, d in enumerate(CAL_DATES)}

MIN_X, MAX_X = 30, 70
MIN_Y, MAX_Y = 35, 70
BBOX_GRIDS = {f"{x}_{y}" for x in range(MIN_X, MAX_X+1) for y in range(MIN_Y, MAX_Y+1)}

# 流量分級 bins (>= 2.0)
FLOW_BINS = [
    (2.0,   10.0,  "🟡 輕流量 [2, 10)"),
    (10.0,  50.0,  "🟠 中流量 [10, 50)"),
    (50.0,  200.0, "🔴 高流量 [50, 200)"),
    (200.0, 500.0, "🔵 樞紐流量 [200, 500)"),
    (500.0, 1e9,   "🟣 超級樞紐 [500+)"),
]

def classify_flow(mf):
    if mf < 2.0:
        return "⚫ 稀疏過濾 (<2.0)"
    for lo, hi, label in FLOW_BINS:
        if lo <= mf < hi:
            return label
    return "🟣 超級樞紐 [500+)"

# ── 資料載入（快取）───────────────────────────────────────────────
@st.cache_data(show_spinner="📦 載入 OD 數據...")
def load_core_data():
    with open(OD_PKL, 'rb') as f:
        od_ts = pickle.load(f)
    with open(DATES_PKL, 'rb') as f:
        dates_str = pickle.load(f)

    class_map = {}
    if CLASS_CSV.exists():
        df_cls = pd.read_csv(CLASS_CSV)
        class_map = dict(zip(df_cls['grid_id'], df_cls['final_class']))

    obs_idx = {d: i for i, d in enumerate(dates_str)}

    rows = []
    i_apr01 = CAL_DATE_TO_IDX['20240401']

    for key, raw_arr in od_ts.items():
        parts = key.split('-')
        if len(parts) != 2:
            continue
        o, d = parts[0], parts[1]
        y = np.full(366, np.nan)
        for date_s, oi in obs_idx.items():
            if date_s in CAL_DATE_TO_IDX:
                y[CAL_DATE_TO_IDX[date_s]] = raw_arr[oi]

        mf = float(np.nanmean(y)) if not np.all(np.isnan(y)) else 0.0
        post_mf = float(np.nanmean(y[i_apr01:])) if not np.all(np.isnan(y[i_apr01:])) else 0.0

        rows.append({
            "key": key, "origin": o, "dest": d,
            "is_diag": o == d,
            "in_bbox": o in BBOX_GRIDS and d in BBOX_GRIDS,
            "mean_flow": round(mf, 3),
            "post_mean_flow": round(post_mf, 3),
            "flow_class": classify_flow(mf),
            "qualifies": mf >= 2.0 and post_mf >= 2.0,
            "grid_class": class_map.get(d, "Unknown"),
        })

    df_catalog = pd.DataFrame(rows)
    return od_ts, obs_idx, class_map, df_catalog

def get_y366(od_ts, obs_idx, key):
    raw = od_ts[key]
    y = np.full(366, np.nan)
    for d, oi in obs_idx.items():
        if d in CAL_DATE_TO_IDX:
            y[CAL_DATE_TO_IDX[d]] = raw[oi]
    return y

def main():
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1B263B, #2C3E50); padding: 16px 24px;
                border-radius: 8px; margin-bottom: 18px; color: white;">
        <h2 style="margin: 0; color: #F1C40F;">📊 HuMob 2026 Baseline 視覺化儀表板</h2>
        <p style="margin: 6px 0 0 0; color: #BDC3C7; font-size: 14px;">
            真實人流（每日） vs. <b>宏觀極致平滑 Baseline</b>（1~3月純公式光滑指數恢復） | 流量門檻: <b>≥ 2.0</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    od_ts, obs_idx, class_map, df_catalog = load_core_data()

    # ── Sidebar ───────────────────────────────────────────────────
    st.sidebar.markdown("### 🎯 1. 流量等級")
    all_class_options = ["全部合格 (≥2.0)"] + [lbl for _, _, lbl in FLOW_BINS]
    sel_flow_class = st.sidebar.selectbox("選擇流量等級", all_class_options, index=0)

    st.sidebar.markdown("### 📂 2. OD 類型")
    pair_type = st.sidebar.radio("", ["對角線（留存）", "非對角線（跨區）", "全部"], index=0)

    only_bbox = st.sidebar.checkbox("只顯示 Bounding Box 內的網格", value=True)

    # 篩選 catalog
    df = df_catalog[df_catalog['qualifies']].copy()
    if sel_flow_class != "全部合格 (≥2.0)":
        df = df[df['flow_class'] == sel_flow_class]

    if pair_type == "對角線（留存）":
        df = df[df['is_diag']]
    elif pair_type == "非對角線（跨區）":
        df = df[~df['is_diag']]
    if only_bbox:
        df = df[df['in_bbox']]

    df = df.sort_values('mean_flow', ascending=False).reset_index(drop=True)
    st.sidebar.caption(f"符合條件：{len(df)} 條 OD pair (≥2.0)")

    if df.empty:
        st.warning("無符合條件的 OD pair，請調整篩選條件。")
        return

    # Destination → Origin 兩層選擇
    st.sidebar.markdown("### 🗺️ 3. 選擇網格")

    dest_options = (df.groupby('dest')['mean_flow']
                    .max()
                    .sort_values(ascending=False)
                    .index.tolist())
    sel_dest = st.sidebar.selectbox("Destination Grid", dest_options)

    origin_df = df[df['dest'] == sel_dest].sort_values('mean_flow', ascending=False)
    origin_options = origin_df['origin'].tolist()
    default_orig_idx = origin_options.index(sel_dest) if sel_dest in origin_options else 0
    sel_orig = st.sidebar.selectbox("Origin Grid", origin_options, index=default_orig_idx)

    od_key = f"{sel_orig}-{sel_dest}"

    # ── 時間範圍 ──────────────────────────────────────────────────
    st.sidebar.markdown("### 📅 4. 時間範圍")
    date_min = datetime(2023, 11, 1).date()
    date_max = datetime(2024, 10, 31).date()
    col1, col2 = st.sidebar.columns(2)
    with col1:
        date_start = st.date_input("起點", value=date_min, min_value=date_min, max_value=date_max)
    with col2:
        date_end = st.date_input("終點", value=date_max, min_value=date_min, max_value=date_max)

    if date_start >= date_end:
        st.sidebar.error("起點必須早於終點")
        return

    # ── 圖層開關 ──────────────────────────────────────────────────
    st.sidebar.markdown("### 🎛️ 5. 圖層開關")
    show_real = st.sidebar.checkbox("🔵 每日真實人流", value=True)
    show_base = st.sidebar.checkbox("🔴 Baseline（宏觀絲滑中軸）", value=True)
    show_excl = st.sidebar.checkbox("✖ 官方排除日標記", value=True)

    # ── 計算 Baseline ────────────────────────────────────────────
    if od_key not in od_ts:
        st.error(f"找不到 OD Pair: {od_key}")
        return

    y_366 = get_y366(od_ts, obs_idx, od_key)
    mean_flow = float(np.nanmean(y_366)) if not np.all(np.isnan(y_366)) else 0.0
    b_366, lam, grid_type = compute_full_baseline(y_366, CAL_DATES, CAL_DATE_TO_IDX)

    # ── 裁切時間範圍 ──────────────────────────────────────────────
    i0 = CAL_DATE_TO_IDX.get(date_start.strftime('%Y%m%d'), 0)
    i1 = CAL_DATE_TO_IDX.get(date_end.strftime('%Y%m%d'), 365)
    xs   = CAL_DTS[i0: i1+1]
    real = [float(v) if not np.isnan(v) else None for v in y_366[i0: i1+1]]
    base = [float(v) if not np.isnan(v) else None for v in b_366[i0: i1+1]]

    # ── Plotly 圖 ─────────────────────────────────────────────────
    fig = go.Figure()

    def vrect(lo_str, hi_str, color, label):
        lo = CAL_DTS[CAL_DATE_TO_IDX[lo_str]]
        hi = CAL_DTS[CAL_DATE_TO_IDX[hi_str]]
        if lo <= CAL_DTS[i1] and hi >= CAL_DTS[i0]:
            fig.add_vrect(x0=max(lo, CAL_DTS[i0]), x1=min(hi, CAL_DTS[i1]),
                          fillcolor=color, line_width=0,
                          annotation_text=label, annotation_position="top left",
                          annotation=dict(font_size=10, font_color='gray'))

    vrect('20231101', '20231231', "rgba(70,130,180,0.07)",  "震前 11~12月")
    vrect('20240101', '20240131', "rgba(255,165,0,0.10)",   "1月（衝擊槽底）")
    vrect('20240201', '20240331', "rgba(220,60,60,0.08)",   "2~3月盲區（光滑指數外推）")
    vrect('20240401', '20241031', "rgba(46,160,90,0.07)",   "4~10月（宏觀真實中軸）")

    if show_real:
        fig.add_trace(go.Scatter(
            x=xs, y=real, mode='lines+markers',
            name='🔵 每日真實人流',
            line=dict(color='royalblue', width=1.4),
            marker=dict(size=3, color='royalblue', opacity=0.5),
            connectgaps=False
        ))

    if show_base:
        fig.add_trace(go.Scatter(
            x=xs, y=base, mode='lines',
            name='🔴 Baseline（宏觀絲滑中軸）',
            line=dict(color='crimson', width=2.8),
            connectgaps=False
        ))

    if show_excl:
        excl_x, excl_y = [], []
        for excl in EXCLUDED_DATES:
            if excl in CAL_DATE_TO_IDX:
                ei = CAL_DATE_TO_IDX[excl]
                if i0 <= ei <= i1 and not np.isnan(y_366[ei]):
                    excl_x.append(CAL_DTS[ei])
                    excl_y.append(float(y_366[ei]))
        if excl_x:
            fig.add_trace(go.Scatter(
                x=excl_x, y=excl_y, mode='markers',
                name='✖ 官方排除日',
                marker=dict(symbol='x', size=9, color='black', line_width=2)
            ))

    c_name = class_map.get(sel_dest, 'Unknown')
    half_life = f"{0.693/lam:.1f} 天" if lam > 0 else "N/A"

    fig.update_layout(
        title=dict(
            text=f"<b>OD Pair [{od_key}]</b> | Grid Class: <span style='color:#E67E22'>{c_name}</span>"
                 f" | Dynamics: {grid_type} | λ={lam:.4f} (T½={half_life})"
                 f" | mean={mean_flow:.2f} | {classify_flow(mean_flow)}",
            font=dict(size=15, color='#2C3E50')
        ),
        xaxis=dict(title="日期", showgrid=True, gridcolor="rgba(200,200,200,0.3)",
                   range=[CAL_DTS[i0], CAL_DTS[i1]]),
        yaxis=dict(title="人流量 (人/天)", showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,0.85)"),
        hovermode="x unified",
        height=560,
        margin=dict(l=45, r=30, t=80, b=45)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── 數值摘要表 ────────────────────────────────────────────────
    st.subheader("📋 關鍵時間點數值摘要")
    checkpoints = [
        ('2023-11-15','20231115'), ('2023-12-15','20231215'),
        ('2024-01-07','20240107'), ('2024-01-15','20240115'),
        ('2024-01-31','20240131'), ('2024-02-15','20240215'),
        ('2024-03-15','20240315'), ('2024-04-01','20240401'),
        ('2024-05-01','20240501'), ('2024-07-01','20240701'),
    ]
    rows = []
    for label, d in checkpoints:
        if d not in CAL_DATE_TO_IDX: continue
        idx = CAL_DATE_TO_IDX[d]
        real_v = y_366[idx]
        base_v = b_366[idx]
        rows.append({
            "日期": label,
            "真實值": f"{real_v:.3f}" if not np.isnan(real_v) else "NaN",
            "Baseline": f"{base_v:.3f}" if not np.isnan(base_v) else "NaN",
            "殘差 (真實 - Baseline)": f"{real_v - base_v:.3f}" if not np.isnan(real_v) and not np.isnan(base_v) else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
