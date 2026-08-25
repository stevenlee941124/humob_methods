import sys
import os
import json
import math
import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="HuMob 2026: GARCH 移動性預測儀表板",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

PACKAGE_ROOT = Path(__file__).resolve().parent

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

def load_core_data():
    p_ts = PACKAGE_ROOT / "data" / "processed" / "od_time_series.pkl"
    p_dt = PACKAGE_ROOT / "data" / "processed" / "dates.pkl"
    with open(p_ts, 'rb') as f:
        od_ts = pickle.load(f)
    with open(p_dt, 'rb') as f:
        dates_str = pickle.load(f)

    # Classification
    p_cls = PACKAGE_ROOT / "data" / "processed" / "grid_final_classification.csv"
    df_cls = pd.read_csv(p_cls) if p_cls.exists() else pd.DataFrame()
    class_map = dict(zip(df_cls['grid_id'], df_cls['final_class'])) if not df_cls.empty else {}

    # Top Error ODs
    p_top_od = PACKAGE_ROOT / "data" / "processed" / "isolated_top_error_od_pairs.csv"
    df_top_ods = pd.read_csv(p_top_od) if p_top_od.exists() else None

    # Load TSVs
    def parse_tsv(filepath):
        data = {}
        filepath = Path(filepath)
        if not filepath.exists(): return data
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                pts = line.strip().split('\t')
                if len(pts) >= 2:
                    try:
                        raw = pts[1].replace(': NA', ': None').replace(':NA', ':None')
                        od = eval(raw, {'__builtins__': {}}, {'None': None})
                        if od is not None: data[pts[0]] = od
                    except: pass
        return data

    p_garch = PACKAGE_ROOT / "data" / "outputs" / "wave_garch_fullyear_holiday_garch.tsv"
    p_base = PACKAGE_ROOT / "data" / "outputs" / "gap90_midpoint_centerline_baseline.tsv"

    garch_data = parse_tsv(p_garch)
    base_data = parse_tsv(p_base)

    idx_apr1 = dates_str.index('20240401') if '20240401' in dates_str else 92
    idx_apr30 = dates_str.index('20240430') if '20240430' in dates_str else 121

    return od_ts, dates_str, class_map, df_top_ods, garch_data, base_data, idx_apr1, idx_apr30

def in_eval_bbox(g_str):
    if g_str == '-1_-1': return False
    pts = g_str.split('_')
    if len(pts) != 2: return False
    try:
        gx, gy = int(pts[0]), int(pts[1])
        return (30 <= gx <= 70) and (35 <= gy <= 70)
    except:
        return False

def main():
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1B263B, #2C3E50); padding: 18px 24px; border-radius: 8px; margin-bottom: 20px; color: white;">
        <h2 style="margin: 0; color: #F1C40F;">👑 HuMob 2026: 全年去污染 GARCH 模型預測儀表板</h2>
        <p style="margin: 6px 0 0 0; color: #BDC3C7; font-size: 14px;">
            Hermite Spline 宏觀平滑中軸 + 日曆去污染 7D 週期載波 ψ + GARCH(1,1) 動態條件異方差
        </p>
    </div>
    """, unsafe_allow_html=True)

    od_ts, dates_str, class_map, df_top_ods, garch_data, base_data, idx_apr1, idx_apr30 = load_core_data()

    # ── Top Scoreboard ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🏆 Combined NRMSE", "0.34558", "GARCH(1,1) 動力學模型")
    with c2:
        st.metric("停留流動 (Diag) NRMSE", "0.16274", "RMSE 4.32 人")
    with c3:
        st.metric("跨區流動 (Off-Diag) NRMSE", "0.52843", "RMSE 0.0093 人")
    with c4:
        st.metric("🎯 官方評測範圍", "1,476 網格", "x:30~70, y:35~70 (1,836 路線)")

    # ── Sidebar Controls ──
    st.sidebar.markdown("### 🎯 路線與分類導航")

    # 0. 空間範圍
    scope_options = [
        "🎯 官方評測範圍 (x: 30~70, y: 35~70)",
        "🌐 全域所有路線 (含邊緣與外部)"
    ]
    sel_scope = st.sidebar.selectbox("🗺️ 0. 選擇空間範圍 (Spatial Scope)", scope_options, index=0)

    # 1. 導航模式 / 類別
    nav_mode = st.sidebar.selectbox("📂 1. 導航模式", ["🔥 Top 20 高誤差關鍵樞紐", "📁 按 9 大動力學類別探索", "🌟 全域路線自選"], index=0)

    # 2. 流量等級過濾
    flow_options = [
        "🌟 全部流量規模 (All Scales)",
        "👑 樞紐巨量 (Hub, ≥500人)",
        "🚄 大量走廊 (Large, 200~500人)",
        "🚗 中量流通 (Medium, 50~200人)",
        "🚲 小量分支 (Small, 10~50人)",
        "🚶 微量流動 (Micro, 1~10人)",
        "⚪ 極低/散粒 (Ultra-Low, <1人)"
    ]
    sel_flow = st.sidebar.selectbox("📊 2. 選擇流量分類 (Flow Scale)", flow_options, index=0)

    # 計算訓練集天數均值
    train_days_idx = [i for i, d in enumerate(dates_str) if d < '20240101' or d >= '20240501']

    # 篩選候選 OD
    filtered_pairs = []
    top20_keys = [
        '41_47-41_47', '21_60-21_60', '11_25-11_25', '24_63-24_63', '10_23-10_21',
        '10_23-10_23', '11_28-11_28', '10_27-10_27', '35_48-35_48', '37_48-37_48',
        '40_47-40_47', '41_48-41_48', '42_47-42_47', '39_47-39_47', '43_47-43_47',
        '38_48-38_48', '40_48-40_48', '36_48-36_48', '42_48-42_48', '44_47-44_47'
    ]

    for k, raw in od_ts.items():
        if "官方評測" in sel_scope:
            if k.startswith('-1_-1-'): o, d = '-1_-1', k[6:]
            elif k.endswith('--1_-1'): o, d = k[:-6], '-1_-1'
            else: pts = k.split('-'); o, d = pts[0], pts[1]
            if not (in_eval_bbox(o) and in_eval_bbox(d)):
                continue

        if "Top 20" in nav_mode and k not in top20_keys:
            continue

        if "全部" not in sel_flow:
            y_train = [raw[oi] if not np.isnan(raw[oi]) else 0.0 for oi in train_days_idx]
            mean_f = float(np.mean(y_train))
            max_f = float(np.max(y_train))

            if "Hub" in sel_flow and not (mean_f >= 500.0 or max_f >= 1000.0): continue
            elif "Large" in sel_flow and not ((200.0 <= mean_f < 500.0) or (500.0 <= max_f < 1000.0)): continue
            elif "Medium" in sel_flow and not ((50.0 <= mean_f < 200.0) or (150.0 <= max_f < 500.0)): continue
            elif "Small" in sel_flow and not ((10.0 <= mean_f < 50.0) or (30.0 <= max_f < 150.0)): continue
            elif "Micro" in sel_flow and not ((1.0 <= mean_f < 10.0) or (5.0 <= max_f < 30.0)): continue
            elif "Ultra-Low" in sel_flow and not (mean_f < 1.0 and max_f < 5.0): continue

        filtered_pairs.append(k)

    st.sidebar.caption(f"🔍 符合篩選條件路線數: **{len(filtered_pairs):,} 條**")

    # 建立 出發地 -> 目的地 映射
    orig_to_dests = {}
    for k in filtered_pairs:
        if k.startswith('-1_-1-'): o, d = '-1_-1', k[6:]
        elif k.endswith('--1_-1'): o, d = k[:-6], '-1_-1'
        else: pts = k.split('-'); o, d = pts[0], pts[1]
        orig_to_dests.setdefault(o, []).append(d)

    all_origs = sorted(orig_to_dests.keys())
    if not all_origs:
        st.sidebar.warning("⚠️ 該組合下無匹配路線，請放寬篩選條件")
        st.stop()

    if '41_47' in all_origs: def_o_idx = all_origs.index('41_47')
    elif '21_60' in all_origs: def_o_idx = all_origs.index('21_60')
    elif '25_60' in all_origs: def_o_idx = all_origs.index('25_60')
    elif '11_28' in all_origs: def_o_idx = all_origs.index('11_28')
    else: def_o_idx = 0

    sel_orig = st.sidebar.selectbox("🚩 3. 選擇起點 / 出發地 (Origin)", all_origs, index=def_o_idx)
    valid_dests = sorted(orig_to_dests.get(sel_orig, []))
    def_d_idx = valid_dests.index(sel_orig) if sel_orig in valid_dests else 0
    sel_dest = st.sidebar.selectbox("🎯 4. 選擇終點 / 目的地 (Destination)", valid_dests, index=def_d_idx)

    od_key = f"-1_-1-{sel_dest}" if sel_orig == '-1_-1' else (f"{sel_orig}--1_-1" if sel_dest == '-1_-1' else f"{sel_orig}-{sel_dest}")

    st.sidebar.divider()
    st.sidebar.markdown("### 🎛️ 圖層顯示開關")
    show_gt = st.sidebar.checkbox("🔴 全年真實流量 (Ground Truth)", True)
    show_apr_gt = st.sidebar.checkbox("🔥 四月真實答案 (Ground Truth 標記)", True)
    show_base = st.sidebar.checkbox("📏 宏觀平滑中軸 Baseline", True)
    show_garch = st.sidebar.checkbox("👑 全年度去污染 GARCH 預測", True)

    if od_key not in od_ts:
        st.error(f"找不到 OD Pair: {od_key}")
        return

    ts_orig = od_ts[od_key]
    c_name = class_map.get(sel_dest, 'Unknown')

    start_dt = datetime(2023, 11, 1)
    end_dt = datetime(2024, 10, 31)
    total_days = (end_dt - start_dt).days + 1
    cal_dts_366 = [start_dt + timedelta(days=i) for i in range(total_days)]
    cal_dates_366 = [d.strftime("%Y%m%d") for d in cal_dts_366]

    gt_dict = dict(zip(dates_str, ts_orig))
    ts_masked_366 = []
    for d_str in cal_dates_366:
        if d_str in EXCLUDED_DATES:
            ts_masked_366.append(np.nan)
        else:
            val = gt_dict.get(d_str, np.nan)
            ts_masked_366.append(val if not np.isnan(val) else np.nan)

    # 4月真值
    apr_dts, apr_vals = [], []
    for i in range(idx_apr1, idx_apr30 + 1):
        if i < len(dates_str):
            d_str = dates_str[i]
            if d_str not in EXCLUDED_DATES:
                apr_dts.append(datetime.strptime(d_str, "%Y%m%d"))
                apr_vals.append(ts_orig[i])

    # GARCH 與 Baseline 預測
    gap90_dts = [datetime(2024, 2, 1) + timedelta(days=i) for i in range(90)]
    gap90_dates = [d.strftime("%Y%m%d") for d in gap90_dts]

    garch_preds, base_preds = [], []
    for d_str in gap90_dates:
        garch_val = garch_data.get(d_str, {}).get(sel_orig, {}).get(sel_dest, 0.0)
        base_val = base_data.get(d_str, {}).get(sel_orig, {}).get(sel_dest, 0.0)
        garch_preds.append(garch_val)
        base_preds.append(base_val)

    fig = go.Figure()

    if show_gt:
        fig.add_trace(go.Scatter(
            x=cal_dts_366, y=ts_masked_366,
            mode='lines',
            line=dict(color='rgba(231, 76, 60, 0.85)', width=1.5),
            name='🔴 全年真實流量 (Ground Truth)',
            connectgaps=False
        ))

    if show_apr_gt and apr_dts:
        fig.add_trace(go.Scatter(
            x=apr_dts, y=apr_vals,
            mode='lines',
            line=dict(color='rgba(192, 57, 43, 1.0)', width=2.8),
            name='🔥 四月真實答案 (Ground Truth 標記)',
            connectgaps=False
        ))

    if show_base:
        fig.add_trace(go.Scatter(
            x=gap90_dts, y=base_preds,
            mode='lines',
            line=dict(color='#7F8C8D', width=2.0, dash='dash'),
            name='📏 宏觀平滑中軸 Baseline'
        ))

    if show_garch:
        fig.add_trace(go.Scatter(
            x=gap90_dts, y=garch_preds,
            mode='lines',
            line=dict(color='#F1C40F', width=2.5),
            name='👑 全年度去污染 GARCH'
        ))

    fig.add_vrect(x0="2024-01-01", x1="2024-01-31", fillcolor="rgba(231, 76, 60, 0.08)", line_width=0,
                  annotation_text="1月 地震衝擊期", annotation_position="top left")
    fig.add_vrect(x0="2024-02-01", x1="2024-04-30", fillcolor="rgba(241, 196, 15, 0.08)", line_width=0,
                  annotation_text="2~4月 90天盲區預報 (4月官方評測)", annotation_position="top left")

    fig.update_layout(
        title=f"<b>OD 路線時序檢視: {od_key}</b> (起點: {sel_orig} ➔ 終點: {sel_dest}) | 網格動力學類別: <b>{c_name}</b>",
        xaxis=dict(title="日期 (2023/11 ~ 2024/10)", showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(title="人口流動量 (人次)", showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
        height=520,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
