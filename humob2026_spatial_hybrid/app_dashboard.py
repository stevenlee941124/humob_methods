"""
===============================================================================
HuMob 2026: Spatial-Temporal Hybrid Gated Diffusion Interactive Dashboard
===============================================================================
"""
import sys, pickle, numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from pathlib import Path
from datetime import datetime, timedelta

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / 'src'))
from japan_calendar import JAPAN_HOLIDAYS, get_holiday_features

st.set_page_config(page_title="HuMob 2026 Spatial Hybrid Gated Diffusion", layout="wide", page_icon="🔮")

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

OD_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DT_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
BASE_PKL  = PACKAGE_ROOT / 'data' / 'outputs' / 'hybrid_base_and_gates.pkl'
PRED_TSV  = PACKAGE_ROOT / 'data' / 'outputs' / 'hybrid_predictions.tsv'
FIELD_NPZ = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_field_90d.npz'

def load_base_structures():
    with open(OD_PKL, 'rb') as f: od_ts = pickle.load(f)
    with open(DT_PKL, 'rb') as f: dates_str = pickle.load(f)
    hybrid_models = pickle.load(open(BASE_PKL, 'rb')) if BASE_PKL.exists() else {}
    field_data = dict(np.load(FIELD_NPZ)) if FIELD_NPZ.exists() else None
    return od_ts, dates_str, hybrid_models, field_data

od_ts, dates_str, hybrid_models, field_data = load_base_structures()

# 動態即時讀取最新預測 TSV
def load_latest_predictions():
    pred_data = {}
    if PRED_TSV.exists():
        with open(PRED_TSV, 'r', encoding='utf-8') as f:
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 2:
                    raw = p[1].replace(': NA', ': None').replace(':NA', ':None')
                    od = eval(raw, {'__builtins__': {}}, {'None': None})
                    if od is not None: pred_data[p[0]] = od
    return pred_data

pred_data = load_latest_predictions()

st.title("🔮 HuMob 2026: Spatial-Temporal Hybrid Gated Diffusion 儀表板")
st.markdown("結合 **「零值真實感知 Baseline」+「超低流量前置分離」+「自適應通勤載波 $\psi$」+「2D 空間地理擴散」**。")

# ── 頂部模型指標看板 ──────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🏆 最新 Combined NRMSE", "0.31146", delta="-0.160 vs 純Diffusion", delta_color="inverse")
with c2:
    st.metric("👔 高/中規律通勤走廊", "1,092 條", "全開/門控 7D ψ 齒輪")
with c3:
    st.metric("🚗 偶發中低流量路線", "8,963 條", "稀疏分位自適應貼地")
with c4:
    st.metric("⬛ 超低流量/死寂孤島", "5,074 條 (33.5%)", "前置分離，100% 貼 0")

tab1, tab2 = st.tabs(["📈 OD 路線三層波形分解檢視", "🗺️ 2D 全域空間地理擴散熱力圖"])

with tab1:
    st.sidebar.header("🎯 路線與分類二維導航")

    # 1. 規律度分類選單
    reg_options = [
        "🌟 全部規律度 (All)",
        "🔥 Top 20 高誤差關鍵樞紐 (20 條)",
        "👔 高規律度通勤路線 (S_reg ≥ 0.45, 完整 ψ) [232 條]",
        "🚗 中規律度生活走廊 (0.20 ≤ S_reg < 0.45, 門控 ψ) [865 條]",
        "⚪ 低規律/散粒噪聲網格 (S_reg < 0.20) [14,032 條]"
    ]
    sel_reg = st.sidebar.selectbox("📂 1. 選擇規律度分類 (Regularity)", reg_options, index=0)

    # 2. 流量規模分類選單
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

    # 兩級交叉篩選
    filtered_pairs = []
    top20_keys = [
        '41_47-41_47', '21_60-21_60', '11_25-11_25', '24_63-24_63', '10_23-10_21',
        '10_23-10_23', '11_28-11_28', '10_27-10_27', '35_48-35_48', '37_48-37_48',
        '40_47-40_47', '41_48-41_48', '42_47-42_47', '39_47-39_47', '43_47-43_47',
        '38_48-38_48', '40_48-40_48', '36_48-36_48', '42_48-42_48', '44_47-44_47'
    ]

    for k, raw in od_ts.items():
        m = hybrid_models.get(k, {})
        reg_cat = m.get('category', 'Low_Regularity_Noise')

        # 規律度過濾
        if "Top 20" in sel_reg:
            if k not in top20_keys: continue
        elif "高規律度" in sel_reg:
            if reg_cat != 'High_Regularity_Commuter': continue
        elif "中規律度" in sel_reg:
            if reg_cat != 'Medium_Regularity_Corridor': continue
        elif "低規律" in sel_reg:
            if reg_cat != 'Low_Regularity_Noise': continue

        # 流量規模過濾
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

    # 建立 目的地 -> 出發地 映射
    dest_to_origs = {}
    for k in filtered_pairs:
        if k.startswith('-1_-1-'): o, d = '-1_-1', k[6:]
        elif k.endswith('--1_-1'): o, d = k[:-6], '-1_-1'
        else: pts = k.split('-'); o, d = pts[0], pts[1]
        dest_to_origs.setdefault(d, []).append(o)

    all_dests = sorted(dest_to_origs.keys())
    if not all_dests:
        st.sidebar.warning("⚠️ 該二維組合下無匹配路線，請放寬篩選條件")
        st.stop()

    # 智慧預設目的地
    if '41_47' in all_dests: def_d_idx = all_dests.index('41_47')
    elif '21_60' in all_dests: def_d_idx = all_dests.index('21_60')
    elif '11_28' in all_dests: def_d_idx = all_dests.index('11_28')
    else: def_d_idx = 0

    # 3. 選擇目的地
    sel_dest = st.sidebar.selectbox("🎯 3. 選擇目的地 (Destination)", all_dests, index=def_d_idx)

    # 4. 選擇出發地 (自動連動)
    valid_origs = sorted(dest_to_origs.get(sel_dest, []))
    def_o_idx = valid_origs.index(sel_dest) if sel_dest in valid_origs else 0
    sel_orig = st.sidebar.selectbox("🚩 4. 選擇出發地 (Origin)", valid_origs, index=def_o_idx)

    sel_od = f"-1_-1-{sel_dest}" if sel_orig == '-1_-1' else (f"{sel_orig}--1_-1" if sel_dest == '-1_-1' else f"{sel_orig}-{sel_dest}")

    # 圖層開關
    st.sidebar.subheader("圖層顯示開關")
    show_gt = st.sidebar.checkbox("🔴 全年真實流量 (Ground Truth)", True)
    show_apr = st.sidebar.checkbox("🔥 四月真實答案 (Ground Truth 標記)", True)
    show_base = st.sidebar.checkbox("📏 Layer 1 宏觀物理 Baseline", True)
    show_psi = st.sidebar.checkbox("⚡ Layer 2 確定性通勤載波 (Base + ψ)", True)
    show_pred = st.sidebar.checkbox("🔮 最終三層合成預測時序", True)

    m_info = hybrid_models.get(sel_od, {})
    s_reg = m_info.get('s_reg', 0.0)
    gate_g = m_info.get('gate_g', 0.0)
    cat_name = m_info.get('category', 'Unknown')
    b_366 = m_info.get('b_366', np.full(366, np.nan))
    c_func = m_info.get('carrier_func')
    sig_w = m_info.get('sigma_weekly', 0.0)

    st.markdown(f"**當前路線**: `{sel_od}` | **分類**: `{cat_name}` | **規律度評分 (S_reg)**: `{s_reg:.3f}` | **門控開度 (G_i)**: `{gate_g:.3f}` | **週波振幅 (σ_w)**: `{sig_w:.2f} 人`")

    # 構建 366 天時間軸
    start_dt = datetime(2023, 11, 1)
    cal_dts = [start_dt + timedelta(days=i) for i in range(366)]
    cal_dates = [d.strftime("%Y%m%d") for d in cal_dts]
    cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}

    raw_arr = od_ts.get(sel_od, np.full(len(dates_str), np.nan))
    gt_map = dict(zip(dates_str, raw_arr))

    ts_gt_masked = []
    for d_str in cal_dates:
        if ('20240201' <= d_str <= '20240430') or d_str in EXCLUDED_DATES:
            ts_gt_masked.append(np.nan)
        elif d_str not in gt_map:
            ts_gt_masked.append(0.0)
        else:
            v = gt_map[d_str]
            ts_gt_masked.append(0.0 if np.isnan(v) else float(v))

    apr_dts = [datetime(2024, 4, 1) + timedelta(days=i) for i in range(30)]
    apr_gt = []
    for dt in apr_dts:
        d_str = dt.strftime("%Y%m%d")
        if d_str in EXCLUDED_DATES:
            apr_gt.append(np.nan)
        elif d_str not in gt_map:
            apr_gt.append(0.0)
        else:
            v = gt_map[d_str]
            apr_gt.append(0.0 if np.isnan(v) else float(v))

    gap90_dts = [datetime(2024, 2, 1) + timedelta(days=i) for i in range(90)]
    gap90_dates = [d.strftime("%Y%m%d") for d in gap90_dts]

    # Layer 2 載波時序
    psi_series = []
    for dt in cal_dts:
        ci = cal_date_to_idx[dt.strftime("%Y%m%d")]
        b_val = b_366[ci] if not np.isnan(b_366[ci]) else 0.0
        dow = dt.weekday()
        psi_val = float(c_func(14.0 + dow)) if c_func is not None else 0.0
        psi_series.append(max(0.0, b_val + gate_g * psi_val * sig_w))

    # 最終預測時序
    orig_s, dest_s = (sel_od.split('-')[0], sel_od.split('-')[1]) if not sel_od.startswith('-1_-1-') else ('-1_-1', sel_od.replace('-1_-1-', ''))
    pred_flows = []
    for d_str in gap90_dates:
        if d_str in EXCLUDED_DATES:
            pred_flows.append(np.nan)
        else:
            v = pred_data.get(d_str, {}).get(orig_s, {}).get(dest_s, 0.0)
            pred_flows.append(float(v) if not np.isnan(v) else 0.0)

    # 繪圖
    fig = go.Figure()
    if show_gt:
        fig.add_trace(go.Scatter(x=cal_dts, y=ts_gt_masked, mode='lines', name='🔴 全年真實流量 (Ground Truth)', line=dict(color='#E53935', width=2.0), connectgaps=False))
    if show_apr:
        fig.add_trace(go.Scatter(x=apr_dts, y=apr_gt, mode='lines', name='🔥 四月真實答案 (Ground Truth 標記)', line=dict(color='#D50000', width=3.2), connectgaps=False))
    if show_base:
        fig.add_trace(go.Scatter(x=cal_dts, y=b_366, mode='lines', name='📏 Layer 1 宏觀物理 Baseline', line=dict(color='#7F8C8D', width=2.2, dash='dash'), connectgaps=True))
    if show_psi:
        fig.add_trace(go.Scatter(x=cal_dts, y=psi_series, mode='lines', name='⚡ Layer 2 確定性通勤載波 (Base + ψ)', line=dict(color='#FF9800', width=2.0, dash='dot'), connectgaps=True))
    if show_pred:
        fig.add_trace(go.Scatter(x=gap90_dts, y=pred_flows, mode='lines', name='🔮 最終三層合成預測時序 (Layer 1+2+3)', line=dict(color='#00BCD4', width=2.8), connectgaps=True))

    fig.add_vrect(x0=datetime(2024, 1, 1), x1=datetime(2024, 1, 31), fillcolor="red", opacity=0.08, line_width=0, annotation_text="1月 地震衝擊期", annotation_position="top left")
    fig.add_vrect(x0=datetime(2024, 2, 1), x1=datetime(2024, 4, 30), fillcolor="cyan", opacity=0.06, line_width=0, annotation_text="2~4月 90天盲區預報 (4月官方評測)", annotation_position="top left")

    fig.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🗺️ 2D 全域空間地理擴散熱力圖")
    if field_data is not None:
        z_spatial_field = field_data['z_spatial_pred'] # (90, 4, 70, 100)
        blind_dates_f = field_data['dates']

        c_slider, c_chan = st.columns([3, 1])
        with c_slider:
            day_idx = st.slider("選擇 90 天盲區預測日期", 0, len(blind_dates_f) - 1, 0, format=f"Day %d")
            sel_date_str = blind_dates_f[day_idx]
            st.info(f"📅 預測日期: **{sel_date_str}**")
        with c_chan:
            ch_idx = st.selectbox("選擇空間通道", [0, 1, 2, 3], format_func=lambda x: {0: "Ch 0: 區域留存密度", 1: "Ch 1: 內部流出場", 2: "Ch 2: 內部流入場", 3: "Ch 3: 外域交換量"}[x])

        grid_2d = z_spatial_field[day_idx, ch_idx]
        fig_heat = go.Figure(data=go.Heatmap(z=grid_2d, colorscale='Viridis', zmin=-2.0, zmax=2.0))
        fig_heat.update_layout(title=f"2D 空間擴散場 - {sel_date_str} (通道 {ch_idx})", height=500, xaxis_title="X 空間座標 (0~100)", yaxis_title="Y 空間座標 (0~70)")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("尚無 2D 空間場數據，請先執行 pipeline 生成。")
