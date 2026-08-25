"""
===============================================================================
HuMob 2026: 2D Spatial-Temporal Diffusion Model Analysis Dashboard
===============================================================================
"""
import math
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="HuMob 2026: 2D Spatial Diffusion Dashboard",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

PACKAGE_ROOT = Path(__file__).resolve().parent

OD_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DT_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
PRED_TSV  = PACKAGE_ROOT / 'data' / 'outputs' / 'diffusion_predictions.tsv'
FIELD_NPZ = PACKAGE_ROOT / 'data' / 'outputs' / 'spatial_field_90d.npz'
BASE_PKL  = PACKAGE_ROOT / 'data' / 'outputs' / 'full_year_baseline.pkl'

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

def load_base_structures():
    with open(OD_PKL, 'rb') as f: od_ts = pickle.load(f)
    with open(DT_PKL, 'rb') as f: dates_str = pickle.load(f)
    base_data = pickle.load(open(BASE_PKL, 'rb')) if BASE_PKL.exists() else {}
    field_data = dict(np.load(FIELD_NPZ)) if FIELD_NPZ.exists() else None
    return od_ts, dates_str, base_data, field_data

od_ts, dates_str, base_data, field_data = load_base_structures()

def parse_pred_tsv(filepath):
    data = {}
    p = Path(filepath)
    if not p.exists(): return data
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            pts = line.strip().split('\t')
            if len(pts) >= 2:
                try:
                    raw = pts[1].replace(': NA', ': None').replace(':NA', ':None')
                    od = eval(raw, {'__builtins__': {}}, {'None': None})
                    if od is not None: data[pts[0]] = od
                except: pass
    return data

pred_data = parse_pred_tsv(PRED_TSV)

# ── 頁首 Header ──────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(90deg, #0B1E36, #1B3B6F); padding: 18px 24px; border-radius: 8px; margin-bottom: 20px; color: white;">
    <h2 style="margin: 0; color: #00E5FF;">🌌 HuMob 2026: 2D 空間條件擴散模型分析儀表板</h2>
    <p style="margin: 6px 0 0 0; color: #CFD8DC; font-size: 14px;">
        4 通道 2D 空間流動張量 (70×100) + Spatial U-Net (DDIM 50-Step 採樣) 全域地理時空流動重建
    </p>
</div>
""", unsafe_allow_html=True)

# ── 頂部模型指標看板 ──────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🏆 Combined NRMSE", "0.30979", "純 2D 空間擴散模型 (4月錨定)")
with c2:
    st.metric("停留流動 (Diag) NRMSE", "0.17616", "RMSE 4.68 人")
with c3:
    st.metric("跨區流動 (Off-Diag) NRMSE", "0.44342", "RMSE 0.0078 人")
with c4:
    st.metric("🎯 官方評測範圍", "1,476 網格", "x:30~70, y:35~70 (1,836 路線)")

tab1, tab2 = st.tabs(["📈 OD 路線時序波形與擴散預測檢視", "🗺️ 2D 全域空間地理擴散熱力圖"])

with tab1:
    st.sidebar.header("🎯 路線與分類導航")

    # 0. 空間範圍選單 (官方評測範圍 x:30~70, y:35~70)
    scope_options = [
        "🎯 官方評測範圍 (x: 30~70, y: 35~70)",
        "🌐 全域所有路線 (含邊緣與外部)"
    ]
    sel_scope = st.sidebar.selectbox("🗺️ 0. 選擇空間範圍 (Spatial Scope)", scope_options, index=0)

    def in_eval_bbox(g_str):
        if g_str == '-1_-1': return False
        pts = g_str.split('_')
        if len(pts) != 2: return False
        try:
            gx, gy = int(pts[0]), int(pts[1])
            return (30 <= gx <= 70) and (35 <= gy <= 70)
        except:
            return False

    # 1. 規律度分類選單
    reg_options = [
        "🌟 全部路線 (All)",
        "🔥 Top 20 高誤差關鍵樞紐 (20 條)"
    ]
    sel_reg = st.sidebar.selectbox("📂 1. 選擇路線群組 (Group)", reg_options, index=0)

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

    # 交叉篩選
    filtered_pairs = []
    top20_keys = [
        '41_47-41_47', '21_60-21_60', '11_25-11_25', '24_63-24_63', '10_23-10_21',
        '10_23-10_23', '11_28-11_28', '10_27-10_27', '35_48-35_48', '37_48-37_48',
        '40_47-40_47', '41_48-41_48', '42_47-42_47', '39_47-39_47', '43_47-43_47',
        '38_48-38_48', '40_48-40_48', '36_48-36_48', '42_48-42_48', '44_47-44_47'
    ]

    for k, raw in od_ts.items():
        # 空間範圍過濾
        if "官方評測" in sel_scope:
            if k.startswith('-1_-1-'): o, d = '-1_-1', k[6:]
            elif k.endswith('--1_-1'): o, d = k[:-6], '-1_-1'
            else: pts = k.split('-'); o, d = pts[0], pts[1]
            if not (in_eval_bbox(o) and in_eval_bbox(d)):
                continue

        # 規律度/Top20 過濾
        if "Top 20" in sel_reg:
            if k not in top20_keys: continue

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

    # 智慧預設出發地
    if '41_47' in all_origs: def_o_idx = all_origs.index('41_47')
    elif '21_60' in all_origs: def_o_idx = all_origs.index('21_60')
    elif '25_60' in all_origs: def_o_idx = all_origs.index('25_60')
    elif '11_28' in all_origs: def_o_idx = all_origs.index('11_28')
    else: def_o_idx = 0

    # 3. 選擇出發地 (Origin)
    sel_orig = st.sidebar.selectbox("🚩 3. 選擇起點 / 出發地 (Origin)", all_origs, index=def_o_idx)

    # 4. 選擇目的地 (Destination, 自動連動)
    valid_dests = sorted(orig_to_dests.get(sel_orig, []))
    def_d_idx = valid_dests.index(sel_orig) if sel_orig in valid_dests else 0
    sel_dest = st.sidebar.selectbox("🎯 4. 選擇終點 / 目的地 (Destination)", valid_dests, index=def_d_idx)

    sel_od = f"-1_-1-{sel_dest}" if sel_orig == '-1_-1' else (f"{sel_orig}--1_-1" if sel_dest == '-1_-1' else f"{sel_orig}-{sel_dest}")

    # 圖層開關
    st.sidebar.subheader("圖層顯示開關")
    show_gt = st.sidebar.checkbox("🔴 全年真實流量 (Ground Truth)", True)
    show_apr = st.sidebar.checkbox("🔥 四月真實答案 (Ground Truth 標記)", True)
    show_base = st.sidebar.checkbox("📏 宏觀物理 Baseline", True)
    show_pred = st.sidebar.checkbox("🔮 2D 空間 Diffusion 預測時序", True)

    # ── 繪製 Plotly 折線圖 ─────────────────────────────────────
    start_dt = datetime(2023, 11, 1)
    cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
    cal_dts   = [start_dt + timedelta(days=i) for i in range(366)]
    cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}

    raw_y = [np.nan] * 366
    for i, d in enumerate(dates_str):
        if d in cal_date_to_idx:
            val = od_ts.get(sel_od, [np.nan] * len(dates_str))[i]
            if d not in EXCLUDED_DATES:
                raw_y[cal_date_to_idx[d]] = val if not np.isnan(val) else np.nan

    # 4月官方評測答案
    apr_dts, apr_gt = [], []
    for d_str in [f"202404{i:02d}" for i in range(1, 31)]:
        if d_str in dates_str and d_str not in EXCLUDED_DATES:
            idx = dates_str.index(d_str)
            apr_dts.append(datetime.strptime(d_str, "%Y%m%d"))
            apr_gt.append(od_ts.get(sel_od, [np.nan] * len(dates_str))[idx])

    # 90天盲區預測時序
    gap90_dts = [datetime(2024, 2, 1) + timedelta(days=i) for i in range(90)]
    gap90_dates = [d.strftime("%Y%m%d") for d in gap90_dts]

    pred_flows = []
    for d in gap90_dates:
        f = pred_data.get(d, {}).get(sel_orig, {}).get(sel_dest, 0.0)
        pred_flows.append(f)

    # Baseline
    base_val = base_data.get(sel_od, None)
    if isinstance(base_val, np.ndarray):
        base_flows = base_val
    elif isinstance(base_val, dict):
        base_flows = base_val.get('baseline', [np.nan] * 366)
    else:
        base_flows = [np.nan] * 366

    fig = go.Figure()

    # 1. 全年真值
    if show_gt:
        fig.add_trace(go.Scatter(
            x=cal_dts, y=raw_y,
            mode='lines',
            line=dict(color='rgba(231, 76, 60, 0.85)', width=1.5),
            name='全年真實流量 (Ground Truth)',
            connectgaps=False
        ))

    # 2. 4月真實答案
    if show_apr and apr_dts:
        fig.add_trace(go.Scatter(
            x=apr_dts, y=apr_gt,
            mode='lines',
            line=dict(color='rgba(192, 57, 43, 1.0)', width=2.8),
            name='🔥 四月真實答案 (Ground Truth 標記)',
            connectgaps=False
        ))

    # 3. Baseline
    if show_base and len(base_flows) == 366:
        fig.add_trace(go.Scatter(
            x=cal_dts, y=base_flows,
            mode='lines',
            line=dict(color='#7F8C8D', width=2.0, dash='dash'),
            name='📏 宏觀物理 Baseline'
        ))

    # 4. 2D 空間 Diffusion 預測
    if show_pred:
        fig.add_trace(go.Scatter(
            x=gap90_dts, y=pred_flows,
            mode='lines',
            line=dict(color='#00E5FF', width=2.5),
            name='🔮 2D 空間 Diffusion 預測時序'
        ))

    # 震災與盲區背景區間
    fig.add_vrect(x0="2024-01-01", x1="2024-01-31", fillcolor="rgba(231, 76, 60, 0.08)", line_width=0,
                  annotation_text="1月 地震衝擊期", annotation_position="top left")
    fig.add_vrect(x0="2024-02-01", x1="2024-04-30", fillcolor="rgba(0, 229, 255, 0.05)", line_width=0,
                  annotation_text="2~4月 90天盲區預報 (4月官方評測)", annotation_position="top left")

    fig.update_layout(
        title=f"<b>OD 路線時序檢視: {sel_od}</b> (起點: {sel_orig} ➔ 終點: {sel_dest})",
        xaxis=dict(title="日期 (2023/11 ~ 2024/10)", showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(title="人口流動量 (人次)", showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
        height=520,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🗺️ 2D 全域空間地理擴散殘差場 (70 × 100 網格)")
    if field_data is not None and 'arr_0' in field_data:
        tensor_90d = field_data['arr_0']
        day_idx = st.slider("選擇 90 天盲區預報天數 (Day 1 ~ Day 90)", 1, 90, 60)
        target_date_str = (datetime(2024, 2, 1) + timedelta(days=day_idx-1)).strftime("%Y-%m-%d")
        st.write(f"📅 當前觀測日期: **{target_date_str}**")

        c_map1, c_map2 = st.columns(2)
        with c_map1:
            st.markdown("**【Channel 0】 內部停留流動空間場 (Stay / Diag)**")
            fig_hm0 = go.Figure(data=go.Heatmap(
                z=tensor_90d[day_idx-1, 0, :, :],
                colorscale='Viridis',
                zmin=-2.0, zmax=2.0
            ))
            fig_hm0.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_hm0, use_container_width=True)

        with c_map2:
            st.markdown("**【Channel 1】 跨區流出空間場 (Outflow)**")
            fig_hm1 = go.Figure(data=go.Heatmap(
                z=tensor_90d[day_idx-1, 1, :, :],
                colorscale='Plasma',
                zmin=-2.0, zmax=2.0
            ))
            fig_hm1.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_hm1, use_container_width=True)
    else:
        st.info("ℹ️ 未檢測到 2D 空間場快照數據，請先執行 step4 採樣輸出 spatial_field_90d.npz")
