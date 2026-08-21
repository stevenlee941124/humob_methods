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
    page_title="HuMob 2026: 移動性預測對比儀表板",
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

@st.cache_data
def load_core_data():
    p_ts = PACKAGE_ROOT / "data" / "processed" / "od_time_series.pkl"
    p_dt = PACKAGE_ROOT / "data" / "processed" / "dates.pkl"
    with open(p_ts, 'rb') as f:
        od_ts = pickle.load(f)
    with open(p_dt, 'rb') as f:
        dates_str = pickle.load(f)

    # Classification
    p_cls = PACKAGE_ROOT / "data" / "processed" / "grid_final_classification.csv"
    df_cls = pd.read_csv(p_cls)
    class_map = dict(zip(df_cls['grid_id'], df_cls['final_class']))

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

    # Destination mapping
    dest_map = {}
    for k in od_ts.keys():
        pts = k.replace('-1_-1-', '-1_-1_').split('-') if k.startswith('-1_-1-') else k.split('-')
        orig = '-1_-1' if k.startswith('-1_-1-') else pts[0]
        dest = pts[1].replace('_', '-') if k.startswith('-1_-1-') else pts[1]
        dest_map.setdefault(dest, []).append(orig)

    idx_apr1 = dates_str.index('20240401') if '20240401' in dates_str else 92
    idx_apr30 = dates_str.index('20240430') if '20240430' in dates_str else 121

    return od_ts, dates_str, class_map, df_top_ods, garch_data, base_data, dest_map, idx_apr1, idx_apr30

def main():
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1B263B, #2C3E50); padding: 18px 24px; border-radius: 8px; margin-bottom: 20px; color: white;">
        <h2 style="margin: 0; color: #F1C40F;">👑 HuMob 2026 移動性預測對比儀表板</h2>
        <p style="margin: 6px 0 0 0; color: #BDC3C7; font-size: 14px;">
            專注於 <b>【全年真實流量】</b>、<b>【四月真實答案】</b>、<b>【宏觀平滑 Baseline】</b> 與 <b>【👑 全年度去污染 GARCH 模型】</b> 波形與誤差切片
        </p>
    </div>
    """, unsafe_allow_html=True)

    od_ts, dates_str, class_map, df_top_ods, garch_data, base_data, dest_map, idx_apr1, idx_apr30 = load_core_data()

    # ── Top Scoreboard (Latest Official Spec: mean_actual_diag=26.57, mean_actual_offdiag=0.0176) ──
    c_card1, c_card2, c_card3 = st.columns([1.2, 1.2, 1.0])
    with c_card1:
        st.markdown("""
        <div style="background-color: #FEF9E7; border-left: 5px solid #F1C40F; padding: 12px 18px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h5 style="margin: 0 0 8px 0; color: #7D6608;">👑 GARCH 模型 🏆</h5>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 13.5px;">
                <div><b>• 最新 Combined NRMSE:</b> <span style="color: #27AE60; font-weight: bold; font-size: 15px;">0.27621</span></div>
                <div><b>• Diag RMSE:</b> <b>3.28 人</b></div>
                <div><b>• NRMSE Diag:</b> <b>0.12342</b></div>
                <div><b>• NRMSE Off:</b> <b>0.42900</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_card2:
        st.markdown("""
        <div style="background-color: #F8F9F9; border-left: 5px solid #7F8C8D; padding: 12px 18px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h5 style="margin: 0 0 8px 0; color: #2C3E50;">📏 宏觀平滑 Baseline (灰虛線基準)</h5>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 13.5px;">
                <div><b>• 最新 Combined NRMSE:</b> <span style="color: #7F8C8D; font-weight: bold; font-size: 15px;">0.28940</span></div>
                <div><b>• Diag RMSE:</b> <b>3.73 人</b></div>
                <div><b>• NRMSE Diag:</b> <b>0.14020</b></div>
                <div><b>• NRMSE Off:</b> <b>0.43860</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_card3:
        st.markdown("""
        <div style="background-color: #EAF2F8; border-left: 5px solid #2980B9; padding: 12px 18px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h5 style="margin: 0 0 8px 0; color: #1A5276;">🔥 官方規則改善收益</h5>
            <div style="font-size: 13.5px;">
                <div><b>• NRMSE 絕對降幅:</b> <span style="color: #27AE60; font-weight: bold;">-0.01319 (-1.32%)</span></div>
                <div><b>• 實體誤差減少:</b> <span style="color: #27AE60; font-weight: bold;">-0.45 人 / 網格</span></div>
                <div><b>• 官方分母:</b> <b>26.57 / 0.0176</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    # ── Flow Scale Stratification & Top Error OD Table ──
    with st.expander("📊 查看五級流量切片 (Flow Scale) 與 Top 20 關鍵首惡 OD 清單", expanded=False):
        t_col1, t_col2 = st.columns([1, 1.2])
        with t_col1:
            st.markdown("##### 📈 真實流量規模分桶大表")
            flow_df = pd.DataFrame([
                {"規模等級": "1. Tiny (<10人)", "樣本數": "33,811 (86.4%)", "MSE 佔比": "10.1%", "RMSE": "1.16 人", "MAE": "0.62 人"},
                {"規模等級": "2. Small (10~50人)", "樣本數": "3,445 (8.8%)", "MSE 佔比": "17.1%", "RMSE": "4.74 人", "MAE": "3.66 人"},
                {"規模等級": "3. Medium (50~200人)", "樣本數": "1,481 (3.8%)", "MSE 佔比": "34.7% 🔥", "RMSE": "10.32 人", "MAE": "7.87 人"},
                {"規模等級": "4. Large (200~500人)", "樣本數": "293 (0.7%)", "MSE 佔比": "21.6%", "RMSE": "18.28 人", "MAE": "14.30 人"},
                {"規模等級": "5. Hub (≥500人)", "樣本數": "84 (0.2%)", "MSE 佔比": "16.6%", "RMSE": "29.92 人", "MAE": "23.31 人"}
            ])
            st.dataframe(flow_df, use_container_width=True, hide_index=True)

        with t_col2:
            st.markdown("##### 🚨 支配全域 49.8% MSE 之 Top 10 首惡 OD 路線")
            if df_top_ods is not None:
                disp_cols = ['rank', 'od_key', 'class', 'mean_true', 'max_true', 'rmse', 'cum_mse_pct']
                df_top_show = df_top_ods[disp_cols].head(10).copy()
                df_top_show.columns = ['排名', 'OD 路線', '類別', '均值', '最高峰', 'RMSE', '累積 MSE 🔥']
                st.dataframe(df_top_show, use_container_width=True, hide_index=True)

    # ── Sidebar Controls ──
    st.sidebar.markdown("### 🎯 1. 路線選擇導航")
    nav_mode = st.sidebar.radio(
        "導航模式",
        ["🔥 快速直達 Top 20 高誤差關鍵樞紐", "📁 按 9 大動力學類別探索"],
        index=0
    )

    if nav_mode == "🔥 快速直達 Top 20 高誤差關鍵樞紐" and df_top_ods is not None:
        df_top20 = df_top_ods.head(20)
        options = [
            f"#{r['rank']:02d} | {r['od_key']} (RMSE: {r['rmse']:.1f}人 | 佔總MSE {r['mse_share_pct']:.1f}% | {r['class']})"
            for _, r in df_top20.iterrows()
        ]
        sel_str = st.sidebar.selectbox("選擇高誤差關鍵 OD", options, index=0)
        sel_rank = int(sel_str.split('|')[0].replace('#', '').strip())
        od_key = df_top20.iloc[sel_rank - 1]['od_key']
        parts = od_key.replace('-1_-1-', '-1_-1_').split('-') if od_key.startswith('-1_-1-') else od_key.split('-')
        sel_orig = '-1_-1' if od_key.startswith('-1_-1-') else parts[0]
        sel_dest = parts[1].replace('_', '-') if od_key.startswith('-1_-1-') else parts[1]
    else:
        dests_all = set(dest_map.keys())
        classes = sorted(set(class_map.get(d, "Unknown") for d in dests_all))
        sel_cls = st.sidebar.selectbox("Grid Class", classes, index=classes.index("True Stable") if "True Stable" in classes else 0)
        dests_filtered = sorted(d for d in dests_all if class_map.get(d, "Unknown") == sel_cls)
        sel_dest = st.sidebar.selectbox("Destination Grid", dests_filtered)
        origs = sorted(dest_map.get(sel_dest, []))
        sel_orig = st.sidebar.selectbox("Origin Grid", origs, index=origs.index(sel_dest) if sel_dest in origs else 0)
        od_key = f"-1_-1-{sel_dest}" if sel_orig == '-1_-1' else f"{sel_orig}-{sel_dest}"

    st.sidebar.divider()
    st.sidebar.markdown("### 🎛️ 2. 波形圖層開關")
    show_gt = st.sidebar.checkbox("🔴 顯示真實流量 Ground Truth (全年紅實線)", value=True)
    show_apr_gt = st.sidebar.checkbox("🔥 顯示四月真實答案高亮 (亮紅標記)", value=True)
    show_base = st.sidebar.checkbox("📏 顯示宏觀平滑中軸基準線 (灰色虛線)", value=True)
    show_garch = st.sidebar.checkbox("👑 全年度去污染 GARCH (亮金粗線)", value=True)

    if od_key not in od_ts:
        st.error(f"找不到 OD Pair: {od_key}")
        return

    ts_orig = od_ts[od_key]
    c_name = class_map.get(sel_dest, 'Unknown')

    # Continuous 366 days calendar
    start_dt = datetime(2023, 11, 1)
    end_dt = datetime(2024, 10, 31)
    total_days = (end_dt - start_dt).days + 1
    cal_dts_366 = [start_dt + timedelta(days=i) for i in range(total_days)]
    cal_dates_366 = [d.strftime("%Y%m%d") for d in cal_dts_366]

    gt_dict = dict(zip(dates_str, ts_orig))
    ts_masked_366 = []
    for d_str in cal_dates_366:
        if d_str.startswith('202402') or d_str.startswith('202403') or d_str in EXCLUDED_DATES or d_str not in gt_dict:
            ts_masked_366.append(np.nan)
        else:
            v = gt_dict[d_str]
            ts_masked_366.append(np.nan if np.isnan(v) else float(v))

    gap90_dts = [datetime(2024, 2, 1) + timedelta(days=i) for i in range(90)]
    apr_dts = [datetime(2024, 4, 1) + timedelta(days=i) for i in range(30)]

    true_apr = np.array(ts_orig[idx_apr1:idx_apr30+1], dtype=float)
    for i, d in enumerate(apr_dts):
        if d.strftime("%Y%m%d") in EXCLUDED_DATES:
            true_apr[i] = np.nan

    def extract_pred(data_dict, src, dst, dates):
        flows = []
        for d in dates:
            d_str = d.strftime("%Y%m%d")
            if d_str in EXCLUDED_DATES:
                flows.append(np.nan)
            else:
                flows.append(data_dict.get(d_str, {}).get(src, {}).get(dst, np.nan))
        return flows

    garch_flows = extract_pred(garch_data, sel_orig, sel_dest, gap90_dts)
    base_flows = extract_pred(base_data, sel_orig, sel_dest, gap90_dts)

    # ── Plotting ──
    fig = go.Figure()

    if show_gt:
        fig.add_trace(go.Scatter(
            x=cal_dts_366, y=ts_masked_366, mode='lines',
            name='🔴 全年真實流量 (Ground Truth)',
            line=dict(color='#E53935', width=2.5),
            connectgaps=False
        ))

    if show_apr_gt:
        fig.add_trace(go.Scatter(
            x=apr_dts, y=true_apr, mode='lines',
            name='🔥 四月真實答案 (Ground Truth 標記)',
            line=dict(color='#D50000', width=3.8),
            connectgaps=False
        ))

    if show_base:
        fig.add_trace(go.Scatter(
            x=gap90_dts, y=base_flows, mode='lines',
            name='📏 宏觀平滑 Baseline (平滑灰虛線)',
            line=dict(color='#7F8C8D', width=2.2, dash='dash')
        ))

    if show_garch:
        fig.add_trace(go.Scatter(
            x=gap90_dts, y=garch_flows, mode='lines',
            name='👑 全年去污染 GARCH (亮金線)',
            line=dict(color='#F39C12', width=3.2)
        ))

    # Layout styling
    fig.add_vrect(
        x0=datetime(2024, 1, 1), x1=datetime(2024, 1, 31),
        fillcolor="rgba(231, 76, 60, 0.08)", line_width=0,
        annotation_text="1月 地震衝擊期", annotation_position="top left"
    )
    fig.add_vrect(
        x0=datetime(2024, 2, 1), x1=datetime(2024, 4, 30),
        fillcolor="rgba(52, 152, 219, 0.08)", line_width=0,
        annotation_text="2~4月 90天盲區預報 (4月官方評測)", annotation_position="top left"
    )

    fig.update_layout(
        title=dict(
            text=f"<b>OD Pair [{od_key}] 波形對比</b> | 目的地類別: <span style='color:#E67E22;'>{c_name}</span>",
            font=dict(size=17, color="#2C3E50")
        ),
        xaxis=dict(
            title="時間 (Date)",
            showgrid=True, gridcolor="rgba(200,200,200,0.3)",
            range=[datetime(2023, 11, 1), datetime(2024, 5, 5)]
        ),
        yaxis=dict(
            title="人流量 (Persons / Day)",
            showgrid=True, gridcolor="rgba(200,200,200,0.3)"
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.85)"
        ),
        hovermode="x unified",
        height=580,
        margin=dict(l=40, r=30, t=75, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Single OD Error Forensics Card ──
    if df_top_ods is not None and od_key in df_top_ods['od_key'].values:
        row_od = df_top_ods[df_top_ods['od_key'] == od_key].iloc[0]
        st.markdown(f"""
        <div style="background-color: #F4F6F7; border: 1px solid #D5DBDB; border-radius: 6px; padding: 12px 18px; margin-top: 10px;">
            <h5 style="margin: 0 0 6px 0; color: #2C3E50;">🔍 當前 OD [{od_key}] 誤差切片指標 (Rank #{row_od['rank']})</h5>
            <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; font-size: 13.5px;">
                <div><b>• 全域排名:</b> <span style="color:#C0392B; font-weight:bold;">Top #{row_od['rank']}</span></div>
                <div><b>• 真實日均流量:</b> {row_od['mean_true']:.1f} 人</div>
                <div><b>• 真實最高波峰:</b> {row_od['max_true']:.1f} 人</div>
                <div><b>• GARCH RMSE:</b> <span style="color:#D35400; font-weight:bold;">{row_od['rmse']:.2f} 人</span></div>
                <div><b>• GARCH MAE:</b> {row_od['mae']:.2f} 人</div>
                <div><b>• 佔全域總 MSE:</b> <span style="color:#E74C3C; font-weight:bold;">{row_od['mse_share_pct']:.2f}%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
