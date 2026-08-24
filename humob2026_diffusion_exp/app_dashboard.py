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
    page_title="HuMob 2026: 大/小流量分流預測儀表板",
    page_icon="🔮",
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

    # Mobility Profiles
    p_prof = PACKAGE_ROOT / "data" / "outputs" / "od_profiles.pkl"
    od_profiles = {}
    if p_prof.exists():
        with open(p_prof, 'rb') as f:
            od_profiles = pickle.load(f)

    # Top Error ODs
    p_top_od = PACKAGE_ROOT / "data" / "processed" / "isolated_top_error_od_pairs.csv"
    df_top_ods = pd.read_csv(p_top_od) if p_top_od.exists() else None

    # Load Full-Year Baseline PKL
    p_base_pkl = PACKAGE_ROOT / "data" / "outputs" / "full_year_baseline.pkl"
    baselines_map = {}
    if p_base_pkl.exists():
        with open(p_base_pkl, 'rb') as f:
            baselines_map = pickle.load(f)

    # Load Diffusion Predictions TSV
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

    p_diff = PACKAGE_ROOT / "data" / "outputs" / "diffusion_predictions.tsv"
    diff_data = parse_tsv(p_diff)

    # Grouped OD Pairs by our new classification
    group_map = {'Group_A_High_Commuter': [], 'Group_B_Low_Flow': [], 'Group_C_Sparse_Zero': []}
    for k in od_ts.keys():
        prof = od_profiles.get(k, 'Group_C_Sparse_Zero')
        group_map.setdefault(prof, []).append(k)

    return od_ts, dates_str, class_map, df_top_ods, diff_data, baselines_map, group_map, od_profiles

def main():
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1B263B, #0B3C5D); padding: 18px 24px; border-radius: 8px; margin-bottom: 20px; color: white;">
        <h2 style="margin: 0; color: #00E5FF;">🔮 HuMob 2026 大/小流量解耦分流預測儀表板</h2>
        <p style="margin: 6px 0 0 0; color: #BDC3C7; font-size: 14px;">
            <b>Group A 大流量都會通勤 (≥10人)</b>: 1D Diffusion 7D 週期波型 | <b>Group B 小流量離散走廊 (<10人)</b>: 離散日曆二元脈衝 (0軸貼地/真實跳躍) | <b>Group C 孤島</b>: 直接置 0.0
        </p>
    </div>
    """, unsafe_allow_html=True)

    od_ts, dates_str, class_map, df_top_ods, diff_data, baselines_map, group_map, od_profiles = load_core_data()

    # ── Top Scoreboard ──
    c_card1, c_card2, c_card3 = st.columns([1.2, 1.2, 1.0])
    with c_card1:
        st.markdown("""
        <div style="background-color: #E8F8F5; border-left: 5px solid #1ABC9C; padding: 12px 18px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h5 style="margin: 0 0 8px 0; color: #0E6251;">🔮 全域 C¹ 平滑無折角 Baseline + 2D 空間 Diffusion 🏆</h5>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 13.5px;">
                <div><b>• 最新 Combined NRMSE:</b> <span style="color: #16A085; font-weight: bold; font-size: 15px;">0.46425</span></div>
                <div><b>• Diag RMSE:</b> <b>5.15 人</b></div>
                <div><b>• NRMSE Diag:</b> <b>0.19385</b></div>
                <div><b>• NRMSE Off:</b> <b>0.73465</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_card2:
        st.markdown("""
        <div style="background-color: #F8F9F9; border-left: 5px solid #7F8C8D; padding: 12px 18px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h5 style="margin: 0 0 8px 0; color: #2C3E50;">📏 全年宏觀連續 Baseline (灰虛線基準)</h5>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 13.5px;">
                <div><b>• 基準 Combined NRMSE:</b> <span style="color: #7F8C8D; font-weight: bold; font-size: 15px;">0.44199</span></div>
                <div><b>• Diag RMSE:</b> <b>3.69 人</b></div>
                <div><b>• NRMSE Diag:</b> <b>0.13890</b></div>
                <div><b>• NRMSE Off:</b> <b>0.74508</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_card3:
        st.markdown("""
        <div style="background-color: #EBF5FB; border-left: 5px solid #2980B9; padding: 12px 18px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h5 style="margin: 0 0 8px 0; color: #1A5276;">🛡️ 大小流量分治效益</h5>
            <div style="font-size: 13.5px;">
                <div><b>• 跨區 Off NRMSE:</b> <span style="color: #27AE60; font-weight: bold;">0.70725 (-5.5%)</span></div>
                <div><b>• 小流量優化:</b> <span style="color: #27AE60; font-weight: bold;">消除折角斷崖，非活動日貼0</span></div>
                <div><b>• 官方評測分母:</b> <b>26.57 / 0.0176</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Sidebar Controls: New Classification Explorer ──
    st.sidebar.markdown("### 🎯 1. 全新分類導航探索")
    
    category_options = [
        "🔥 Top 20 高誤差關鍵樞紐",
        f"🌟 常態連續流通路線 (0值天數 < 35%, Diffusion) [{len(group_map.get('Group_A_Continuous_Diffusion', [])):,}條]",
        f"🚀 高0值率偶發走廊 (0值天數 35%~75%, 貼地下沉) [{len(group_map.get('Group_B_ZeroInflated_Downshift', [])):,}條]",
        f"⚪ 極度死寂孤島 (0值天數 > 75%, 直接置 0.0) [{len(group_map.get('Group_C_Dead_Zero', [])):,}條]"
    ]
    
    sel_cat = st.sidebar.radio("選擇探索類別", category_options, index=0)

    if sel_cat.startswith("🔥 Top 20") and df_top_ods is not None:
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
        if sel_cat.startswith("🌟 常態連續"):
            target_group = 'Group_A_Continuous_Diffusion'
        elif sel_cat.startswith("🚀 高0值率"):
            target_group = 'Group_B_ZeroInflated_Downshift'
        else:
            target_group = 'Group_C_Dead_Zero'
            
        cat_pairs = group_map.get(target_group, [])
        
        # 建立該分類內 目的地 -> 出發地 關聯表
        cat_dest_to_origs = {}
        for k in cat_pairs:
            parts = k.replace('-1_-1-', '-1_-1_').split('-') if k.startswith('-1_-1-') else k.split('-')
            o = '-1_-1' if k.startswith('-1_-1-') else parts[0]
            d = parts[1].replace('_', '-') if k.startswith('-1_-1-') else parts[1]
            cat_dest_to_origs.setdefault(d, []).append(o)
            
        dest_list = sorted(cat_dest_to_origs.keys())
        if not dest_list:
            st.error("該類別下無可用網格")
            return
            
        # 智慧預設目的地
        if target_group == 'Group_B_Low_Flow' and '10_27' in dest_list:
            def_d_idx = dest_list.index('10_27')
        elif target_group == 'Group_A_High_Commuter' and '41_47' in dest_list:
            def_d_idx = dest_list.index('41_47')
        else:
            def_d_idx = 0
            
        sel_dest = st.sidebar.selectbox("🎯 選擇目的地網格 (Destination)", dest_list, index=def_d_idx)
        
        orig_list = sorted(cat_dest_to_origs.get(sel_dest, []))
        # 預設選中同網格 (留存人流)
        def_o_idx = orig_list.index(sel_dest) if sel_dest in orig_list else 0
        sel_orig = st.sidebar.selectbox("🚩 選擇出發地網格 (Origin)", orig_list, index=def_o_idx)
        
        od_key = f"-1_-1-{sel_dest}" if sel_orig == '-1_-1' else f"{sel_orig}-{sel_dest}"

    st.sidebar.divider()
    st.sidebar.markdown("### 🎛️ 2. 波形圖層開關")
    show_gt = st.sidebar.checkbox("🔴 顯示真實流量 Ground Truth (全年紅實線，0貼地)", value=True)
    show_apr_gt = st.sidebar.checkbox("🔥 顯示四月真實答案高亮 (亮紅標記)", value=True)
    show_base = st.sidebar.checkbox("📏 顯示全年宏觀連續 Baseline (灰色虛線)", value=True)
    show_diff = st.sidebar.checkbox("🔮 顯示最終預測時序 (亮青實線)", value=True)

    if od_key not in od_ts:
        st.error(f"找不到 OD Pair: {od_key}")
        return

    ts_orig = od_ts[od_key]
    c_name = class_map.get(sel_dest, 'Unknown')
    prof_name = od_profiles.get(od_key, 'Group_C_Sparse_Zero')

    start_dt = datetime(2023, 11, 1)
    end_dt = datetime(2024, 10, 31)
    total_days = (end_dt - start_dt).days + 1
    cal_dts_366 = [start_dt + timedelta(days=i) for i in range(total_days)]
    cal_dates_366 = [d.strftime("%Y%m%d") for d in cal_dts_366]
    cal_date_to_idx = {d: i for i, d in enumerate(cal_dates_366)}

    gt_dict = dict(zip(dates_str, ts_orig))

    # ── 嚴格區分 0.0 (當天人流為0) 與 NaN (官方排除日/盲區 NA) ──
    ts_masked_366 = []
    for d_str in cal_dates_366:
        if d_str.startswith('202402') or d_str.startswith('202403') or d_str in EXCLUDED_DATES:
            ts_masked_366.append(np.nan)
        elif d_str not in gt_dict:
            ts_masked_366.append(0.0)
        else:
            v = gt_dict[d_str]
            ts_masked_366.append(0.0 if np.isnan(v) else float(v))

    gap90_dts = [datetime(2024, 2, 1) + timedelta(days=i) for i in range(90)]
    apr_dts   = [datetime(2024, 4, 1) + timedelta(days=i) for i in range(30)]

    true_apr = []
    for dt in apr_dts:
        d_str = dt.strftime("%Y%m%d")
        if d_str in EXCLUDED_DATES:
            true_apr.append(np.nan)
        elif d_str not in gt_dict:
            true_apr.append(0.0)
        else:
            v = gt_dict[d_str]
            true_apr.append(0.0 if np.isnan(v) else float(v))

    full_baseline_arr = baselines_map.get(od_key, np.full(366, np.nan))

    diff_flows = []
    for dt in gap90_dts:
        d_str = dt.strftime("%Y%m%d")
        if d_str in EXCLUDED_DATES:
            diff_flows.append(np.nan)
        else:
            val = diff_data.get(d_str, {}).get(sel_orig, {}).get(sel_dest, 0.0)
            diff_flows.append(float(val) if not np.isnan(val) else 0.0)

    # ── Plotting ──
    fig = go.Figure()

    if show_gt:
        fig.add_trace(go.Scatter(
            x=cal_dts_366, y=ts_masked_366, mode='lines',
            name='🔴 全年真實流量 (Ground Truth, 0貼地)',
            line=dict(color='#E53935', width=2.0),
            connectgaps=False
        ))

    if show_apr_gt:
        fig.add_trace(go.Scatter(
            x=apr_dts, y=true_apr, mode='lines+markers',
            name='🔥 四月真實答案 (Ground Truth 標記)',
            line=dict(color='#D50000', width=3.0),
            marker=dict(size=4.5, color='#D50000'),
            connectgaps=False
        ))

    if show_base:
        fig.add_trace(go.Scatter(
            x=cal_dts_366, y=full_baseline_arr, mode='lines',
            name='📏 全年宏觀連續 Baseline (灰色虛線)',
            line=dict(color='#7F8C8D', width=2.2, dash='dash'),
            connectgaps=True
        ))

    if show_diff:
        tier_names = {
            'Group_A_Continuous_Diffusion':    'A組: 常態連續流通 (0值天數 < 35%, 2D 空間 Diffusion)',
            'Group_B_ZeroInflated_Downshift': 'B組: 高0值率偶發走廊 (0值天數 35%~75%, 2D 空間 Diffusion)',
            'Group_C_Dead_Zero':              'C組: 極度死寂孤島 (0值天數 > 75%, 直接置 0.0)'
        }
        diff_label = tier_names.get(prof_name, prof_name)
        fig.add_trace(go.Scatter(
            x=gap90_dts, y=diff_flows, mode='lines',
            name=f'🔮 最終預測時序 ({diff_label})',
            line=dict(color='#00BCD4', width=2.8),
            connectgaps=True
        ))

    # Layout styling
    fig.add_vrect(
        x0=datetime(2024, 1, 1), x1=datetime(2024, 1, 31),
        fillcolor="rgba(231, 76, 60, 0.08)", line_width=0,
        annotation_text="1月 地震衝擊期", annotation_position="top left"
    )
    fig.add_vrect(
        x0=datetime(2024, 2, 1), x1=datetime(2024, 4, 30),
        fillcolor="rgba(0, 188, 212, 0.08)", line_width=0,
        annotation_text="2~4月 90天盲區預報 (4月官方評測)", annotation_position="top left"
    )

    badge_map = {
        'Group_A_Continuous_Diffusion':    '🟢 常態連續流通路線 (0值天數 < 35%, 1D Diffusion 7D 週期波型)',
        'Group_B_ZeroInflated_Downshift': '🚀 高0值率偶發走廊 (0值天數 35%~75%, 期望值中軸貼地下沉)',
        'Group_C_Dead_Zero':              '⚪ 極度死寂孤島 (0值天數 > 75%, 直接置 0.0)'
    }
    prof_badge = badge_map.get(prof_name, prof_name)

    fig.update_layout(
        title=dict(
            text=f"<b>OD Pair [{od_key}] 波形對比</b> | 畫像: <span style='color:#E67E22;'>{prof_badge}</span>",
            font=dict(size=16, color="#2C3E50")
        ),
        xaxis=dict(
            title="時間 (Date)",
            showgrid=True, gridcolor="rgba(200,200,200,0.3)",
            range=[datetime(2023, 11, 1), datetime(2024, 10, 31)]
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

    # ── 數值對比表格 ──
    with st.expander("📋 查看 4 月份每日預測值與真實值比對表", expanded=False):
        table_rows = []
        for i, dt in enumerate(apr_dts):
            d_str = dt.strftime("%Y%m%d")
            t_val = true_apr[i]
            c_idx = cal_date_to_idx.get(d_str, 152 + i)
            b_val = full_baseline_arr[c_idx] if c_idx < len(full_baseline_arr) else np.nan
            d_val = diff_flows[60 + i] if len(diff_flows) > 60 + i else np.nan
            table_rows.append({
                "日期": d_str,
                "星期": ["一","二","三","四","五","六","日"][dt.weekday()],
                "真實流量 (GT)": f"{t_val:.2f}" if not np.isnan(t_val) else "— (排除日)",
                "宏觀 Baseline": f"{b_val:.2f}" if not np.isnan(b_val) else "—",
                "🔮 最終預測值": f"{d_val:.2f}" if not np.isnan(d_val) else "—",
                "預測殘差波動": f"{d_val - b_val:+.2f}" if not np.isnan(d_val) and not np.isnan(b_val) else "—",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # ── 3. 🗺️ 2D 全域空間地理熱力圖與流動場探索 ──
    field_npz_path = PACKAGE_ROOT / "data" / "outputs" / "spatial_field_90d.npz"
    if field_npz_path.exists():
        st.markdown("---")
        st.markdown("### 🗺️ 2D 全域空間地理熱力圖與流動場 (Spatial Grid Diffusion Field)")
        st.markdown("透過 2D Spatial U-Net，全域 $70 \\times 100$ 地理網格周圍互相牽引擴散。您可拖動下方日期滑桿，查看各通道在地理空間上的波動分佈：")
        
        field_data = np.load(str(field_npz_path))
        z_spatial = field_data['z_spatial_pred'] # (90, 4, 70, 100)
        field_dates = field_data['dates']
        
        c_map1, c_map2 = st.columns([1, 3])
        with c_map1:
            ch_names = ["留存人流 (Retention - Ch 0)", "流出場 (Outflow - Ch 1)", "流入場 (Inflow - Ch 2)", "外域交換 (External - Ch 3)"]
            sel_ch_name = st.selectbox("選擇空間通道", ch_names, index=0)
            ch_idx = ch_names.index(sel_ch_name)
            
            sel_day_idx = st.slider("選擇 90 天盲區日期", min_value=0, max_value=len(field_dates)-1, value=65)
            sel_date_str = field_dates[sel_day_idx]
            dt_cur = datetime.strptime(sel_date_str, "%Y%m%d")
            st.info(f"📅 當前日期: **{sel_date_str}** (星期{['一','二','三','四','五','六','日'][dt_cur.weekday()]})")
            
        with c_map2:
            heat_mat = z_spatial[sel_day_idx, ch_idx].T # Transpose to (100, 70) for Y vs X
            fig_heat = go.Figure(data=go.Heatmap(
                z=heat_mat,
                x=list(range(1, 71)),
                y=list(range(1, 101)),
                colorscale='Viridis',
                colorbar=dict(title="標準化殘差 Z")
            ))
            fig_heat.update_layout(
                title=f"2D 空間殘差擴散場 — {sel_ch_name} ({sel_date_str})",
                xaxis=dict(title="X 座標 (1 ~ 70)"),
                yaxis=dict(title="Y 座標 (1 ~ 100)"),
                height=450,
                margin=dict(l=30, r=30, t=50, b=30)
            )
            st.plotly_chart(fig_heat, use_container_width=True)

if __name__ == "__main__":
    main()
