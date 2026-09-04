"""
===============================================================================
HuMob 2026: Per-Origin Flow Matching Analysis & Comparison Dashboard
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
import plotly.express as px

st.set_page_config(
    page_title="HuMob 2026: Per-Origin Flow Matching Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

PACKAGE_ROOT = Path(__file__).resolve().parent
SHARED_DATA  = PACKAGE_ROOT.parent / 'humob2026_destination_diffusion' / 'data'

OD_PKL       = SHARED_DATA / 'processed' / 'od_time_series.pkl'
DT_PKL       = SHARED_DATA / 'processed' / 'dates.pkl'
DIFF_PRED    = SHARED_DATA / 'outputs' / 'dest1476_predictions_v2.tsv'
BASE_PKL     = SHARED_DATA / 'outputs' / 'full_year_baseline.pkl'
META_DEST    = SHARED_DATA / 'outputs' / 'meta_1476.pkl'

FM_PRED      = PACKAGE_ROOT / 'data' / 'outputs' / 'origin_fm_predictions.tsv'
FM_META      = PACKAGE_ROOT / 'data' / 'outputs' / 'origin_fm_meta.pkl'

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

@st.cache_data
def load_base_structures():
    with open(OD_PKL, 'rb') as f: od_ts = pickle.load(f)
    with open(DT_PKL, 'rb') as f: dates_str = pickle.load(f)
    base_data = pickle.load(open(BASE_PKL, 'rb')) if BASE_PKL.exists() else {}
    meta_d = pickle.load(open(META_DEST, 'rb')) if META_DEST.exists() else {}
    meta_f = pickle.load(open(FM_META, 'rb')) if FM_META.exists() else {}
    return od_ts, dates_str, base_data, meta_d, meta_f

od_ts, dates_str, base_data, meta_d, meta_f = load_base_structures()

@st.cache_data(ttl=5)
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

fm_pred_data   = parse_pred_tsv(FM_PRED)
diff_pred_data = parse_pred_tsv(DIFF_PRED)

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}
obs_date_to_idx = {d: i for i, d in enumerate(dates_str)}

st.title("🌊 HuMob 2026: Per-Origin Flow Matching Dashboard")
st.markdown("""
<div style="background-color: #1e293b; padding: 12px 18px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 20px;">
  <span style="font-size: 16px; font-weight: bold; color: #38bdf8;">🚀 模型表現總覽：</span>
  <span style="color: #cbd5e1; margin-left: 10px;">
    <b>Flow Matching (基礎推論)</b>: Combined NRMSE = <b style="color: #34d399;">0.34286</b> (Off-Diag: <b style="color: #38bdf8;">0.51917</b> | Diag: 0.16656) | 
    <b>Destination Diffusion (終極調優)</b>: Combined NRMSE = <b style="color: #facc15;">0.31401</b> (Off-Diag: 0.51141 | Diag: 0.11660)
  </span>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📈 OD 路線時序波形與模型對比", 
    "🗺️ 2D 起點-全島目的地 Flow Matching 空間分佈", 
    "📊 9-Class 災害分類多子圖全覽"
])

def in_eval_bbox(g_str):
    if g_str == '-1_-1': return False
    pts = g_str.split('_')
    if len(pts) != 2: return False
    try:
        gx, gy = int(pts[0]), int(pts[1])
        return (1 <= gx <= 70) and (1 <= gy <= 100)
    except:
        return False

route_class_map = meta_d.get('route_class_info', {})

with tab1:
    st.sidebar.markdown("### 🎛️ 預測模型版本切換")
    model_view_option = st.sidebar.radio(
        "選擇預測曲線顯示版本:",
        [
            "🌊 Per-Origin Flow Matching (ODE 推論)",
            "🌌 Destination Diffusion (1476-CH 自適應)",
            "📊 同步對比模式 (Flow Matching vs Diffusion)"
        ],
        index=2
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 導航篩選器 (9 大災害分類)")
    
    class_category_options = [
        "🌟 全部類別 (All 9 Classes)",
        "⬛ Class 1: (人流量低下區) Persistent Zero",
        "⛺ Class 2: (災後臨時避難區) Temporary Shelter & Assembly",
        "🏚️ Class 3: (地震重災與長期衰退區) Heavy Damage & Long-term Decline",
        "📈 Class 4: (部分恢復區) Partial Recovery",
        "🔄 Class 5: (大致恢復區) General Recovered",
        "⚖️ Class 6: (常態平穩區) Normal Steady",
        "💥 Class 7: (短期流入暴增後消退) Short-term Surge & Dissipation",
        "🌫️ Class 8: (部分消退區) Partial Dissipation",
        "🚀 Class 9: (長期增加區) Persistent Increase"
    ]
    
    sel_class_label = st.sidebar.selectbox("🏷️ 1. 選擇 9 大災害物理類別:", class_category_options, index=0)
    
    sel_class_id = None
    for target_id in range(1, 10):
        if f"Class {target_id}:" in sel_class_label:
            sel_class_id = target_id
            break

    volume_options = [
        "🌟 所有流量 (All Volumes)",
        "🔥 A 級極高流量 (Mean >= 50)",
        "📈 B 級高流量 (Mean 10 ~ 50)",
        "📊 C 級中流量 (Mean 3 ~ 10)",
        "📉 D 級低流量 (Mean 1 ~ 3)",
        "🧊 E 級稀疏流量 (Mean < 1)"
    ]
    sel_vol_label = st.sidebar.selectbox("📏 2. 人流數量分類過濾:", volume_options, index=0)

    filtered_pairs_map = {}
    for pair_k in od_ts.keys():
        if pair_k.startswith('-1_-1-'): o, d = '-1_-1', pair_k[6:]
        elif pair_k.endswith('--1_-1'): o, d = pair_k[:-6], '-1_-1'
        else: pts = pair_k.split('-'); o, d = pts[0], pts[1]
        
        c_info = route_class_map.get(pair_k, {})
        c_id = c_info.get('class_id', 6)
        if sel_class_id is not None and c_id != sel_class_id:
            continue
            
        if sel_vol_label != "🌟 所有流量 (All Volumes)":
            raw_data = od_ts.get(pair_k, [])
            valid_v = [x for x in raw_data if not np.isnan(x)]
            mean_v = np.mean(valid_v) if valid_v else 0.0
            
            if "A 級" in sel_vol_label and mean_v < 50.0: continue
            if "B 級" in sel_vol_label and (mean_v < 10.0 or mean_v >= 50.0): continue
            if "C 級" in sel_vol_label and (mean_v < 3.0 or mean_v >= 10.0): continue
            if "D 級" in sel_vol_label and (mean_v < 1.0 or mean_v >= 3.0): continue
            if "E 級" in sel_vol_label and mean_v >= 1.0: continue
            
        if o not in filtered_pairs_map:
            filtered_pairs_map[o] = []
        filtered_pairs_map[o].append((d, pair_k))

    total_matching_routes = sum(len(d_list) for d_list in filtered_pairs_map.values())
    st.sidebar.markdown(f"**符合分類之路線數: `{total_matching_routes:,}` 條**")

    if not filtered_pairs_map:
        st.sidebar.warning("⚠️ 該篩選條件下無對應路線，請切換分類。")
        selected_pair = None
    else:
        sorted_origins = sorted(filtered_pairs_map.keys(), key=lambda o: -len(filtered_pairs_map[o]))
        
        default_orig_idx = 0
        for candidate in ["39_46", "30_69", "41_47", "39_44", "61_63", "21_30"]:
            if candidate in sorted_origins:
                default_orig_idx = sorted_origins.index(candidate)
                break
                
        sel_origin = st.sidebar.selectbox("🚩 3. 選擇出發起點 (Origin Grid):", sorted_origins, index=default_orig_idx)

        avail_dests_tuples = filtered_pairs_map.get(sel_origin, [])
        formatted_dests = []
        dest_to_pair = {}
        for d, pk in avail_dests_tuples:
            label = f"{d} (🏠 自身停留 Stay)" if d == sel_origin else f"{d} (🚀 跨區流動)"
            if d == sel_origin:
                formatted_dests.insert(0, label)
            else:
                formatted_dests.append(label)
            dest_to_pair[label] = pk

        sel_dest_label = st.sidebar.selectbox("🎯 4. 選擇到達終點 (Destination Grid):", formatted_dests, index=0)
        selected_pair = dest_to_pair.get(sel_dest_label)

    if selected_pair:
        pts = selected_pair.split('-')
        o_str = '-1_-1' if selected_pair.startswith('-1_-1-') else pts[0]
        d_str = pts[1].replace('_', '-') if selected_pair.startswith('-1_-1-') else pts[1]

        raw_arr = od_ts.get(selected_pair)
        b_366 = base_data.get(selected_pair)

        valid_v = [x for x in raw_arr if not np.isnan(x)] if raw_arr is not None else []
        mean_v = np.mean(valid_v) if valid_v else 0.0
        max_v = np.max(valid_v) if valid_v else 0.0
        is_diag = (o_str == d_str)
        cls_info = meta_d.get('route_class_info', {}).get(selected_pair, {}).get('class_name', 'Class 6: Normal Steady')

        # 頂部資訊卡片
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🚩 起點 → 終點", f"{o_str} → {d_str}")
        col2.metric("🏷️ 災害物理分類", cls_info.split(':')[0])
        col3.metric("📈 歷史平均日流", f"{mean_v:.2f} 人")
        col4.metric("🔄 路線屬性", "🏠 自身停留 (Stay)" if is_diag else "🚀 跨區流動 (Cross)")

        y_obs = [np.nan] * 366
        if raw_arr is not None:
            for d_str_k, oi in obs_date_to_idx.items():
                if d_str_k in cal_date_to_idx:
                    val = raw_arr[oi]
                    if d_str_k in EXCLUDED_DATES:
                        val = np.nan
                    elif np.isnan(val):
                        val = 0.0
                    y_obs[cal_date_to_idx[d_str_k]] = val

        b_vals = b_366 if (b_366 is not None and isinstance(b_366, (list, np.ndarray))) else [0.0] * 366

        fm_vals = [np.nan] * 366
        for d_str_k, od_map in fm_pred_data.items():
            if d_str_k in cal_date_to_idx:
                val = od_map.get(o_str, {}).get(d_str, np.nan)
                fm_vals[cal_date_to_idx[d_str_k]] = val

        diff_vals = [np.nan] * 366
        for d_str_k, od_map in diff_pred_data.items():
            if d_str_k in cal_date_to_idx:
                val = od_map.get(o_str, {}).get(d_str, np.nan)
                diff_vals[cal_date_to_idx[d_str_k]] = val

        df_ts = pd.DataFrame({
            "Date": [datetime.strptime(d, "%Y%m%d") for d in cal_dates],
            "Actual": y_obs,
            "Baseline": b_vals,
            "Flow_Matching": fm_vals,
            "Diffusion_v2": diff_vals
        })

        fig = go.Figure()

        # 盲區陰影
        fig.add_vrect(
            x0=datetime(2024, 2, 1), x1=datetime(2024, 4, 30),
            fillcolor="rgba(16, 185, 129, 0.08)", line_width=0,
            annotation_text="90 天評測盲區 (2024/02/01 ~ 2024/04/30)",
            annotation_position="top left",
            annotation_font=dict(size=12, color="#10B981")
        )

        # 錨點標記
        fig.add_vrect(
            x0=datetime(2024, 1, 20), x1=datetime(2024, 1, 31),
            fillcolor="rgba(239, 68, 68, 0.12)", line_width=1, line_dash="dash", line_color="#EF4444",
            annotation_text="⚓ P_jan (災後左錨點)", annotation_position="bottom left", annotation_font=dict(size=10, color="#EF4444")
        )
        fig.add_vrect(
            x0=datetime(2024, 4, 1), x1=datetime(2024, 4, 14),
            fillcolor="rgba(56, 189, 248, 0.12)", line_width=1, line_dash="dash", line_color="#38BDF8",
            annotation_text="⚓ P_apr (復原右錨點)", annotation_position="bottom right", annotation_font=dict(size=10, color="#38BDF8")
        )

        # 真實觀測值
        fig.add_trace(go.Scatter(
            x=df_ts["Date"], y=df_ts["Actual"],
            mode="lines+markers", name="🔴 Actual (真實觀測，含4月GT)",
            line=dict(color="#EF4444", width=1.5),
            marker=dict(size=4, color="#EF4444")
        ))

        # Baseline
        fig.add_trace(go.Scatter(
            x=df_ts["Date"], y=df_ts["Baseline"],
            mode="lines", name="⚪ 9-Class 物理平滑 Baseline",
            line=dict(color="#94A3B8", width=2.0, dash="dash")
        ))

        # 依選擇繪製模型
        if "Flow Matching" in model_view_option or "對比模式" in model_view_option:
            fig.add_trace(go.Scatter(
                x=df_ts["Date"], y=df_ts["Flow_Matching"],
                mode="lines", name="🌊 Per-Origin Flow Matching 預測",
                line=dict(color="#10B981", width=2.8)
            ))

        if "Destination Diffusion" in model_view_option or "對比模式" in model_view_option:
            fig.add_trace(go.Scatter(
                x=df_ts["Date"], y=df_ts["Diffusion_v2"],
                mode="lines", name="🌌 Destination Diffusion v2 預測",
                line=dict(color="#38BDF8", width=2.5, dash="dot")
            ))

        fig.update_layout(
            title=f"<b>時序波形分析: {selected_pair} ({cls_info})</b>",
            xaxis_title="日期",
            yaxis_title="人流量 (人次)",
            template="plotly_dark",
            height=580,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### 🗺️ 2D 全島目的地 Flow Matching 空間分佈 (Vector Field)")
    st.markdown("觀察選定起點在全島 $70 \\times 100$ 網格上的目的地生成強度與流向分佈。")
    
    col_d1, col_d2 = st.columns([1, 3])
    with col_d1:
        sel_heat_date = st.selectbox("📅 選擇推論日期:", blind_zone, index=0)
        sel_heat_origin = st.selectbox("🚩 選擇展示起點:", sorted_origins if filtered_pairs_map else ["39_46"], index=0)
    
    # 建立 2D 目的地矩陣
    grid_mat = np.zeros((70, 100), dtype=np.float32)
    day_od_map = fm_pred_data.get(sel_heat_date, {}).get(sel_heat_origin, {})
    for d_k, v in day_od_map.items():
        if d_k != '-1_-1' and '_' in d_k:
            try:
                gx, gy = map(int, d_k.split('_'))
                if 1 <= gx <= 70 and 1 <= gy <= 100:
                    grid_mat[gx - 1, gy - 1] = float(v)
            except: pass
            
    fig_heat = px.imshow(
        grid_mat.T,
        origin='lower',
        labels=dict(x="X Coordinate (1~70)", y="Y Coordinate (1~100)", color="Flow Volume"),
        color_continuous_scale="Viridis",
        title=f"起點 {sel_heat_origin} 於 {sel_heat_date} 之全島目的地 Flow Matching 生成熱力圖"
    )
    fig_heat.update_layout(template="plotly_dark", height=650)
    st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    st.markdown("### 📊 9-Class 災害分類多子圖全覽 (Dark Theme)")
    st.markdown("以下為 Per-Origin Flow Matching 在 9 大物理分類代表路線上的 90 天完整盲區生成曲線。")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### 🏠 對角線 (自身停留 Stay 路線)")
        diag_img_path = PACKAGE_ROOT / 'data' / 'outputs' / 'fm_9class_diagonal.png'
        if diag_img_path.exists():
            st.image(str(diag_img_path), use_container_width=True)
        else:
            st.info("圖表生成中...")
            
    with col_g2:
        st.markdown("#### 🚀 非對角線 (跨區流動 Cross 路線)")
        offdiag_img_path = PACKAGE_ROOT / 'data' / 'outputs' / 'fm_9class_offdiagonal.png'
        if offdiag_img_path.exists():
            st.image(str(offdiag_img_path), use_container_width=True)
        else:
            st.info("圖表生成中...")
