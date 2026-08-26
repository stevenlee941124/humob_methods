"""
===============================================================================
HuMob 2026: 2D Spatial Diffusion Analysis Dashboard
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
BASE_PKL  = PACKAGE_ROOT / 'data' / 'outputs' / 'full_year_baseline.pkl'

@st.cache_data
def load_base_structures():
    with open(OD_PKL, 'rb') as f: od_ts = pickle.load(f)
    with open(DT_PKL, 'rb') as f: dates_str = pickle.load(f)
    base_data = pickle.load(open(BASE_PKL, 'rb')) if BASE_PKL.exists() else {}
    return od_ts, dates_str, base_data

od_ts, dates_str, base_data = load_base_structures()

@st.cache_data(ttl=2)
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

start_dt = datetime(2023, 11, 1)
cal_dates = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(366)]
cal_date_to_idx = {d: i for i, d in enumerate(cal_dates)}
obs_date_to_idx = {d: i for i, d in enumerate(dates_str)}

# ── 頁首 Header ──────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(90deg, #064E3B, #0F766E); padding: 18px 24px; border-radius: 8px; margin-bottom: 20px; color: white;">
    <h2 style="margin: 0; color: #6EE7B7;">🌌 HuMob 2026: 4 通道 2D 空間條件擴散模型儀表板</h2>
    <p style="margin: 6px 0 0 0; color: #E2E8F0; font-size: 14px;">
        9 大類別災害物理動力學 Baseline + 2D Spatial U-Net (DDPM/DDIM)
    </p>
</div>
""", unsafe_allow_html=True)

# ── 空間邊界判定函式 ──
def in_eval_bbox(g_str):
    if g_str == '-1_-1': return False
    pts = g_str.split('_')
    if len(pts) != 2: return False
    try:
        gx, gy = int(pts[0]), int(pts[1])
        return (30 <= gx <= 70) and (35 <= gy <= 70)
    except:
        return False

# 側邊欄
st.sidebar.markdown("### 🧭 導航篩選器 (9 大災害分類優先)")

META_PKL = PACKAGE_ROOT.parent / 'humob2026_destination_diffusion' / 'data' / 'outputs' / 'meta_1476.pkl'
route_class_map = pickle.load(open(META_PKL, 'rb')).get('route_class_info', {}) if META_PKL.exists() else {}

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

sel_class_label = st.sidebar.selectbox("🏷️ 1. 先選擇 9 大災害物理類別:", class_category_options, index=0)

sel_class_id = None
for target_id in range(1, 10):
    if f"Class {target_id}:" in sel_class_label:
        sel_class_id = target_id
        break

scope_only_eval = st.sidebar.checkbox("🎯 僅顯示官方評測範圍 (X: 30~70, Y: 35~70)", value=True)

filtered_pairs_map = {}
for pair_k in od_ts.keys():
    if pair_k.startswith('-1_-1-'): o, d = '-1_-1', pair_k[6:]
    elif pair_k.endswith('--1_-1'): o, d = pair_k[:-6], '-1_-1'
    else: pts = pair_k.split('-'); o, d = pts[0], pts[1]
    
    if scope_only_eval and not (in_eval_bbox(o) and in_eval_bbox(d)):
        continue
        
    c_info = route_class_map.get(pair_k, {})
    c_id = c_info.get('class_id', 6)
    if sel_class_id is not None and c_id != sel_class_id:
        continue
        
    if o not in filtered_pairs_map:
        filtered_pairs_map[o] = []
    filtered_pairs_map[o].append((d, pair_k))

total_matching_routes = sum(len(d_list) for d_list in filtered_pairs_map.values())
st.sidebar.markdown(f"**符合條件之路線數: `{total_matching_routes:,}` 條**")

if not filtered_pairs_map:
    st.sidebar.warning("⚠️ 該篩選條件下無對應路線。")
    selected_pair = None
else:
    sorted_origins = sorted(filtered_pairs_map.keys(), key=lambda o: -len(filtered_pairs_map[o]))
    default_orig_idx = 0
    for candidate in ["39_46", "30_69", "41_47", "39_44"]:
        if candidate in sorted_origins:
            default_orig_idx = sorted_origins.index(candidate)
            break
            
    sel_origin = st.sidebar.selectbox("🚩 2. 選擇出發起點 (Origin Grid):", sorted_origins, index=default_orig_idx)

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

    sel_dest_label = st.sidebar.selectbox("🎯 3. 選擇到達終點 (Destination Grid):", formatted_dests, index=0)
    selected_pair = dest_to_pair.get(sel_dest_label)

if selected_pair:
    st.sidebar.markdown(f"**當前檢視路線**: `{selected_pair}`")
    pts = selected_pair.split('-')
    o_str = '-1_-1' if selected_pair.startswith('-1_-1-') else pts[0]
    d_str = pts[1].replace('_', '-') if selected_pair.startswith('-1_-1-') else pts[1]

    raw_arr = od_ts.get(selected_pair)
    b_366 = base_data.get(selected_pair)

    valid_v = [x for x in raw_arr if not np.isnan(x)] if raw_arr is not None else []
    mean_v = np.mean(valid_v) if valid_v else 0.0
    std_v = np.std(valid_v) if valid_v else 0.0
    is_diag = (o_str == d_str)

    b1, b2, b3 = st.columns(3)
    with b1:
        st.info(f"**📌 路線類型**: `{'🏠 停留流動 (Stay/Diag)' if is_diag else '🚀 跨區流動 (Off-Diag)'}`")
    with b2:
        st.info(f"**📊 歷史人流特徵**: 均值 `{mean_v:.1f}` 人 | 波動 $\\sigma$: `{std_v:.1f}` 人")
    with b3:
        st.info(f"**🧭 路線端點**: 起點 `{o_str}` ──> 終點 `{d_str}`")

    y_obs = [np.nan] * 366
    if raw_arr is not None:
        for d_str_k, oi in obs_date_to_idx.items():
            if d_str_k in cal_date_to_idx:
                val = raw_arr[oi]
                if np.isnan(val): val = 0.0
                y_obs[cal_date_to_idx[d_str_k]] = val

    b_vals = b_366 if (b_366 is not None and isinstance(b_366, (list, np.ndarray))) else [0.0] * 366

    pred_vals = [np.nan] * 366
    for d_str_k, od_map in pred_data.items():
        if d_str_k in cal_date_to_idx:
            val = od_map.get(o_str, {}).get(d_str, np.nan)
            pred_vals[cal_date_to_idx[d_str_k]] = val

    df_ts = pd.DataFrame({
        "Date": [datetime.strptime(d, "%Y%m%d") for d in cal_dates],
        "Actual": y_obs,
        "Baseline": b_vals,
        "Pred": pred_vals
    })

    fig = go.Figure()

    fig.add_vrect(
        x0=datetime(2024, 2, 1), x1=datetime(2024, 4, 30),
        fillcolor="rgba(16, 185, 129, 0.08)", line_width=0,
        annotation_text="90 天評測盲區 (2024/02/01 ~ 2024/04/30)",
        annotation_position="top left",
        annotation_font=dict(size=12, color="#10B981")
    )

    fig.add_trace(go.Scatter(
        x=df_ts["Date"], y=df_ts["Actual"],
        mode="lines+markers", name="🔴 Actual (真實觀測值，含4月GT)",
        line=dict(color="#EF4444", width=1.5),
        marker=dict(size=4, color="#EF4444")
    ))

    fig.add_trace(go.Scatter(
        x=df_ts["Date"], y=df_ts["Baseline"],
        mode="lines", name="⚪ 9-Class 物理 Baseline",
        line=dict(color="#94A3B8", width=2.5, dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=df_ts["Date"], y=df_ts["Pred"],
        mode="lines", name="🟢 2D Spatial Diffusion 預測值",
        line=dict(color="#10B981", width=2.5)
    ))

    fig.update_layout(
        title=f"<b>【{selected_pair}】 366 天波形檢視 (起點: {o_str} ──> 終點: {d_str})</b>",
        xaxis_title="日期 (Date)",
        yaxis_title="人流人次 (Persons)",
        height=530,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
