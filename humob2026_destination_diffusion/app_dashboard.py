"""
===============================================================================
HuMob 2026: (1476, 70, 100) Multi-Channel Spatial Diffusion Analysis Dashboard
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
    page_title="HuMob 2026: 1476-Channel Destination Diffusion Dashboard",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

PACKAGE_ROOT = Path(__file__).resolve().parent

OD_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'od_time_series.pkl'
DT_PKL    = PACKAGE_ROOT / 'data' / 'processed' / 'dates.pkl'
PRED_TSV  = PACKAGE_ROOT / 'data' / 'outputs' / 'dest1476_predictions.tsv'
BASE_PKL  = PACKAGE_ROOT / 'data' / 'outputs' / 'full_year_baseline.pkl'
META_PKL  = PACKAGE_ROOT / 'data' / 'outputs' / 'meta_1476.pkl'

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
    meta = pickle.load(open(META_PKL, 'rb')) if META_PKL.exists() else {}
    return od_ts, dates_str, base_data, meta

od_ts, dates_str, base_data, meta = load_base_structures()

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
<div style="background: linear-gradient(90deg, #0A192F, #1E3A8A); padding: 18px 24px; border-radius: 8px; margin-bottom: 20px; color: white;">
    <h2 style="margin: 0; color: #38BDF8;">🌌 HuMob 2026: (1476, 70, 100) 多光譜空間吸引力場擴散模型儀表板</h2>
    <p style="margin: 6px 0 0 0; color: #E2E8F0; font-size: 14px;">
        9 大類別災害物理動力學 Baseline (IQR 離群清洗 + 雙錨點三次 S 曲線轉移) + 1,476 通道 Spatial Diffusion
    </p>
</div>
""", unsafe_allow_html=True)

# ── 頂部模型指標看板 ──────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🏆 9-Class 物理 Baseline", "0.27048", delta="-0.0114 vs 舊版", delta_color="inverse")
with c2:
    st.metric("停留流動 (Diag) NRMSE", "0.13212", "實體 RMSE: 3.51 人 🔥")
with c3:
    st.metric("跨區流動 (Off-Diag) NRMSE", "0.40884", "實體 RMSE: 0.0072 人")
with c4:
    st.metric("🎯 9 大災害分類覆蓋", "15,129 路線", "雙錨點 OD 矩陣動態轉移")

tab1, tab2 = st.tabs(["📈 OD 路線時序波形與 9-Class Baseline 檢視", "🗺️ 2D 全域空間引力集水區熱力圖"])

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

# 建立路線與分類資訊字典
route_class_map = meta.get('route_class_info', {})

with tab1:
    st.sidebar.markdown("### 🧭 導航篩選器 (9 大災害分類優先)")
    
    # ── 1. 先選擇 9 大災害物理分類 (精確區分 1 ~ 9 類) ──
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
    
    # 解析選中的 class_id (1 ~ 9)
    sel_class_id = None
    for target_id in range(1, 10):
        if f"Class {target_id}:" in sel_class_label:
            sel_class_id = target_id
            break

    # ── 1.5. 人流數量分類過濾 ──
    volume_options = [
        "🌟 所有流量 (All Volumes)",
        "🔥 A 級極高流量 (Mean >= 50)",
        "📈 B 級高流量 (Mean 10 ~ 50)",
        "📊 C 級中流量 (Mean 3 ~ 10)",
        "📉 D 級低流量 (Mean 1 ~ 3)",
        "🧊 E 級稀疏流量 (Mean < 1)"
    ]
    sel_vol_label = st.sidebar.selectbox("📏 1.5. 人流數量分類過濾:", volume_options, index=0)

    # 空間範圍過濾
    scope_only_eval = st.sidebar.checkbox("🎯 僅顯示官方評測範圍 (X: 30~70, Y: 35~70)", value=True)

    # ── 2. 根據 9 大類別與流量篩選可用的起點與終點 ──
    filtered_pairs_map = {}
    for pair_k in od_ts.keys():
        if pair_k.startswith('-1_-1-'): o, d = '-1_-1', pair_k[6:]
        elif pair_k.endswith('--1_-1'): o, d = pair_k[:-6], '-1_-1'
        else: pts = pair_k.split('-'); o, d = pts[0], pts[1]
        
        # 空間範圍過濾
        if scope_only_eval and not (in_eval_bbox(o) and in_eval_bbox(d)):
            continue
            
        # 類別過濾
        c_info = route_class_map.get(pair_k, {})
        c_id = c_info.get('class_id', 6)
        if sel_class_id is not None and c_id != sel_class_id:
            continue
            
        # 人流數量過濾
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

    # 統計當前篩選下的路線總數
    total_matching_routes = sum(len(d_list) for d_list in filtered_pairs_map.values())
    st.sidebar.markdown(f"**符合分類之路線數: `{total_matching_routes:,}` 條**")

    if not filtered_pairs_map:
        st.sidebar.warning("⚠️ 該分類在當前空間範圍下無對應路線，請切換分類或取消勾選範圍。")
        selected_pair = None
    else:
        # 排序起點 (按流出路線數量排序)
        sorted_origins = sorted(filtered_pairs_map.keys(), key=lambda o: -len(filtered_pairs_map[o]))
        
        default_orig_idx = 0
        for candidate in ["39_46", "30_69", "41_47", "39_44"]:
            if candidate in sorted_origins:
                default_orig_idx = sorted_origins.index(candidate)
                break
                
        sel_origin = st.sidebar.selectbox("🚩 2. 選擇出發起點 (Origin Grid):", sorted_origins, index=default_orig_idx)

        # 終點清單
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
        max_v = np.max(valid_v) if valid_v else 0.0
        std_v = np.std(valid_v) if valid_v else 0.0
        is_diag = (o_str == d_str)
        c_idx = meta.get('dest_to_channel', {}).get(d_str, "非官方評測框")
        cls_info = meta.get('route_class_info', {}).get(selected_pair, {}).get('class_name', 'Class 6: Normal Steady')

        # ── 路線核心屬性 Badges ──
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.info(f"**📌 路線類型**: `{'🏠 停留流動 (Stay/Diag)' if is_diag else '🚀 跨區流動 (Off-Diag)'}`")
        with b2:
            st.info(f"**📊 歷史人流特徵**: 均值 `{mean_v:.1f}` 人 | 波動 $\\sigma$: `{std_v:.1f}` 人")
        with b3:
            st.info(f"**🏷️ 9 大災害物理類別**: `{cls_info}`")
        with b4:
            st.info(f"**🎯 目的地通道**: `Channel {c_idx} / 1476` (終點: {d_str})")

        # 組合時間序列繪圖資料
        y_obs = [np.nan] * 366
        if raw_arr is not None:
            for d_str_k, oi in obs_date_to_idx.items():
                if d_str_k in cal_date_to_idx:
                    val = raw_arr[oi]
                    if np.isnan(val):
                        val = 0.0 # 真實世界沒記錄代表 0 人，連起線來
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

        # 1. 盲區陰影 (2024/02/01 ~ 2024/04/30)
        fig.add_vrect(
            x0=datetime(2024, 2, 1), x1=datetime(2024, 4, 30),
            fillcolor="rgba(56, 189, 248, 0.08)", line_width=0,
            annotation_text="90 天評測盲區 (2024/02/01 ~ 2024/04/30)",
            annotation_position="top left",
            annotation_font=dict(size=12, color="#38BDF8")
        )

        # 2. 雙錨點標記 (P_jan: 1/20~1/31, P_apr: 4/1~4/14)
        fig.add_vrect(
            x0=datetime(2024, 1, 20), x1=datetime(2024, 1, 31),
            fillcolor="rgba(239, 68, 68, 0.12)", line_width=1, line_dash="dash", line_color="#EF4444",
            annotation_text="⚓ P_jan (災後應急左錨點)", annotation_position="bottom left", annotation_font=dict(size=10, color="#EF4444")
        )
        fig.add_vrect(
            x0=datetime(2024, 4, 1), x1=datetime(2024, 4, 14),
            fillcolor="rgba(16, 185, 129, 0.12)", line_width=1, line_dash="dash", line_color="#10B981",
            annotation_text="⚓ P_apr (待復原右錨點)", annotation_position="bottom right", annotation_font=dict(size=10, color="#10B981")
        )

        # 3. 繪製曲線
        fig.add_trace(go.Scatter(
            x=df_ts["Date"], y=df_ts["Actual"],
            mode="lines+markers", name="🔴 Actual (真實觀測值，含4月GT)",
            line=dict(color="#EF4444", width=1.5),
            marker=dict(size=4, color="#EF4444")
        ))

        fig.add_trace(go.Scatter(
            x=df_ts["Date"], y=df_ts["Baseline"],
            mode="lines", name="⚪ 9-Class 物理 Baseline (雙錨點 S 曲線轉移)",
            line=dict(color="#94A3B8", width=2.5, dash="dash")
        ))

        fig.add_trace(go.Scatter(
            x=df_ts["Date"], y=df_ts["Pred"],
            mode="lines", name="🟢 1476-Diffusion (空間擴散融合預測值)",
            line=dict(color="#38BDF8", width=2.5)
        ))

        fig.update_layout(
            title=f"<b>【{selected_pair}】 366 天波形與 9 大類別災害物理 Baseline (起點: {o_str} ──> 終點: {d_str})</b>",
            xaxis_title="日期 (Date)",
            yaxis_title="人流人次 (Persons)",
            height=530,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🗺️ 2D 全域空間引力集水區與熱力圖分析")
    st.markdown("直觀探索 $70 \\times 100$ 能登半島全島空間人流如何受災難與空間幾何引力影響，流向指定的評測目標目的地。")

    m_c1, m_c2 = st.columns([1, 2])

    with m_c1:
        st.markdown("### 🎛️ 地圖控制項")
        dest_options = []
        for d_k in meta.get('dest_grid_list', []):
            c_idx = meta['dest_to_channel'][d_k]
            routes_in = [r for r in meta.get('active_routes', []) if r['c_idx'] == c_idx]
            if routes_in:
                dest_options.append((d_k, len(routes_in)))

        dest_options.sort(key=lambda x: -x[1])
        dest_display_list = [f"{d} ({cnt} 條流入路線)" for d, cnt in dest_options]

        sel_dest_str = st.selectbox("1. 選擇目標目的地網格 (Destination Grid):", dest_display_list, index=0)
        cur_dest = sel_dest_str.split(" ")[0]
        cur_c_idx = meta['dest_to_channel'][cur_dest]

        cur_routes = [r for r in meta.get('active_routes', []) if r['c_idx'] == cur_c_idx]

        st.write(f"**該目的地通道**: `Channel {cur_c_idx} / 1476`")
        st.write(f"**全島流入路線數**: `{len(cur_routes)}` 條")

        dx, dy = map(int, cur_dest.split('_'))
        st.write(f"**目的地坐標**: `X = {dx}, Y = {dy}`")

    with m_c2:
        catchment_map = np.zeros((70, 100), dtype=np.float32)
        for r in cur_routes:
            ox, oy = r['ox'], r['oy']
            pair_k = r['pair_key']
            raw = od_ts.get(pair_k)
            if raw is not None:
                valid_v = [v for v in raw if not np.isnan(v)]
                catchment_map[ox, oy] = np.mean(valid_v) if valid_v else 0.0

        fig_map = px.imshow(
            catchment_map.T,
            labels=dict(x="Longitude 網格 (X)", y="Latitude 網格 (Y)", color="流入常態人流 (人)"),
            x=list(range(1, 71)),
            y=list(range(1, 101)),
            color_continuous_scale="Viridis",
            origin="lower",
            title=f"<b>前往目的地 【{cur_dest}】 的全島 2D 起點引力集水區熱力圖</b>"
        )

        fig_map.add_trace(go.Scatter(
            x=[dx], y=[dy],
            mode="markers+text",
            marker=dict(color="#EF4444", size=15, symbol="star", line=dict(color="white", width=2)),
            name=f"🎯 目的地 {cur_dest}",
            text=[f"🎯 {cur_dest}"],
            textposition="top center",
            textfont=dict(color="#EF4444", size=13)
        ))

        fig_map.update_layout(height=600, xaxis=dict(dtick=10), yaxis=dict(dtick=10))
        st.plotly_chart(fig_map, use_container_width=True)
