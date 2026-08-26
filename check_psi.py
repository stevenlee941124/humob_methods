import pickle, numpy as np, pandas as pd
from datetime import datetime

OD = "humob2026_destination_diffusion/data/processed/od_time_series.pkl"
DATES = "humob2026_destination_diffusion/data/processed/dates.pkl"
with open(OD, 'rb') as f: od_ts = pickle.load(f)
with open(DATES, 'rb') as f: dates_str = pickle.load(f)

raw = od_ts['39_46-39_46']
train_days = [(dates_str[i], raw[i]) for i in range(len(raw)) if dates_str[i] < '20240101' and not np.isnan(raw[i])]
overall_m = np.mean([v for _, v in train_days])
wd_map = {w: [] for w in range(7)}
for d_str, v in train_days:
    wd_map[datetime.strptime(d_str, '%Y%m%d').weekday()].append(v)
psi_7 = np.zeros(7)
for w in range(7):
    psi_7[w] = np.mean(wd_map[w]) - overall_m

print("psi_7:", psi_7)
