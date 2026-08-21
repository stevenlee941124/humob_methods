import os
import pickle
from pathlib import Path

MIN_X, MAX_X = 30, 70
MIN_Y, MAX_Y = 35, 70

EXCLUDED_DATES = {
    '20231126', '20231130', '20231201', '20231203', '20231204', '20231205',
    '20231214', '20240118', '20240123', '20240124', '20240202', '20240305',
    '20240408', '20240426', '20240529', '20240708'
}

def is_in_bbox(grid_str):
    try:
        x, y = map(int, grid_str.split('_'))
        return MIN_X <= x <= MAX_X and MIN_Y <= y <= MAX_Y
    except:
        return False

def parse_tsv(filepath):
    data = {}
    filepath = Path(filepath)
    if not filepath.exists(): return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            date_str = parts[0]
            try:
                raw = parts[1].replace(': NA', ': None').replace(':NA', ':None')
                od = eval(raw, {'__builtins__': {}}, {'None': None})
                if od is not None:
                    data[date_str] = od
            except:
                pass
    return data
