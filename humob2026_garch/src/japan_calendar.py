import math
from datetime import datetime, timedelta

# Complete Japanese National Holidays & Major Festival Windows for 2023-11-01 to 2024-10-31
JAPAN_HOLIDAYS = {
    # 2023 Historical Period
    '20231103': '文化の日 (Culture Day)',
    '20231123': '勤労感謝の日 (Labor Thanksgiving Day)',
    '20231229': '年末年始 (New Year Eve Window)',
    '20231230': '年末年始 (New Year Eve Window)',
    '20231231': '年末年始 (New Year Eve Window)',
    
    # 2024 Historical & Disaster Period
    '20240101': '元日 (New Year Day)',
    '20240102': '年末年始 (New Year Holiday)',
    '20240103': '年末年始 (New Year Holiday)',
    '20240108': '成人の日 (Coming of Age Day)',
    
    # 2024 Blind Zone Period (Feb ~ Apr)
    '20240211': '建国記念の日 (National Foundation Day)',
    '20240212': '振替休日 (Substitute Holiday)',
    '20240223': '天皇誕生日 (Emperor Birthday)',
    '20240320': '春分の日 (Vernal Equinox Day)',
    '20240429': '昭和の日 (Showa Day)',
    
    # 2024 Post-Evaluation Period (May ~ Oct)
    '20240503': '憲法記念日 (Constitution Memorial Day)',
    '20240504': 'みどりの日 (Greenery Day)',
    '20240505': 'こどもの日 (Children Day)',
    '20240506': '振替休日 (Substitute Holiday)',
    '20240715': '海の日 (Marine Day)',
    '20240811': '山の日 (Mountain Day)',
    '20240812': '振替休日 (Substitute Holiday)',
    '20240813': 'お盆 (Obon Festival)',
    '20240814': 'お盆 (Obon Festival)',
    '20240815': 'お盆 (Obon Festival)',
    '20240816': 'お盆 (Obon Festival)',
    '20240916': '敬老の日 (Respect for the Aged Day)',
    '20240922': '秋分の日 (Autumnal Equinox Day)',
    '20240923': '振替休日 (Substitute Holiday)',
    '20241014': 'スポーツの日 (Sports Day)',
}

def get_holiday_features(date_str: str):
    """
    Extracts complete calendar regime features for any given date:
    - is_holiday: True if date is a public holiday / festival
    - is_holiday_eve: True if tomorrow is a holiday
    - is_consecutive: True if part of a 3+ day long weekend / holiday block
    - holiday_dist: signed distance in days to nearest holiday (-3 .. +3)
    """
    dt = datetime.strptime(date_str, '%Y%m%d')
    is_hol = date_str in JAPAN_HOLIDAYS
    
    # Tomorrow
    dt_tom = dt + timedelta(days=1)
    is_eve = dt_tom.strftime('%Y%m%d') in JAPAN_HOLIDAYS
    
    # Yesterday
    dt_yest = dt - timedelta(days=1)
    is_post = dt_yest.strftime('%Y%m%d') in JAPAN_HOLIDAYS
    
    # Check if 3-day weekend
    dow = dt.weekday() # 0=Mon, ..., 6=Sun
    is_consec = False
    if is_hol and dow in [0, 4]: # Mon holiday or Fri holiday
        is_consec = True
    elif dow == 5 and (dt + timedelta(days=2)).strftime('%Y%m%d') in JAPAN_HOLIDAYS: # Sat before Mon hol
        is_consec = True
    elif dow == 6 and (dt + timedelta(days=1)).strftime('%Y%m%d') in JAPAN_HOLIDAYS: # Sun before Mon hol
        is_consec = True
    elif is_hol:
        is_consec = True
        
    # Distance to nearest holiday
    min_dist = 999
    for h_str in JAPAN_HOLIDAYS.keys():
        h_dt = datetime.strptime(h_str, '%Y%m%d')
        diff = (dt - h_dt).days
        if abs(diff) < abs(min_dist):
            min_dist = diff
            
    return {
        'is_holiday': is_hol,
        'holiday_name': JAPAN_HOLIDAYS.get(date_str, 'Regular Day'),
        'is_holiday_eve': is_eve,
        'is_holiday_post': is_post,
        'is_consecutive': is_consec,
        'holiday_dist': min_dist if abs(min_dist) <= 7 else 99
    }
