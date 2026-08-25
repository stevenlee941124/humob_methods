"""
===============================================================================
HuMob 2026: Japan Holiday Calendar Module
===============================================================================
"""
from datetime import datetime, timedelta

JAPAN_HOLIDAYS = {
    '20231103': 'Culture Day',
    '20231123': 'Labor Thanksgiving Day',
    '20240101': "New Year's Day",
    '20240108': 'Coming of Age Day',
    '20240211': 'National Foundation Day',
    '20240212': 'Substitute Holiday',
    '20240223': "Emperor's Birthday",
    '20240320': 'Vernal Equinox Day',
    '20240429': 'Showa Day',
    '20240503': 'Constitution Memorial Day',
    '20240504': 'Greenery Day',
    '20240505': "Children's Day",
    '20240506': 'Substitute Holiday',
    '20240715': 'Marine Day',
    '20240811': 'Mountain Day',
    '20240812': 'Substitute Holiday',
    '20240916': 'Respect for the Aged Day',
    '20240922': 'Autumnal Equinox Day',
    '20240923': 'Substitute Holiday',
    '20241014': 'Sports Day',
}

def get_holiday_features(date_str: str) -> dict:
    dt = datetime.strptime(date_str, '%Y%m%d')
    is_holiday = date_str in JAPAN_HOLIDAYS
    next_day_str = (dt + timedelta(days=1)).strftime('%Y%m%d')
    is_holiday_eve = next_day_str in JAPAN_HOLIDAYS or dt.weekday() == 4
    prev_day_str = (dt - timedelta(days=1)).strftime('%Y%m%d')
    is_post_holiday = prev_day_str in JAPAN_HOLIDAYS
    is_weekend = dt.weekday() in (5, 6)
    is_extended_off = is_holiday or is_weekend

    return {
        'is_holiday': is_holiday,
        'is_holiday_eve': is_holiday_eve,
        'is_post_holiday': is_post_holiday,
        'is_weekend': is_weekend,
        'is_extended_off': is_extended_off,
        'weekday': dt.weekday(),
        'month': dt.month,
        'day': dt.day
    }
