"""Shared UTC time conventions; interval-start labels internally."""
import pandas as pd
STEP = pd.Timedelta(minutes=15)
MARKET_TZ = "Etc/GMT-10"  # fixed UTC+10, no daylight saving
REGIONS = ("NSW1", "QLD1", "VIC1", "SA1", "TAS1")
NAMES = {"NSW1":"New South Wales / ACT","QLD1":"Queensland","VIC1":"Victoria","SA1":"South Australia","TAS1":"Tasmania","NEM":"NEM · consolidated"}
def utc(value=None):
    t = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if t.tzinfo is None:
        raise ValueError("Timestamp must include timezone")
    return t.tz_convert("UTC")
def stamp(value):
    return utc(value).isoformat()
