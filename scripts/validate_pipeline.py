"""
validate_pipeline.py — Standalone data quality health check.

Reads master_df.csv and runs 6 checks. Prints a summary and exits with
code 1 if any check fails (suitable for CI / GitHub Actions alerting).

Usage:
    python3.11 scripts/validate_pipeline.py
"""

import sys
import os
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import MASTER_DF_PATH, SELECTED_FEATURES

# ─────────────────────────────────────────────
# Value range expectations
# ─────────────────────────────────────────────
RANGE_CHECKS = {
    'Close_BTC':                   (100,     500_000),
    'Sentiment_BTC_index_value':   (0,       100),
    'OnChain_Hash_Rate':           (0,       None),   # just > 0
    'OnChain_MVRV_Ratio':          (0,       20),
}

STALE_THRESHOLD_DAYS = 1   # last_date must be <= today - this
GAP_THRESHOLD_DAYS   = 3   # no gap in date index larger than this

failures = []
warnings = []


def fail(msg: str):
    failures.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg: str):
    warnings.append(msg)
    print(f"  INFO  {msg}")


def ok(msg: str):
    print(f"  OK    {msg}")


# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────

print(f"\nValidating: {MASTER_DF_PATH}\n")

if not os.path.exists(MASTER_DF_PATH):
    print(f"  FAIL  master_df.csv not found at {MASTER_DF_PATH}")
    sys.exit(1)

df = pd.read_csv(MASTER_DF_PATH, index_col='date', parse_dates=True)
df = df.sort_index()
today = date.today()
last_date = df.index.max().date()

print(f"Shape: {df.shape}  |  Last date: {last_date}  |  Today: {today}\n")


# ─────────────────────────────────────────────
# Check 1 — Currency
# ─────────────────────────────────────────────

print("Check 1 — Currency")
if last_date >= today - timedelta(days=STALE_THRESHOLD_DAYS):
    ok(f"last_date={last_date} is within {STALE_THRESHOLD_DAYS} day(s) of today")
else:
    fail(f"last_date={last_date} is stale — {(today - last_date).days} day(s) behind today")


# ─────────────────────────────────────────────
# Check 2 — Staleness (frozen / NaN in last row)
# ─────────────────────────────────────────────

print("\nCheck 2 — Staleness")
last_btc = df['Close_BTC'].iloc[-1] if 'Close_BTC' in df.columns else None

if last_btc is None:
    fail("Close_BTC column missing")
elif pd.isna(last_btc):
    fail(f"Close_BTC is NaN in last row ({last_date})")
else:
    ok(f"Close_BTC last value: {last_btc:,.0f}")

if 'Close_BTC' in df.columns and len(df) >= 3:
    last3 = df['Close_BTC'].iloc[-3:].dropna()
    if len(last3) == 3 and last3.nunique() == 1:
        fail(f"Close_BTC is identical across last 3 rows ({last3.iloc[0]:,.0f}) — possible silent ffill")
    else:
        ok("Close_BTC varies across last 3 rows (not frozen)")

# Non-BTC price columns: check last 5 weekday rows aren't all identical
# (catches the case where the pipeline ran before market close and ffill locked in a stale value)
for asset_col in ('Close_SP500', 'Close_NASDAQ', 'Close_GOLD', 'Close_DXY'):
    if asset_col not in df.columns:
        warn(f"{asset_col} not in master_df — skipping frozen-value check")
        continue
    weekday_series = df[asset_col][df.index.dayofweek < 5].dropna()
    if len(weekday_series) >= 5:
        last5 = weekday_series.iloc[-5:]
        if last5.nunique() == 1:
            fail(f"{asset_col} is identical across last 5 weekdays ({last5.iloc[0]:,.2f}) — data likely frozen")
        else:
            ok(f"{asset_col} varies across last 5 weekdays (not frozen)")


# CoinMetrics + Fear & Greed: daily-frequency sources should vary across the last 5 rows.
# Unlike FRED (monthly), these publish daily — identical values for 5 days means the
# source is down and ffill is propagating a stale value into model features.
for daily_col, label in (
    ('OnChain_Hash_Rate',          'CoinMetrics (Hash Rate)'),
    ('Sentiment_BTC_index_value',  'Fear & Greed'),
):
    if daily_col not in df.columns:
        warn(f"{daily_col} not in master_df — skipping freshness check")
        continue
    last5 = df[daily_col].dropna().iloc[-5:]
    if len(last5) >= 5 and last5.nunique() == 1:
        fail(f"{label} identical across last 5 rows ({last5.iloc[0]}) — source may be down, ffill propagating stale data")
    else:
        ok(f"{label} has varied in last 5 rows (source fresh)")


# FRED macro: find the last date any FRED column actually changed value.
# Row-uniformity checks are unreliable here because master_df is daily but FRED series
# are monthly/weekly — forward-fill legitimately produces long runs of identical rows.
# Instead, we look at when values last changed across the full history and fail only if
# it's been >60 days. M2 is weekly, so 60 days of total silence means the API is broken.
fred_cols = [c for c in ['Macro_CPI', 'Macro_Interest_Rate', 'Macro_Unemployment_Rate',
                          'Macro_PCE', 'Macro_GDP', 'Macro_10Y_Treasury_Yield',
                          'Macro_M2_Money_Supply'] if c in df.columns]
if fred_cols:
    changed_mask = df[fred_cols].ne(df[fred_cols].shift()).any(axis=1)
    changed_dates = df.index[changed_mask]
    if len(changed_dates) == 0:
        fail("No FRED macro changes found in entire history — data may be corrupt")
    else:
        last_fred_change = changed_dates.max()
        days_since = (date.today() - last_fred_change.date()).days
        if days_since > 60:
            fail(f"FRED macro last changed {days_since} days ago ({last_fred_change.date()}) — API key may be missing or FRED unreachable")
        else:
            ok(f"FRED macro last changed {days_since} days ago ({last_fred_change.date()})")


# ─────────────────────────────────────────────
# Check 3 — Value ranges
# ─────────────────────────────────────────────

print("\nCheck 3 — Value ranges")
for col, (lo, hi) in RANGE_CHECKS.items():
    if col not in df.columns:
        warn(f"{col} not in master_df — skipping range check")
        continue
    series = df[col].dropna()
    if series.empty:
        fail(f"{col} is all NaN")
        continue
    if lo is not None and (series < lo).any():
        bad = series[series < lo]
        fail(f"{col} has {len(bad)} value(s) below {lo} — min={series.min():.2f}")
    elif hi is not None and (series > hi).any():
        bad = series[series > hi]
        fail(f"{col} has {len(bad)} value(s) above {hi} — max={series.max():.2f}")
    else:
        bounds = f"{lo}–{hi}" if hi is not None else f">= {lo}"
        ok(f"{col} in range ({bounds})")


# ─────────────────────────────────────────────
# Check 4 — SELECTED_FEATURES present and non-NaN after ffill
# ─────────────────────────────────────────────

print("\nCheck 4 — SELECTED_FEATURES")
missing_cols = [f for f in SELECTED_FEATURES if f not in df.columns]
if missing_cols:
    fail(f"{len(missing_cols)} feature(s) missing from master_df: {missing_cols}")
else:
    ok(f"All {len(SELECTED_FEATURES)} features present in master_df")

last_row_ffill = df[SELECTED_FEATURES].ffill().iloc[-1] if not missing_cols else None
if last_row_ffill is not None:
    nan_features = last_row_ffill[last_row_ffill.isna()].index.tolist()
    if nan_features:
        fail(f"{len(nan_features)} feature(s) still NaN in last row after ffill: {nan_features}")
    else:
        ok(f"All {len(SELECTED_FEATURES)} features non-NaN in last row after ffill")


# ─────────────────────────────────────────────
# Check 5 — Date continuity
# ─────────────────────────────────────────────

print("\nCheck 5 — Date continuity")
dupes = df.index.duplicated().sum()
if dupes:
    fail(f"{dupes} duplicate date(s) in index")
else:
    ok("No duplicate dates")

diffs = df.index.to_series().diff().dropna()
max_gap = diffs.max()
if max_gap.days > GAP_THRESHOLD_DAYS:
    gap_loc = diffs.idxmax()
    fail(f"Largest gap is {max_gap.days} days (ending {gap_loc.date()}) — threshold is {GAP_THRESHOLD_DAYS} days")
else:
    ok(f"No date gap > {GAP_THRESHOLD_DAYS} days (largest gap: {max_gap.days} days)")


# ─────────────────────────────────────────────
# Check 6 — API contract logging (info only)
# ─────────────────────────────────────────────

print("\nCheck 6 — API contract logging (info only)")
source_column_groups = {
    'yfinance (BTC)':     [c for c in df.columns if c.endswith('_BTC') and not c.startswith(('MACD', 'RSI', 'SMA', 'EMA', 'OBV', 'Volume_P', 'Signal', '%', 'VWAP', 'OnChain', 'Target'))],
    'FRED macro':         [c for c in df.columns if c.startswith('Macro_')],
    'Fear & Greed':       [c for c in df.columns if c.startswith('Sentiment_BTC')],
    'Google Trends':      [c for c in df.columns if c.startswith('Sentiment_GT')],
    'CoinMetrics':        [c for c in df.columns if c.startswith('OnChain_')],
    'ETF flows':          [c for c in df.columns if c.startswith('ETF_')],
}
for source, cols in source_column_groups.items():
    warn(f"{source}: {cols}")


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
if failures:
    print(f"RESULT: {len(failures)} check(s) FAILED, {len(warnings)} info note(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(f"RESULT: All checks passed ({len(warnings)} info note(s))")
    sys.exit(0)
