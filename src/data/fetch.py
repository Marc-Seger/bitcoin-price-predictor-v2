"""
fetch.py — Incremental data fetching from all sources.

Each function accepts a last_date (string 'YYYY-MM-DD') and returns only
new rows from last_date+1 to today. Returns a pandas DataFrame indexed by date.

Sources:
- yfinance       : BTC, SP500, NASDAQ, Gold, DXY prices
- FRED API       : Macro indicators (CPI, rates, M2, etc.)
- Alternative.me : BTC Fear & Greed index
- pytrends       : Google Trends for "bitcoin"
- CoinMetrics    : On-chain metrics (free tier)
"""

import os
import sys
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from fredapi import Fred
from pytrends.request import TrendReq
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TICKERS, FRED_SERIES, COINMETRICS_METRICS

load_dotenv(os.path.join(os.path.dirname(__file__), '../../..', '.env'))
FRED_API_KEY = os.getenv('FRED_API_KEY')


def _date_range(last_date: str):
    """
    Returns (start, end) as date objects for the fetch window.
    start = last_date + 1 day, end = today.
    Returns None, None if already up to date.
    """
    start = (pd.to_datetime(last_date) + timedelta(days=1)).date()
    end = date.today()
    if start > end:
        return None, None
    return start, end


# ─────────────────────────────────────────────
# 1. PRICES — yfinance
# ─────────────────────────────────────────────

def fetch_yfinance(last_date: str) -> pd.DataFrame:
    """
    Fetches OHLCV data for all assets from yfinance in a single batch call.
    Columns are renamed to match existing master_df convention: Open_BTC, Close_SP500, etc.
    """
    start, end = _date_range(last_date)
    if not start:
        print('yfinance: already up to date.')
        return pd.DataFrame()

    tickers_str = ' '.join(TICKERS.values())
    raw = yf.download(tickers_str, start=str(start), end=str(end),
                      progress=False, auto_adjust=True)

    if raw.empty:
        print('yfinance: no new data.')
        return pd.DataFrame()

    frames = []
    for asset, ticker in TICKERS.items():
        try:
            df = raw.xs(ticker, axis=1, level=1)[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        except KeyError:
            print(f'yfinance: no new data for {asset}.')
            continue
        df.columns = [f'{col}_{asset}' for col in df.columns]
        df.index = pd.to_datetime(df.index).normalize()
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, axis=1)
    result.index.name = 'date'
    print(f'yfinance: fetched {len(result)} new rows ({start} → {end}).')
    return result


# ─────────────────────────────────────────────
# 2. MACRO — FRED API
# ─────────────────────────────────────────────

# Note: FRED series are monthly (or quarterly for GDP).
# We fetch the latest values and forward-fill into daily rows
# when merging — using the last known value until the next release.
# This is standard practice (point-in-time data).

def fetch_fred(last_date: str) -> pd.DataFrame:
    """
    Fetches macro indicators from FRED.
    Returns a DataFrame indexed by date (monthly/quarterly release dates).
    Forward-filling into daily frequency happens in merge.py.
    """
    start, end = _date_range(last_date)
    if not start:
        print('FRED: already up to date.')
        return pd.DataFrame()

    fred = Fred(api_key=FRED_API_KEY)
    frames = {}

    for col_name, series_id in FRED_SERIES.items():
        try:
            data = fred.get_series(series_id, observation_start=str(start), observation_end=str(end))
            if not data.empty:
                frames[col_name] = data
        except Exception as e:
            print(f'FRED: failed to fetch {series_id} — {e}')

    if not frames:
        print('FRED: no new data.')
        return pd.DataFrame()

    result = pd.DataFrame(frames)
    result.index = pd.to_datetime(result.index).normalize()
    result.index.name = 'date'
    print(f'FRED: fetched {len(result)} new release dates ({start} → {end}).')
    return result


# ─────────────────────────────────────────────
# 3. BTC FEAR & GREED — Alternative.me
# ─────────────────────────────────────────────

def fetch_fear_greed(last_date: str) -> pd.DataFrame:
    """
    Fetches BTC Fear & Greed index from Alternative.me.
    Free, no API key required. Returns daily values.
    Capped at 365 days per request to stay within API limits.
    """
    start, end = _date_range(last_date)
    if not start:
        print('Fear & Greed: already up to date.')
        return pd.DataFrame()

    days_to_fetch = min((end - start).days + 1, 365)
    url = f'https://api.alternative.me/fng/?limit={days_to_fetch}&date_format=world'

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        raw = r.json().get('data', [])
    except Exception as e:
        print(f'Fear & Greed: fetch failed — {e}')
        return pd.DataFrame()

    records = []
    for entry in raw:
        records.append({
            'date': pd.to_datetime(entry['timestamp'], dayfirst=True),
            'Sentiment_BTC_index_value': int(entry['value']),
            'Sentiment_BTC_index_label': entry['value_classification'],
        })

    if not records:
        print('Fear & Greed: no new data.')
        return pd.DataFrame()

    result = pd.DataFrame(records).set_index('date')
    result.index = result.index.normalize()
    result = result[result.index >= pd.Timestamp(start)]
    print(f'Fear & Greed: fetched {len(result)} new rows.')
    return result


# ─────────────────────────────────────────────
# 4. GOOGLE TRENDS — pytrends
# ─────────────────────────────────────────────

# Important limitation: Google Trends normalises values within the
# requested time window (0–100 scale). Incremental fetches would produce
# values on a different scale than historical data.
# Solution: always re-fetch the last 90 days so the normalisation window
# overlaps with existing data, keeping values comparable.

def fetch_google_trends(last_date: str) -> pd.DataFrame:
    """
    Fetches Google Trends for 'bitcoin' (US).
    Re-fetches last 90 days to maintain normalisation consistency,
    then returns only rows newer than last_date.
    """
    start, end = _date_range(last_date)
    if not start:
        print('Google Trends: already up to date.')
        return pd.DataFrame()

    # Always go back 90 days for normalisation overlap
    window_start = end - timedelta(days=90)
    timeframe = f'{window_start} {end}'

    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(['bitcoin'], cat=0, timeframe=timeframe, geo='US')
        df = pytrends.interest_over_time()
        time.sleep(1)  # avoid rate limiting
    except Exception as e:
        print(f'Google Trends: fetch failed — {e}')
        return pd.DataFrame()

    if df.empty:
        print('Google Trends: no data returned.')
        return pd.DataFrame()

    df = df[['bitcoin']].rename(columns={'bitcoin': 'Sentiment_GT_Bitcoin'})
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = 'date'

    result = df[df.index >= pd.Timestamp(start)]
    print(f'Google Trends: fetched {len(result)} new rows.')
    return result


# ─────────────────────────────────────────────
# 5. ON-CHAIN — CoinMetrics free API
# ─────────────────────────────────────────────

def fetch_onchain(last_date: str) -> pd.DataFrame:
    """
    Fetches on-chain metrics from CoinMetrics free community API.
    5 metrics available on free tier (down from 8 — 3 are paid).
    Dropped: SplyAct1yr, VtyDayRet30d, TxTfrValAdjUSD (paid tier only).
    """
    start, end = _date_range(last_date)
    if not start:
        print('CoinMetrics: already up to date.')
        return pd.DataFrame()

    url = 'https://community-api.coinmetrics.io/v4/timeseries/asset-metrics'
    params = {
        'assets': 'btc',
        'metrics': ','.join(COINMETRICS_METRICS.keys()),
        'start_time': str(start),
        'end_time': str(end),
        'page_size': 1000,
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json().get('data', [])
    except Exception as e:
        print(f'CoinMetrics: fetch failed — {e}')
        return pd.DataFrame()

    if not data:
        print('CoinMetrics: no new data.')
        return pd.DataFrame()

    result = pd.DataFrame(data).drop(columns=['asset'])
    result['date'] = pd.to_datetime(result['time']).dt.normalize()
    result = result.drop(columns=['time']).set_index('date')
    result = result.rename(columns=COINMETRICS_METRICS)
    result = result.apply(pd.to_numeric, errors='coerce')
    print(f'CoinMetrics: fetched {len(result)} new rows ({start} → {end}).')
    return result


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def fetch_all(last_date: str) -> dict:
    """
    Fetches new data from all sources since last_date.
    Returns a dict of DataFrames keyed by source name.
    Pass last_date as 'YYYY-MM-DD' (last date present in master_df).
    """
    print(f'\nFetching new data since {last_date}...\n')
    return {
        'prices':       fetch_yfinance(last_date),
        'macro':        fetch_fred(last_date),
        'fear_greed':   fetch_fear_greed(last_date),
        'trends':       fetch_google_trends(last_date),
        'onchain':      fetch_onchain(last_date),
    }


if __name__ == '__main__':
    # Quick test — fetch last 5 days
    test_date = (date.today() - timedelta(days=5)).strftime('%Y-%m-%d')
    results = fetch_all(test_date)
    for source, df in results.items():
        if not df.empty:
            print(f'\n{source}: {df.shape}\n{df.head(2)}\n')
