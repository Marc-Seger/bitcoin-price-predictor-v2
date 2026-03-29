"""
backfill_history.py — One-time script to rebuild master_df from Jan 2017.

Fetches full history for all sources, merges, forward-fills, computes features.
Run once; daily update_data.py takes over from there.
"""

import os
import sys
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import date
from fredapi import Fred
from dotenv import load_dotenv
from pytrends.request import TrendReq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'data'))

from config import TICKERS, FRED_SERIES, COINMETRICS_METRICS, MASTER_DF_PATH, COL_GOOGLE_TRENDS
from merge import forward_fill_sparse
from features import compute_features
from fetch import fetch_etf_flows

load_dotenv(os.path.join(os.path.dirname(__file__), '../..', '.env'))

START = '2017-01-01'
END   = str(date.today())


# ── 1. Prices ────────────────────────────────────────────────────────────────

def fetch_prices():
    print('Fetching prices (yfinance)...')
    tickers_str = ' '.join(TICKERS.values())
    raw = yf.download(tickers_str, start=START, end=END, progress=False, auto_adjust=True)

    frames = []
    for asset, ticker in TICKERS.items():
        df = raw.xs(ticker, axis=1, level=1)[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.columns = [f'{col}_{asset}' for col in df.columns]
        df.index = pd.to_datetime(df.index).normalize()
        frames.append(df)

    result = pd.concat(frames, axis=1)
    result.index.name = 'date'
    print(f'  Prices: {len(result)} rows ({result.index[0].date()} → {result.index[-1].date()})')
    return result


# ── 2. FRED macro ─────────────────────────────────────────────────────────────

def fetch_macro():
    print('Fetching macro (FRED)...')
    fred = Fred(api_key=os.getenv('FRED_API_KEY'))
    frames = {}
    for col_name, series_id in FRED_SERIES.items():
        try:
            data = fred.get_series(series_id, observation_start=START, observation_end=END)
            if not data.empty:
                frames[col_name] = data
        except Exception as e:
            print(f'  FRED {series_id} failed: {e}')

    result = pd.DataFrame(frames)
    result.index = pd.to_datetime(result.index).normalize()
    result.index.name = 'date'
    print(f'  Macro: {len(result)} release dates, {len(result.columns)} series')
    return result


# ── 3. Fear & Greed (full history) ───────────────────────────────────────────

def fetch_fear_greed():
    print('Fetching Fear & Greed (Alternative.me full history)...')
    url = 'https://api.alternative.me/fng/?limit=3000&date_format=world'
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        raw = r.json().get('data', [])
    except Exception as e:
        print(f'  Fear & Greed failed: {e}')
        return pd.DataFrame()

    records = [{'date': pd.to_datetime(e['timestamp'], dayfirst=True),
                'Sentiment_BTC_index_value': int(e['value']),
                'Sentiment_BTC_index_label': e['value_classification']}
               for e in raw]

    result = pd.DataFrame(records).set_index('date')
    result.index = pd.to_datetime(result.index).tz_localize(None).normalize()
    result.index.name = 'date'
    print(f'  Fear & Greed: {len(result)} rows ({result.index.min().date()} → {result.index.max().date()})')
    return result


# ── 4. CoinMetrics on-chain ───────────────────────────────────────────────────

def fetch_onchain():
    print('Fetching on-chain (CoinMetrics)...')
    url = 'https://community-api.coinmetrics.io/v4/timeseries/asset-metrics'
    all_rows = []
    next_page = None

    params = {
        'assets': 'btc',
        'metrics': ','.join(COINMETRICS_METRICS.keys()),
        'start_time': START,
        'end_time': END,
        'page_size': 1000,
    }

    while True:
        if next_page:
            params['next_page_token'] = next_page
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            body = r.json()
        except Exception as e:
            print(f'  CoinMetrics failed: {e}')
            break

        all_rows.extend(body.get('data', []))
        next_page = body.get('next_page_token')
        if not next_page:
            break
        time.sleep(0.5)

    if not all_rows:
        print('  CoinMetrics: no data.')
        return pd.DataFrame()

    result = pd.DataFrame(all_rows).drop(columns=['asset'])
    result['date'] = pd.to_datetime(result['time']).dt.tz_convert(None).dt.normalize()
    result = result.drop(columns=['time']).set_index('date')
    result.index.name = 'date'
    result = result.rename(columns=COINMETRICS_METRICS)
    result = result.apply(pd.to_numeric, errors='coerce')
    print(f'  On-chain: {len(result)} rows ({result.index.min().date()} → {result.index.max().date()})')
    return result


# ── 5. Google Trends — pytrends (three-window rescaling) ─────────────────────

def _fetch_trends_window(pytrends, start_str, end_str, label):
    """Fetch one pytrends window. Returns weekly DataFrame or None."""
    try:
        pytrends.build_payload(['bitcoin'], cat=0, timeframe=f'{start_str} {end_str}', geo='US')
        df = pytrends.interest_over_time()[['bitcoin']].rename(columns={'bitcoin': COL_GOOGLE_TRENDS})
        df.index = pd.to_datetime(df.index).normalize()
        df.index.name = 'date'
        time.sleep(2)
        print(f'  Window {label} ({start_str} → {end_str}): {len(df)} rows')
        return df
    except Exception as e:
        print(f'  Window {label} failed: {e}')
        return None


def _rescale_to(base, target):
    """Rescale target to match base's scale using their overlapping dates."""
    overlap = base.index.intersection(target.index)
    if len(overlap) < 4:
        print(f'  Overlap too small ({len(overlap)} rows) — no rescaling applied')
        return target
    m_base = base.loc[overlap, COL_GOOGLE_TRENDS].mean()
    m_target = target.loc[overlap, COL_GOOGLE_TRENDS].mean()
    scale = m_base / m_target if m_target > 0 else 1.0
    print(f'  Scale: {scale:.3f}  (base mean={m_base:.1f}, target mean={m_target:.1f}, overlap={len(overlap)} rows)')
    result = target.copy()
    result[COL_GOOGLE_TRENDS] = (result[COL_GOOGLE_TRENDS] * scale).clip(0, 100)
    return result


def fetch_google_trends():
    """
    Fetches Google Trends for 'bitcoin' (US) from 2017 to today.
    pytrends returns weekly data for windows ≤5 years, monthly for longer.
    Monthly is too coarse and compresses early peaks.
    Fix: three overlapping windows all ≤5 years (all weekly), rescaled at overlaps.
      W1: 2017-01-01 → 2020-12-31  (4 years)
      W2: 2020-07-01 → 2023-12-31  (3.5 years, 6-month overlap with W1)
      W3: 2023-07-01 → today       (~2.75 years, 6-month overlap with W2)
    """
    print('Fetching Google Trends (pytrends, three-window rescaling)...')
    pytrends = TrendReq(hl='en-US', tz=360)

    w1 = _fetch_trends_window(pytrends, '2017-01-01', '2020-12-31', '1')
    w2 = _fetch_trends_window(pytrends, '2020-07-01', '2023-12-31', '2')
    w3 = _fetch_trends_window(pytrends, '2023-07-01', str(date.today()), '3')

    if not all([w1 is not None, w2 is not None, w3 is not None]):
        parts = [w for w in [w1, w2, w3] if w is not None]
        print(f'  One or more windows failed — combining {len(parts)} partial windows')
        if not parts:
            return pd.DataFrame()
        combined = pd.concat(parts)
        return combined[~combined.index.duplicated(keep='last')].sort_index()

    w2_scaled = _rescale_to(w1, w2)
    w3_scaled = _rescale_to(w2_scaled, w3)

    combined = pd.concat([
        w1[w1.index < pd.Timestamp('2020-07-01')],
        w2_scaled[(w2_scaled.index >= pd.Timestamp('2020-07-01')) & (w2_scaled.index < pd.Timestamp('2023-07-01'))],
        w3_scaled[w3_scaled.index >= pd.Timestamp('2023-07-01')],
    ])
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    combined.index.name = 'date'
    print(f'  Google Trends: {len(combined)} weekly rows ({combined.index.min().date()} → {combined.index.max().date()})')
    return combined


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 55)
    print(f'Backfill: {START} → {END}')
    print('=' * 55)

    prices    = fetch_prices()
    macro     = fetch_macro()
    fear      = fetch_fear_greed()
    onchain   = fetch_onchain()
    trends    = fetch_google_trends()

    # Pass a pre-ETF date to fetch full history (ETFs launched Jan 2024)
    print('Fetching ETF flows (farside.co.uk)...')
    etf_flows = fetch_etf_flows('2024-01-01')
    if not etf_flows.empty:
        print(f'  ETF flows: {len(etf_flows)} rows ({etf_flows.index.min().date()} → {etf_flows.index.max().date()})')

    print('\nMerging...')
    combined = pd.concat([prices, macro, fear, onchain, trends, etf_flows], axis=1)
    combined = combined.sort_index()
    combined.index.name = 'date'
    print(f'  Raw combined: {combined.shape}')

    combined = forward_fill_sparse(combined)
    final    = compute_features(combined)

    final.to_csv(MASTER_DF_PATH)
    print(f'\nSaved: {final.shape} → {MASTER_DF_PATH}')

    # Summary
    import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from config import SELECTED_FEATURES
    available = [f for f in SELECTED_FEATURES if f in final.columns]
    after_dropna = final[available].dropna()
    print(f'Model-usable rows after dropna: {len(after_dropna)} '
          f'({after_dropna.index.min().date()} → {after_dropna.index.max().date()})')


if __name__ == '__main__':
    main()
