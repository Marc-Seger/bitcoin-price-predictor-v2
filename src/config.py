"""
config.py — Shared constants for the bitcoin-price-predictor project.

Single source of truth for file paths, asset lists, feature sets,
and target column names used across data pipeline and model evaluation.
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MASTER_DF_PATH = os.path.join(PROJECT_ROOT, 'data', 'full_data', 'master_df.csv')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# ─────────────────────────────────────────────
# ASSETS
# ─────────────────────────────────────────────

ASSETS = ['BTC', 'SP500', 'NASDAQ', 'GOLD', 'DXY']

TICKERS = {
    'BTC':    'BTC-USD',
    'SP500':  '^GSPC',
    'NASDAQ': '^IXIC',
    'GOLD':   'GC=F',
    'DXY':    'DX-Y.NYB',
}

# ─────────────────────────────────────────────
# FRED MACRO SERIES
# ─────────────────────────────────────────────

# Maps master_df column name → FRED series ID.
# Also used by merge.py to derive the forward-fill column list.
FRED_SERIES = {
    'Macro_CPI':                  'CPIAUCSL',
    'Macro_Interest_Rate':        'FEDFUNDS',
    'Macro_Unemployment_Rate':    'UNRATE',
    'Macro_PCE':                  'PCEPI',
    'Macro_GDP':                  'GDP',
    'Macro_10Y_Treasury_Yield':   'GS10',
    'Macro_M2_Money_Supply':      'M2SL',
}

# ─────────────────────────────────────────────
# ON-CHAIN METRICS
# ─────────────────────────────────────────────

# Maps CoinMetrics API name → master_df column name
COINMETRICS_METRICS = {
    'AdrActCnt':  'OnChain_Active_Addresses',
    'TxCnt':      'OnChain_Transaction_Count',
    'HashRate':   'OnChain_Hash_Rate',
    'CapMVRVCur': 'OnChain_MVRV_Ratio',
    'ROI30d':     'OnChain_30d_ROI',
}

# ─────────────────────────────────────────────
# COLUMN NAMES — sentiment & external signals
# ─────────────────────────────────────────────

# Used in fetch.py, merge.py, backfill_history.py, and dashboard views.
# Centralised here so a rename only requires one change.
COL_FEAR_GREED    = 'Sentiment_BTC_index_value'
COL_GOOGLE_TRENDS = 'Sentiment_GT_Bitcoin'
COL_ETF_FLOW      = 'ETF_Flow_Total'

# ─────────────────────────────────────────────
# TARGETS
# ─────────────────────────────────────────────

TARGET_1D = 'Target_Return_1d'
TARGET_7D = 'Target_Return_7d'

# ─────────────────────────────────────────────
# SELECTED FEATURES — 52 curated
# ─────────────────────────────────────────────
# Top 50 by RF importance + Close_BTC (analyst override).
# Dropped from consideration:
#   - Google Trends: retail signal diluted post-ETF era (institutional flows dominate)
#   - Macro_Unemployment_Rate: monthly data forward-filled, model can't isolate release-day impact

SELECTED_FEATURES = [
    # BTC momentum & oscillators
    'MACD_Histogram_D_BTC', 'MACD_Histogram_W_BTC', '%D_BTC', '%K_BTC',
    'RSI_Close_BTC', 'Signal_Line_D_BTC', 'MACD_D_BTC',

    # BTC trend & price
    'SMA_9_Close_BTC', 'EMA_9_Close_BTC', 'Close_BTC',

    # BTC volume
    'Volume_Percentile_BTC', 'Volume_BTC', 'OBV_BTC',

    # On-chain
    'OnChain_Hash_Rate', 'OnChain_Active_Addresses', 'OnChain_30d_ROI',
    'OnChain_Transaction_Count', 'OnChain_MVRV_Ratio',

    # Sentiment
    'Sentiment_BTC_index_value',

    # SP500
    'Signal_Line_D_SP500', 'SMA_50_Close_SP500', 'MACD_Histogram_W_SP500',
    'Volume_SP500', 'Volume_Percentile_SP500', 'RSI_Close_SP500',
    '%D_SP500', 'OBV_SP500',

    # NASDAQ
    'MACD_Histogram_W_NASDAQ', 'MACD_D_NASDAQ', '%D_NASDAQ',
    'Volume_NASDAQ', 'Volume_Percentile_NASDAQ', '%K_NASDAQ',
    'Signal_Line_D_NASDAQ', 'MACD_Histogram_D_NASDAQ', 'SMA_200_Close_NASDAQ',

    # DXY (Dollar)
    'SMA_50_Close_DXY', 'RSI_Close_DXY', 'MACD_Histogram_W_DXY',
    'MACD_Histogram_D_DXY', '%D_DXY', '%K_DXY', 'Signal_Line_D_DXY',

    # Gold
    'VWAP_30d_GOLD', 'MACD_Histogram_D_GOLD', '%K_GOLD',
    'MACD_Histogram_W_GOLD', 'RSI_Close_GOLD', 'Volume_GOLD',
    'Volume_Percentile_GOLD', '%D_GOLD',

    # Temporal
    'Day_of_Week',
]
