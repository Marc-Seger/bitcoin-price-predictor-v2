"""
features.py — Compute all technical indicators and features on master_df.

Always runs on the full DataFrame (not just new rows). This is intentional:
technical indicators like SMA(200), OBV (cumulative), and weekly MACD
are path-dependent and require full history to compute correctly.

Indicators computed per asset (BTC, SP500, NASDAQ, GOLD, DXY):
    - SMA: 9, 20, 50, 200
    - EMA: 9, 20, 50, 200, 12, 26
    - MACD (daily + weekly): line, signal, histogram
    - Bollinger Bands: upper, lower (20-day, 2 std)
    - RSI (14-day)
    - Stochastic Oscillator: %K, %D (14-day)
    - VWAP (30-day rolling)
    - OBV (cumulative, single yfinance source)

Signal flags per asset:
    - MACD above signal (daily + weekly)
    - MACD histogram positive/negative (daily + weekly)
    - Stochastic overbought/oversold/cross
    - Price above VWAP, Price above SMA200
    - RSI overbought/oversold
    - Golden cross / Death cross (and event flags)
    - Bollinger band breaks

Temporal features (date-based, no asset suffix):
    - Day of week, Month, Quarter, Is_Weekend, Season

Target columns (BTC only):
    - Target_Close_t+1 (next day close)
    - Target_Return_1d (next day return)
    - Target_Direction_1d (1 if return > 0)
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import ASSETS


# ─────────────────────────────────────────────
# PRICE INDICATORS
# ─────────────────────────────────────────────

def add_sma(df: pd.DataFrame, close_col: str) -> pd.DataFrame:
    for window in [9, 20, 50, 200]:
        df[f'SMA_{window}_{close_col}'] = df[close_col].rolling(window).mean()
    return df


def add_ema(df: pd.DataFrame, close_col: str) -> pd.DataFrame:
    for span in [9, 20, 50, 200, 12, 26]:
        df[f'EMA_{span}_{close_col}'] = df[close_col].ewm(span=span, adjust=False).mean()
    return df


def add_macd(df: pd.DataFrame, close_col: str, asset: str) -> pd.DataFrame:
    # Daily MACD — uses EMA-12 and EMA-26 already computed by add_ema
    ema12 = df[f'EMA_12_{close_col}']
    ema26 = df[f'EMA_26_{close_col}']
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df[f'MACD_D_{asset}'] = macd
    df[f'Signal_Line_D_{asset}'] = signal
    df[f'MACD_Histogram_D_{asset}'] = macd - signal

    # Weekly MACD (resample to weekly, then reindex back to daily)
    weekly = df[close_col].resample('W').last()
    w_ema12 = weekly.ewm(span=12, adjust=False).mean().reindex(df.index).ffill()
    w_ema26 = weekly.ewm(span=26, adjust=False).mean().reindex(df.index).ffill()
    w_macd = w_ema12 - w_ema26
    w_signal = w_macd.ewm(span=9, adjust=False).mean()
    df[f'MACD_W_{asset}'] = w_macd
    df[f'Signal_Line_W_{asset}'] = w_signal
    df[f'MACD_Histogram_W_{asset}'] = w_macd - w_signal
    return df


def add_bollinger(df: pd.DataFrame, close_col: str, asset: str) -> pd.DataFrame:
    sma20 = df[f'SMA_20_{close_col}']  # reuse from add_sma
    std20 = df[close_col].rolling(20).std()
    df[f'Upper_Band_{close_col}'] = sma20 + 2 * std20
    df[f'Lower_Band_{close_col}'] = sma20 - 2 * std20
    return df


def add_rsi(df: pd.DataFrame, close_col: str) -> pd.DataFrame:
    delta = df[close_col].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df[f'RSI_{close_col}'] = 100 - (100 / (1 + rs))
    return df


def add_stochastic(df: pd.DataFrame, high_col: str, low_col: str,
                   close_col: str, asset: str) -> pd.DataFrame:
    lowest_low = df[low_col].rolling(14).min()
    highest_high = df[high_col].rolling(14).max()
    k = 100 * (df[close_col] - lowest_low) / (highest_high - lowest_low)
    df[f'%K_{asset}'] = k
    df[f'%D_{asset}'] = k.rolling(3).mean()
    return df


def add_vwap(df: pd.DataFrame, high_col: str, low_col: str,
             close_col: str, vol_col: str, asset: str) -> pd.DataFrame:
    tp = (df[high_col] + df[low_col] + df[close_col]) / 3
    vwap = (tp * df[vol_col]).rolling(30).sum() / df[vol_col].rolling(30).sum()
    df[f'VWAP_30d_{asset}'] = vwap
    return df


def add_obv(df: pd.DataFrame, close_col: str, vol_col: str,
            asset: str) -> pd.DataFrame:
    """
    Single-source OBV using yfinance volume.
    Replaces the previous dual-source split (yfinance pre-2024, Alpha Vantage post-2024).
    Alpha Vantage is paid — yfinance volume is sufficient and consistent.
    Scaled to millions for readability.
    """
    direction = np.sign(df[close_col].diff())
    obv = (direction * df[vol_col]).cumsum()
    df[f'OBV_{asset}'] = obv / 1e6
    return df


# ─────────────────────────────────────────────
# SIGNAL FLAGS
# ─────────────────────────────────────────────

def add_signal_flags(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    close_col  = f'Close_{asset}'
    vol_col    = f'Volume_{asset}'
    macd_d     = f'MACD_D_{asset}'
    sig_d      = f'Signal_Line_D_{asset}'
    hist_d     = f'MACD_Histogram_D_{asset}'
    macd_w     = f'MACD_W_{asset}'
    sig_w      = f'Signal_Line_W_{asset}'
    hist_w     = f'MACD_Histogram_W_{asset}'
    stoch_k    = f'%K_{asset}'
    stoch_d    = f'%D_{asset}'
    sma200     = f'SMA_200_{close_col}'
    vwap       = f'VWAP_30d_{asset}'
    rsi        = f'RSI_{close_col}'
    upper_band = f'Upper_Band_{close_col}'
    lower_band = f'Lower_Band_{close_col}'

    # Volume percentile (rank within full history)
    df[f'Volume_Percentile_{asset}'] = df[vol_col].rank(pct=True)
    df[f'High_Volume_{asset}'] = (df[f'Volume_Percentile_{asset}'] > 0.9).astype(int)

    # MACD signals
    df[f'MACD_Above_Signal_D_{asset}']    = (df[macd_d] > df[sig_d]).astype(int)
    df[f'MACD_Above_Signal_W_{asset}']    = (df[macd_w] > df[sig_w]).astype(int)
    df[f'MACD_Hist_Positive_D_{asset}']   = (df[hist_d] > 0).astype(int)
    df[f'MACD_Hist_Negative_D_{asset}']   = (df[hist_d] < 0).astype(int)
    df[f'MACD_Hist_Positive_W_{asset}']   = (df[hist_w] > 0).astype(int)
    df[f'MACD_Hist_Negative_W_{asset}']   = (df[hist_w] < 0).astype(int)

    # Stochastic signals
    df[f'Stoch_Overbought_{asset}']  = (df[stoch_k] > 80).astype(int)
    df[f'Stoch_Oversold_{asset}']    = (df[stoch_k] < 20).astype(int)
    df[f'Stoch_Cross_Up_{asset}']    = (
        (df[stoch_k] > df[stoch_d]) & (df[stoch_k].shift(1) <= df[stoch_d].shift(1))
    ).astype(int)
    df[f'Stoch_Cross_Down_{asset}']  = (
        (df[stoch_k] < df[stoch_d]) & (df[stoch_k].shift(1) >= df[stoch_d].shift(1))
    ).astype(int)

    # Price position signals
    df[f'Price_Above_VWAP_{asset}']   = (df[close_col] > df[vwap]).astype(int)
    df[f'Price_Above_SMA200_{asset}'] = (df[close_col] > df[sma200]).astype(int)

    # RSI signals
    df[f'RSI_Overbought_{asset}'] = (df[rsi] > 70).astype(int)
    df[f'RSI_Oversold_{asset}']   = (df[rsi] < 30).astype(int)

    # Golden / Death cross (SMA50 vs SMA200)
    sma50  = f'SMA_50_{close_col}'
    golden = df[sma50] > df[sma200]
    df[f'Golden_Cross_{asset}']       = golden.astype(int)
    df[f'Death_Cross_{asset}']        = (~golden).astype(int)
    df[f'Golden_Cross_Event_{asset}'] = (
        golden & ~golden.shift(1).fillna(False)
    ).astype(int)
    df[f'Death_Cross_Event_{asset}']  = (
        ~golden & golden.shift(1).fillna(True)
    ).astype(int)

    # Bollinger band breaks
    df[f'Bollinger_Upper_Break_{asset}'] = (df[close_col] > df[upper_band]).astype(int)
    df[f'Bollinger_Lower_Break_{asset}'] = (df[close_col] < df[lower_band]).astype(int)

    return df


# ─────────────────────────────────────────────
# TEMPORAL FEATURES
# ─────────────────────────────────────────────

def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df['Day_of_Week'] = df.index.dayofweek          # 0=Monday
    df['Month']       = df.index.month
    df['Quarter']     = df.index.quarter
    df['Is_Weekend']  = (df.index.dayofweek >= 5).astype(int)
    # Vectorized: lookup array indexed by month (0=unused, 1-12=months)
    season_map = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])  # index 0 unused
    df['Season'] = season_map[df.index.month]
    return df


# ─────────────────────────────────────────────
# TARGET VARIABLES
# ─────────────────────────────────────────────

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prediction targets based on BTC close.
    1-day and 7-day return horizons.
    Last N rows will have NaN targets (future unknown).
    """
    close = df['Close_BTC']
    # 1-day targets
    df['Target_Close_t+1']    = close.shift(-1)
    df['Target_Return_1d']    = close.shift(-1) / close - 1
    df['Target_Direction_1d'] = (df['Target_Return_1d'] > 0).astype('Int64')
    # 7-day targets
    df['Target_Return_7d']    = close.shift(-7) / close - 1
    df['Target_Direction_7d'] = (df['Target_Return_7d'] > 0).astype('Int64')
    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full feature engineering pipeline on the DataFrame.
    Always operates on the full dataset (see module docstring for why).
    Returns the DataFrame with all indicators and targets added.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    for asset in ASSETS:
        close_col = f'Close_{asset}'
        high_col  = f'High_{asset}'
        low_col   = f'Low_{asset}'
        vol_col   = f'Volume_{asset}'

        if close_col not in df.columns:
            print(f'features: skipping {asset} — no close column found.')
            continue

        df = add_sma(df, close_col)
        df = add_ema(df, close_col)
        df = add_macd(df, close_col, asset)
        df = add_bollinger(df, close_col, asset)
        df = add_rsi(df, close_col)
        df = add_stochastic(df, high_col, low_col, close_col, asset)
        df = add_vwap(df, high_col, low_col, close_col, vol_col, asset)
        df = add_obv(df, close_col, vol_col, asset)
        df = add_signal_flags(df, asset)

    df = add_temporal(df)
    df = add_targets(df)

    print(f'features: computed {len(df.columns)} columns for {len(df)} rows.')
    return df


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from merge import load_master

    df = load_master()
    print(f'Loaded master_df: {df.shape}')

    result = compute_features(df)
    print(f'After features: {result.shape}')
    print(f'\nSample — last 3 rows, key columns:')
    cols = ['Close_BTC', 'RSI_Close_BTC', 'MACD_D_BTC', 'OBV_BTC', 'Target_Close_t+1']
    print(result[cols].tail(3))
