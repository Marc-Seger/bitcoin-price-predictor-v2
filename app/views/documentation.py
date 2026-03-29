"""
Page 5 — Documentation

How to use the app, data sources, methodology, known limitations.
"""

import streamlit as st


def render():
    st.markdown("<h1 style='margin-bottom:4px;'>Documentation</h1>", unsafe_allow_html=True)

    tab_howto, tab_data, tab_model, tab_limits = st.tabs([
        "How to Use", "Data Sources", "Model Methodology", "Limitations & Flaws"
    ])

    # ══════════════════════════════════════════
    # HOW TO USE
    # ══════════════════════════════════════════
    with tab_howto:
        st.subheader("Getting Started")
        st.markdown("""
        ### Pages

        **1. Financial Dashboard**
        Browse BTC price charts with technical indicators (SMA, Bollinger Bands, RSI, MACD),
        sentiment data (Fear & Greed Index, Google Trends, ETF flows), on-chain metrics,
        and cross-asset correlations vs S&P 500, NASDAQ, Gold, and the Dollar Index.
        Use the timeframe selector and indicator toggles to customize your view.

        **2. 7-Day Forecast**
        The model's current prediction: will BTC go up or down over the next 7 days?
        Predictions are generated automatically each day via GitHub Actions and logged
        to a paper trading record. The confidence level indicates signal strength —
        high-confidence predictions (>5% predicted move) have historically been more accurate.

        **3. Strategy Lab**
        Backtest trading strategies against the model's historical walk-forward predictions.
        Adjust:
        - **Strategy**: direction-only, high-confidence only, RSI-filtered, or trend-following
        - **Leverage**: 1x (no leverage) to 40x
        - **Stop Loss / Take Profit**: auto-close positions at thresholds

        The equity curve shows simulated portfolio value vs buy-and-hold. All predictions
        in the backtest are genuine out-of-sample — the model never saw future data when
        they were generated.

        **4. Model Performance**
        Walk-forward validation results: R-squared, direction accuracy, performance across
        8 distinct market phases (2018 bear market through 2024–2026 cycle), and the
        confidence-accuracy relationship.

        **5. Documentation**
        This page.

        ### Data Updates
        The full pipeline runs automatically every day at 07:00 UTC via GitHub Actions:
        fetch → merge → features → train → predict → upload. No manual steps needed.
        """)

    # ══════════════════════════════════════════
    # DATA SOURCES
    # ══════════════════════════════════════════
    with tab_data:
        st.subheader("Data Sources")
        st.markdown("""
        All data sources are **free** and **fully automated**. The pipeline covers
        **Jan 2017 → present** (~3,375 daily rows, ~2,974 model-usable after feature warmup).

        | Source | Data | Frequency | Notes |
        |--------|------|-----------|-------|
        | yfinance | BTC, S&P 500, NASDAQ, Gold, Dollar Index (OHLCV) | Daily | Real index tickers (^GSPC, ^IXIC, GC=F, DX-Y.NYB) |
        | FRED | CPI, Fed Funds Rate, PCE, GDP, 10Y Treasury, M2, Unemployment | Monthly | Forward-filled to daily |
        | Alternative.me | Bitcoin Fear & Greed Index (0–100) | Daily | Full history to 2018 |
        | pytrends | Google Trends for "bitcoin" (US, 0–100) | Weekly | Three-window rescaling for consistent scale across 9 years |
        | CoinMetrics | Active Addresses, Transaction Count, Hash Rate, MVRV Ratio, 30d ROI | Daily | Free community tier |
        | farside.co.uk | Bitcoin spot ETF net daily flows ($M) | Daily | Available from Jan 2024 (ETF launch) |

        ### Data Pipeline

        `scripts/update_data.py` runs the full pipeline:
        1. **Fetch**: downloads new data from all 6 sources since the last update
        2. **Merge**: appends new rows to master_df, forward-fills monthly FRED and weekly Trends data
        3. **Features**: computes 276 technical indicators across all 5 assets
        4. **Save**: writes the updated master_df.csv (stored as a GitHub Release asset — too large for the repo)

        ### Feature Engineering

        For each asset (BTC, SP500, NASDAQ, Gold, DXY), the pipeline computes:
        - Moving averages: SMA (9, 20, 50, 200), EMA (9, 20, 50, 200, 12, 26)
        - MACD: daily and weekly (line, signal, histogram)
        - Bollinger Bands (20-day, 2 standard deviations)
        - RSI (14-day)
        - Stochastic Oscillator (%K, %D)
        - VWAP (30-day rolling)
        - OBV (on-balance volume, cumulative)
        - Signal flags: golden/death cross, overbought/oversold, band breaks

        Plus temporal features (day of week, month, season) and on-chain/sentiment data.

        **52 features** are selected for the model from the full 276, based on
        Random Forest importance ranking and analyst review.
        """)

        st.subheader("Dropped Data Sources")
        st.markdown("""
        | Source | Reason |
        |--------|--------|
        | Alpha Vantage | Paid API — replaced by yfinance |
        | CNN Fear & Greed (stock market) | No public API — requires DevTools scraping, not automatable |
        | CoinMetrics paid metrics (SplyAct1yr, VtyDayRet30d, TxTfrValAdjUSD) | Paid tier only |
        | Google Trends (as model feature) | Retail signal diluted post-ETF era — kept in dashboard for context |
        | ETF flows (as model feature) | Only available since Jan 2024 — not enough history for model training |
        | Macro Unemployment Rate (as model feature) | Monthly forward-fill prevents isolating release-day impact |
        """)

    # ══════════════════════════════════════════
    # MODEL METHODOLOGY
    # ══════════════════════════════════════════
    with tab_model:
        st.subheader("Model Methodology")
        st.markdown("""
        ### Target Variable
        **7-day forward return** — percentage change in BTC price over the next 7 calendar days.

        We predict returns instead of absolute prices because:
        - Tree-based models (XGBoost, Random Forest) cannot extrapolate beyond their training range
        - If BTC reaches a new all-time high, the model can still predict "+5% return" from any level
        - Absolute price prediction is trivially broken the moment price exceeds training history

        ### Why 7 Days?
        - Daily returns are too noisy — R² < 0 for all models (worse than predicting the mean)
        - 7-day returns smooth out daily noise, letting underlying trends show through
        - Weekly windows align with typical short-term trading decision horizons

        ### Evaluation: Walk-Forward Validation
        The gold standard for time-series model evaluation:

        1. Train on all data up to day T
        2. Predict the 7-day return starting at T
        3. Record the prediction (model has never seen the outcome)
        4. Slide forward 7 days (non-overlapping), repeat

        Every prediction is genuinely out-of-sample. This is much more honest than a simple
        train/test split, where performance depends heavily on which period you happen to test on.

        Non-overlapping windows are used to avoid inflated metrics from overlapping 7-day returns
        sharing 6/7 of the same days.

        ### Models Tested

        | Model | R² | Direction Accuracy | Notes |
        |-------|----|--------------------|-------|
        | **XGBoost** | **-0.056** | **55.2%** | Selected — best out-of-sample performance |
        | Random Forest | -0.12 | 53.8% | Similar to XGBoost but slightly worse |
        | LSTM | < -1.0 | ~50% | Failed — insufficient training data for neural nets |
        | GRU | < -2.0 | ~54% | Failed — same reason |

        **Naive baseline** (always predict UP): **52.3%** direction accuracy.
        XGBoost achieves a **+2.9 percentage point edge** over the naive baseline — modest but consistent.

        ### Confidence Levels
        The predicted return magnitude is used as a confidence proxy:
        - **HIGH** (>5% predicted move): 65.7% direction accuracy — meaningful edge
        - **MEDIUM** (2–5% predicted move): 53.1% accuracy — marginal edge
        - **LOW** (<2% predicted move): 51.0% accuracy — near coin-flip

        This confidence-accuracy relationship is one of the model's most useful properties:
        you can choose to act only on high-confidence signals.

        ### Drift Detection
        The paper trading log monitors rolling direction accuracy. If accuracy drops below
        60% over the last 20 resolved predictions, the app displays a retraining warning.
        """)

    # ══════════════════════════════════════════
    # LIMITATIONS
    # ══════════════════════════════════════════
    with tab_limits:
        st.subheader("Known Limitations & Flaws")
        st.markdown("""
        ### Model Limitations

        - **The edge is small.** 55.2% direction accuracy vs 52.3% naive baseline is a real but
          modest signal. It is not a trading system — it is a demonstration that structured data
          contains some predictive information about short-term BTC direction.
        - **R² is negative.** The model explains less variance than simply predicting the mean return.
          It is directionally useful but poor at predicting return magnitude.
        - **No feature for breaking news.** Regulatory announcements, exchange collapses, ETF approval
          surprises, and macro shocks are invisible to the model — it only sees structured numerical data.
        - **LSTM/GRU failed** with ~3,000 training rows. Deep learning models typically need 10,000+
          samples to generalise. More years of data would help, but daily BTC data only exists since 2010.
        - **Hyperparameter tuning pending** — results will be updated after Optuna tuning completes.

        ### Data Limitations

        - **FRED macro data is monthly**, forward-filled to daily. The model cannot isolate the
          impact of a specific CPI release — it sees the same value for 30 days.
        - **Google Trends is weekly**, forward-filled to daily. The scale is normalised per request
          window (0–100), requiring multi-window rescaling for historical consistency.
        - **ETF flows start Jan 2024** — not enough history to use as a model feature, but shown
          in the dashboard as a retail vs institutional sentiment indicator.
        - **On-chain data** is limited to 5 free CoinMetrics metrics. Paid metrics (supply activity,
          adjusted transfer volumes) might improve predictions.

        ### Strategy Lab Limitations

        - **No transaction fees** (exchanges typically charge 0.1–0.5% per trade)
        - **No slippage** — real orders may fill at worse prices, especially with leverage
        - **7-day windows are fixed** — a real strategy might use variable holding periods
        - **Liquidation is simplified** — real exchanges have funding rates, margin calls,
          and partial liquidation mechanisms not modeled here
        - **Past performance does not predict future results.** This is a portfolio project,
          not financial advice.

        ### What Could Improve the Model

        - Bayesian hyperparameter tuning (Optuna, 100 trials) — in progress
        - Real-time social sentiment (Twitter/X, Reddit, news headlines)
        - Options market data (implied volatility, put/call ratio)
        - Ensemble of XGBoost + Random Forest for more robust predictions
        - Once ETF flow history accumulates (2–3 years), it could become a useful model feature
        """)
