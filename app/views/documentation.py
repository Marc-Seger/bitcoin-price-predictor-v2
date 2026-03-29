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
        sentiment data (Fear & Greed), on-chain metrics, and cross-asset correlations.
        Use the timeframe selector and indicator toggles to customize your view.

        **2. 7-Day Forecast**
        See the model's current prediction: will BTC go up or down over the next 7 days?
        The confidence level tells you how reliable the prediction is likely to be.
        Predictions are logged automatically daily via GitHub Actions.

        **3. Strategy Lab**
        Backtest trading strategies using historical model predictions. Adjust:
        - **Strategy**: how the model signal is used (direction only, high confidence only, with RSI filter, etc.)
        - **Leverage**: 1x (no leverage) to 40x — see how leverage amplifies both gains and losses
        - **Stop Loss**: automatically close the position if price drops by X%
        - **Take Profit**: automatically close the position if price rises by X%

        The equity curve shows your simulated portfolio value vs buy-and-hold.
        Liquidation events are marked with red X markers.

        **4. Model Performance**
        Detailed evaluation metrics from walk-forward validation: R-squared, direction accuracy,
        performance across market phases, and the confidence-accuracy relationship.

        ### Keeping Data Fresh
        The model's predictions are only as good as the data. To update:
        ```
        python scripts/update_data.py
        python scripts/train_production.py
        ```
        Or, if GitHub Actions is configured, this happens automatically every day.
        """)

    # ══════════════════════════════════════════
    # DATA SOURCES
    # ══════════════════════════════════════════
    with tab_data:
        st.subheader("Data Sources")
        st.markdown("""
        All data sources are **free** and **fully automated**.

        | Source | Data | Frequency | API |
        |--------|------|-----------|-----|
        | yfinance | BTC, S&P 500, NASDAQ, Gold, Dollar Index (OHLCV) | Daily | Free, no key |
        | FRED | CPI, Interest Rate, Unemployment, PCE, GDP, 10Y Treasury, M2 Money Supply | Monthly | Free, API key required |
        | Alternative.me | Bitcoin Fear & Greed Index | Daily | Free, no key |
        | pytrends | Google Trends for "bitcoin" | Weekly | Free, no key |
        | CoinMetrics | Active Addresses, Transaction Count, Hash Rate, MVRV Ratio, 30d ROI | Daily | Free community tier |

        ### Data Pipeline

        `scripts/update_data.py` runs the full pipeline:
        1. **Fetch**: downloads new data from all 5 sources since the last update
        2. **Merge**: appends new rows to master_df, forward-fills monthly FRED data
        3. **Features**: computes 269 technical indicators across all 5 assets
        4. **Save**: writes the updated master_df.csv

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

        **52 features** are selected for the model from the full 269, based on
        Random Forest importance ranking and analyst review.
        """)

        st.subheader("Dropped Data Sources")
        st.markdown("""
        | Source | Reason |
        |--------|--------|
        | Alpha Vantage | Paid API — replaced by yfinance |
        | CNN Fear & Greed (stock market) | No public API |
        | ETF flows | Only 18 months of data, not automatable |
        | CoinMetrics paid metrics (SplyAct1yr, VtyDayRet30d, TxTfrValAdjUSD) | Paid tier only |
        | Google Trends (as model feature) | Retail signal diluted post-ETF era |
        | Macro Unemployment Rate (as model feature) | Monthly forward-fill prevents isolating release-day impact |
        """)

    # ══════════════════════════════════════════
    # MODEL METHODOLOGY
    # ══════════════════════════════════════════
    with tab_model:
        st.subheader("Model Methodology")
        st.markdown("""
        ### Target Variable
        **7-day forward return** (percentage change in BTC price over the next 7 calendar days).

        We predict returns instead of absolute prices because:
        - Tree-based models (XGBoost, Random Forest) cannot extrapolate beyond their training data range
        - If BTC reaches $100,000 but training data only goes to $70,000, the model can't predict $100,000
        - But it can predict "+5% return" regardless of the price level

        ### Why 7 Days?
        - Daily returns are too noisy (R² < 0 for all models — worse than guessing the average)
        - 7-day returns smooth out daily noise, letting underlying trends show through
        - Weekly prediction windows align with typical trading decision horizons

        ### Evaluation: Walk-Forward Validation
        The gold standard for time series model evaluation:

        1. Train on all data up to day T
        2. Predict day T+1 (or the next 7 days)
        3. Record the prediction
        4. Slide forward one day, repeat

        Every prediction is made **without seeing future data**. This is much more honest
        than a simple train/test split, which can be lucky or unlucky depending on where you split.

        ### Overlapping Target Correction
        7-day returns evaluated daily share 6/7 days between consecutive predictions,
        which inflates accuracy metrics. We report **non-overlapping** results
        (every 7th prediction) for honest metrics.

        ### Models Tested

        | Model | Type | R² (non-overlapping) | Direction Accuracy |
        |-------|------|---------------------|-------------------|
        | **XGBoost** | Gradient boosted trees | **0.50** | **75.7%** |
        | Random Forest | Bagged decision trees | 0.35 | 71.6% |
        | LSTM | Recurrent neural network | -1.14 | 50.0% |
        | GRU | Recurrent neural network | -2.01 | 54.1% |

        XGBoost was selected as the production model.

        ### Drift Detection
        The paper trading log monitors rolling direction accuracy. If accuracy drops
        below 60% over the last 20 resolved predictions, the app displays a warning
        recommending model retraining.
        """)

    # ══════════════════════════════════════════
    # LIMITATIONS
    # ══════════════════════════════════════════
    with tab_limits:
        st.subheader("Known Limitations & Flaws")
        st.markdown("""
        ### Model Limitations

        - **R² = 0.50 means half the variance is unexplained.** The model catches trends
          but misses sudden events (regulatory announcements, exchange collapses, black swans).
        - **Only tested in a bull market.** Walk-forward evaluation runs from Jan 2024 to Jul 2025,
          which is predominantly a bull market. Performance in a prolonged bear market is unknown.
        - **LSTM/GRU failed** due to insufficient training data (~1,000 rows). With more data
          (5,000+ rows), deep learning might outperform tree models.
        - **No feature for breaking news or social media sentiment.** The model only sees
          structured numerical data, not tweets, headlines, or regulatory announcements.

        ### Data Limitations

        - **FRED macro data is monthly**, forward-filled into daily rows. The model can't
          isolate the impact of a specific CPI release day.
        - **Google Trends normalization** is relative to the request window (0-100 scale).
          Incremental fetches require a 90-day overlap to maintain comparability.
        - **On-chain data** is limited to 5 free CoinMetrics metrics. Paid metrics
          (e.g., supply activity, transfer volumes) might improve predictions.
        - **No ETF flow data** in the model — only 18 months available, not enough history.

        ### Strategy Lab Limitations

        - **No transaction fees** are simulated (exchanges charge 0.1-0.5% per trade).
        - **No slippage** — real orders may fill at worse prices, especially with leverage.
        - **7-day windows are fixed** — a real strategy might use variable holding periods.
        - **Liquidation is simplified** — real exchanges have funding rates, margin calls,
          and partial liquidation mechanisms that aren't modeled.
        - **Past performance does not predict future results.** This is a portfolio project,
          not financial advice.

        ### What Could Improve It

        - More training data (currently Jan 2020 - present, ~2,000 rows)
        - Real-time sentiment from social media (Twitter/X, Reddit)
        - ETF flow data once more history accumulates
        - Ensemble of XGBoost + Random Forest for more robust predictions
        - Bayesian hyperparameter tuning (Optuna) — in progress
        """)
