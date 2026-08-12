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
        before the outcome is known. Read this as the running record of an open experiment,
        not as a forecast to act on: the model has no demonstrated edge (see Methodology).
        Confidence labels reflect predicted move size only, and do not indicate reliability.

        **3. Model Performance**
        Walk-forward validation results, and the August 2026 audit that corrected them:
        leak-free vs original figures, performance across 8 distinct market phases
        (2018 bear market through 2024–2026 cycle), and live production accuracy.

        **4. Documentation**
        This page.

        > **Strategy Lab** is currently unlinked. Its backtests were driven by the leaky
        > walk-forward file, so every equity curve it drew was built on predictions that
        > had seen the future. The code is unchanged in the repository; the page returns
        > once it can be rebuilt on leak-free predictions.

        ### Data Updates
        The full pipeline runs automatically every day at 22:00 UTC via GitHub Actions,
        two hours after the US market close so all daily candles are final:
        fetch → validate → train → predict → upload. Downstream steps are gated on
        validation passing, so a bad data day cannot overwrite a good model.
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

        ### The Target Leak (August 2026 audit)

        The headline number on this app used to be **76.7% direction accuracy**. It was wrong.
        Two separate errors, both in the evaluation, neither in the model.

        **1. Overlapping windows counted as independent.** 2,467 daily predictions were described
        as non-overlapping. They were not: consecutive 7-day windows share six of their seven days,
        so one correct multi-week call was counted up to seven times. Scoring only genuinely
        independent windows gives 73.3%, across 367 windows.

        **2. Target leakage.** `Target_Return_7d = close.shift(-7) / close - 1`. The walk-forward
        loop trained on `df.iloc[:i]` to predict row `i` — but rows `i-6 .. i-1` sit inside that
        training slice, and their targets are computed from closing prices on days `i .. i+5`.
        The model was trained on the answer to the question it was about to be asked.

        Re-running the identical evaluation with those seven rows purged
        (`scripts/evaluate_leakfree.py`):

        | Variant | Direction Accuracy | R² | corr(pred, actual) |
        |---------|--------------------|----|--------------------|
        | As built (leaky) | 73.3% | 0.571 | 0.774 |
        | **Leak-free** | **48.5%** | **-0.308** | **-0.030** |
        | Naive (always UP) | 52.6% | — | — |

        A negative R² means worse than predicting the average return. A correlation of -0.03 means
        the predictions carry no information about the outcome. **This model has no demonstrated edge.**

        The live production log independently confirms it: predictions logged daily since March 2026,
        each recorded before the outcome was known, sitting within a point of the leak-free backtest.

        ### Models Tested

        | Model | R² | Direction Accuracy | Notes |
        |-------|----|--------------------|-------|
        | **XGBoost (leak-free)** | **-0.308** | **48.5%** | 367 independent windows, 2019–2026 |
        | XGBoost (original) | 0.571 | 73.3% | Retracted — target leakage |
        | LSTM | -1.14 | 50.0% | 74 independent windows, 2022–2025 |
        | GRU | -2.01 | 54.1% | 74 independent windows, 2022–2025 |

        **Naive baseline** (always predict UP): **52.6%**. Nothing tested beats it.

        ### Why the XGBoost vs Deep Learning Comparison Was Unfair

        This app used to claim tree models beat LSTM/GRU on this dataset. That comparison was not
        like for like. `evaluate_dl.py` lists "drop last 7 training rows (7-day target peeks into
        test)" as its first safeguard, so the neural nets always ran the honest race. The XGBoost
        path never purged those rows. The trees won a race the networks were not allowed to run.

        Evaluated on equal terms, all three sit near the coin-flip line. The reasons deep learning
        was never likely to work here still hold and are worth stating plainly:

        - **Insufficient data**: a 30-day input window across 52 features gives a sequence model far
          more parameters than signal. With ~3,000 daily rows it fits noise.
        - **Noise dominance**: 7-day BTC returns are close to random. No architecture extracts a
          signal that is not there.

        ### Confidence Levels

        The predicted return magnitude was used as a confidence proxy, and the app previously
        reported that large predicted moves reached ~90% accuracy. That was the same leak.
        Leak-free, the largest-move tercile scores **44.0%**, *below* the smallest-move tercile
        at 48.8%. **Predicted move size carries no information about whether the call is right.**
        Confidence labels are retained in the log for continuity only.

        ### What Would Have To Change

        Three fixes before any future number here should be trusted, in order of importance:

        1. **Purge the last 7 training rows** at every walk-forward step. This is the leak above.
        2. **Select features inside the walk-forward.** The 52 features were picked from 269 by
           importance measured on the full eight years, then scored on those same eight years.
        3. **Tune against a held-out period.** The hyperparameters came from 100 Optuna trials,
           each scored on the entire history, keeping the highest. That is a maximum over 100
           attempts on the data being reported, not an out-of-sample result.

        None of this is done yet, and none of it is expected to rescue the model. Seven-day BTC
        direction from free daily data may simply not be predictable, and a null result is the
        honest outcome rather than a failure.

        ### Monitoring

        A weekly GitHub Actions job checks pipeline health: data freshness, prediction cadence, and
        how long since the model was last retuned. It reports live accuracy as a number rather than
        an alarm, since ~50% is the expected state, and alerts only if accuracy moves far from that
        in **either** direction. Given this history, a sudden jump is more likely to be a new leak
        than a discovery.

        That job was itself silently broken from March to August 2026. A manual repair to the trade
        log left one date in a different format from the rest, the file read threw an error, a broad
        `except` swallowed it, and the job reported "all clear" every week with a green checkmark.
        A check that could not run was being reported as the thing it checked passing. Every check
        now raises an alert when it cannot complete.
        """)

    # ══════════════════════════════════════════
    # LIMITATIONS
    # ══════════════════════════════════════════
    with tab_limits:
        st.subheader("Known Limitations & Flaws")
        st.markdown("""
        ### Model Limitations

        - **There is no edge.** 48.5% leak-free direction accuracy against a 52.6% naive baseline,
          with a correlation of -0.03 between prediction and outcome. The live log agrees. This is
          not a trading system and not a weak trading system — it is a null result, published as one.
          See the Methodology tab for the full audit.
        - **R² is negative** (-0.308). The model explains less variance than simply predicting the
          mean return.
        - **The evaluation is still not clean.** The leak-free re-run reuses hyperparameters that
          were tuned against the leaky objective, and features selected on the full history. It
          isolates the leak; it is not an unbiased out-of-sample estimate. The defensible claim is
          "no demonstrated edge", not "proven unpredictable".
        - **No feature for breaking news.** Regulatory announcements, exchange collapses, ETF approval
          surprises, and macro shocks are invisible to the model — it only sees structured numerical data.
        - **LSTM/GRU also failed**, at 50.0% and 54.1%, with negative R². Unlike XGBoost, their
          evaluation was leak-free from the start.

        ### Data Limitations

        - **FRED macro data is monthly**, forward-filled to daily. The model cannot isolate the
          impact of a specific CPI release — it sees the same value for 30 days.
        - **Google Trends is weekly**, forward-filled to daily. The scale is normalised per request
          window (0–100), requiring multi-window rescaling for historical consistency.
        - **ETF flows start Jan 2024** — not enough history to use as a model feature, but shown
          in the dashboard as a retail vs institutional sentiment indicator.
        - **On-chain data** is limited to 5 free CoinMetrics metrics. Paid metrics (supply activity,
          adjusted transfer volumes) might improve predictions.

        ### Strategy Lab (unlinked)

        The page is currently not reachable from the sidebar. Its backtests ran on the leaky
        walk-forward file, so the equity curves were compounding predictions that had seen the
        future — the headline returns were fiction, not optimism. The code is untouched in the
        repository and the page returns once it can be rebuilt on leak-free predictions.

        Its original caveats still apply and are worth keeping on record: fees and slippage were
        modelled only approximately, 7-day windows were fixed, liquidation was simplified, and
        feature-selection look-ahead inflated the win rate independently of the leak.

        ### What Would Actually Help

        Ordered by expected impact, and honestly: the first three are corrections, not upgrades.

        1. **Fix the evaluation** (purge the leak, select features inside the walk-forward, tune
           against a held-out period). Until this is done, no number here means anything.
        2. **Accept the likely answer.** 7-day BTC direction from free daily data may not be
           predictable. A clean null result is a legitimate finding, and is the current one.
        3. **Rebuild the Strategy Lab** on leak-free predictions — useful mainly as a
           demonstration of what the leak was worth in dollars.
        4. Longer horizons or volatility targets, which are generally more predictable than
           short-horizon direction.
        5. Data the model cannot currently see: options-implied volatility, order book depth,
           real-time social sentiment. Unlikely to change the conclusion on its own.

        **Past performance does not predict future results.** This is a portfolio project,
        not financial advice.
        """)
