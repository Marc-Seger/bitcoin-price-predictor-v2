"""
Bitcoin Price Predictor — Streamlit App

5-page app:
1. Financial Dashboard: price charts, indicators, sentiment, cross-asset
2. 7-Day Forecast: current prediction with confidence and drift detection
3. Strategy Lab: backtest strategies with leverage, SL/TP
4. Model Performance: evaluation metrics, regime analysis
5. Documentation: how to use, data sources, methodology, limitations
"""

import streamlit as st

st.set_page_config(
    page_title="BTC Price Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("BTC Price Predictor")
st.sidebar.markdown("ML-powered Bitcoin analysis & forecasting")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "7-Day Forecast", "Strategy Lab", "Model Performance", "Documentation"],
)

if page == "Dashboard":
    from pages.dashboard import render
elif page == "7-Day Forecast":
    from pages.forecast import render
elif page == "Strategy Lab":
    from pages.strategy_lab import render
elif page == "Model Performance":
    from pages.performance import render
elif page == "Documentation":
    from pages.documentation import render

render()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Built by [Marc Seger](https://github.com/Marc-Seger)  \n"
    "XGBoost | Walk-forward validation  \n"
    "Not financial advice."
)
