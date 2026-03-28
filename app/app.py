"""
Bitcoin Price Predictor — Streamlit App

Three-page app:
1. Dashboard: current 7-day prediction with confidence and key indicators
2. Paper Trading: historical prediction log with simulated P&L
3. Model Performance: evaluation metrics, charts, regime analysis
"""

import streamlit as st

st.set_page_config(
    page_title="BTC Price Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("BTC Price Predictor")
st.sidebar.markdown("ML-powered 7-day Bitcoin forecasts")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Paper Trading", "Model Performance"],
)

if page == "Dashboard":
    from pages.dashboard import render
    render()
elif page == "Paper Trading":
    from pages.paper_trading import render
    render()
elif page == "Model Performance":
    from pages.performance import render
    render()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built by [Marc Seger](https://github.com/Marc-Seger)  \n"
    "XGBoost + walk-forward validation"
)
