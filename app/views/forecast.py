"""
Page 2 — 7-Day Forecast

Current prediction, confidence level, drift monitoring.
Automated logging via GitHub Actions — no manual button needed.
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from predict import predict_current, check_drift
from components import styled_metric


def render():
    st.markdown("<h1 style='margin-bottom:4px;'>7-Day Bitcoin Forecast</h1>", unsafe_allow_html=True)

    # Current prediction
    drift = check_drift()
    prediction = predict_current()

    if 'error' in prediction:
        st.error(prediction['error'])
        return

    # Show the data date clearly
    st.caption(f"Prediction based on data from {prediction['prediction_date']}")

    # Main prediction display
    is_up = prediction['direction'] == 'UP'
    direction_color = 'emerald' if is_up else 'rose'
    price_delta = prediction['predicted_price'] - prediction['btc_price']

    cols = st.columns(3)
    with cols[0]:
        styled_metric(
            "7-Day Direction",
            prediction['direction'],
            f"{prediction['predicted_return']:+.2%}",
            color=direction_color,
        )
    with cols[1]:
        styled_metric(
            "Current BTC Price",
            f"${prediction['btc_price']:,.0f}",
            color='blue',
        )
    with cols[2]:
        styled_metric(
            "Predicted Price (7d)",
            f"${prediction['predicted_price']:,.0f}",
            f"${price_delta:+,.0f}",
            color=direction_color,
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Confidence and details
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Prediction Details")

        confidence_colors = {'HIGH': 'emerald', 'MEDIUM': 'amber', 'LOW': 'rose'}
        confidence_descriptions = {
            'HIGH': 'Model predicts >5% move — 66% direction accuracy across 586 historical predictions',
            'MEDIUM': 'Model predicts 2–5% move — 53% accuracy, modest edge over baseline',
            'LOW': 'Model predicts <2% move — near coin-flip (51%). Model sees no strong signal.',
        }

        conf = prediction['confidence']
        conf_color = {'HIGH': '#10b981', 'MEDIUM': '#f59e0b', 'LOW': '#f43f5e'}[conf]
        st.markdown(
            f"<span style='color:{conf_color};font-weight:700;font-family:JetBrains Mono,monospace;'>"
            f"Confidence: {conf}</span>",
            unsafe_allow_html=True,
        )
        st.caption(confidence_descriptions[conf])

        st.markdown(f"**Prediction date:** {prediction['prediction_date']}")
        st.markdown(f"**Target date:** {prediction['target_date']}")
        st.markdown(f"**Predicted return:** {prediction['predicted_return']:+.2%}")

    with col_right:
        st.markdown("### How It Works")
        st.markdown(
            """
            The model uses **XGBoost** trained on 52 features across 5 asset classes
            (BTC, S&P 500, NASDAQ, Gold, Dollar Index) plus on-chain metrics and
            sentiment data.

            Walk-forward validated on **8 years of data (2018–2026)** across multiple
            market regimes: **76.7% direction accuracy** vs 52% naive baseline (always
            predict UP). High-confidence predictions (>5% predicted move) reach even
            higher accuracy across 2,467 historical windows.

            **Confidence** reflects the size of the predicted return — when the model
            sees strong aligned signals it predicts a large move and is more accurate.
            When signals are mixed it hedges toward zero (LOW confidence, ~coin flip).
            """
        )

    st.markdown("---")
    st.caption(
        "Predictions are logged automatically via GitHub Actions daily at 07:00 UTC. "
        "Visit the Strategy Lab to see historical backtest results."
    )

    # Drift banner at the bottom so it doesn't dominate the page
    if drift['status'] == 'WARNING':
        st.error(f"Model drift detected: {drift['message']}")
    elif drift['status'] == 'OK':
        st.success(drift['message'])
    else:
        st.info(drift['message'])
