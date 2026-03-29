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


def styled_metric(label, value, delta=None, color='blue', invert_delta=False):
    """KPI card with colored left border. invert_delta: red for positive, green for negative."""
    colors = {
        'blue': '#3b82f6', 'emerald': '#10b981', 'amber': '#f59e0b',
        'rose': '#f43f5e', 'violet': '#8b5cf6', 'cyan': '#06b6d4',
    }
    border_color = colors.get(color, colors['blue'])
    delta_html = ""
    if delta is not None:
        is_positive = str(delta).lstrip().startswith("+")
        if invert_delta:
            delta_color = "#f43f5e" if is_positive else "#10b981"
        else:
            delta_color = "#10b981" if is_positive else "#f43f5e"
        delta_html = f"<div style='font-size:12px;color:{delta_color};font-family:JetBrains Mono,monospace;'>{delta}</div>"
    st.markdown(f"""
    <div style="background:#171f30;border:1px solid #263354;border-left:3px solid {border_color};
                border-radius:8px;padding:12px 14px;">
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.8px;
                    color:#56657e;font-weight:600;margin-bottom:4px;">{label}</div>
        <div style="font-size:20px;font-weight:700;color:#e8edf5;
                    font-family:JetBrains Mono,monospace;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render():
    st.markdown("<h1 style='margin-bottom:4px;'>7-Day Bitcoin Forecast</h1>", unsafe_allow_html=True)

    # Drift warning banner
    drift = check_drift()
    if drift['status'] == 'WARNING':
        st.error(f"Model drift detected: {drift['message']}")
    elif drift['status'] == 'OK':
        st.success(drift['message'])
    else:
        st.info(drift['message'])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Current prediction
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
            'HIGH': 'Large predicted move — historically 95% accurate on direction',
            'MEDIUM': 'Moderate predicted move — historically 75% accurate',
            'LOW': 'Small predicted move — historically 66% accurate',
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

            Walk-forward validated: R² = 0.50, 76% direction accuracy on
            non-overlapping 7-day predictions.

            **Confidence levels** are based on the size of the predicted move —
            larger predicted moves have historically been much more accurate.
            """
        )

    st.markdown("---")
    st.caption(
        "Predictions are logged automatically via GitHub Actions daily at 07:00 UTC. "
        "Visit the Strategy Lab to see historical backtest results."
    )
