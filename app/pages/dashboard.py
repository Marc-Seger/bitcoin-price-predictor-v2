"""
Dashboard page — Current prediction, confidence, drift status.
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from predict import predict_current, check_drift, log_prediction, update_outcomes


def render():
    st.title("7-Day Bitcoin Forecast")

    # Drift warning banner
    drift = check_drift()
    if drift['status'] == 'WARNING':
        st.error(f"Model drift detected: {drift['message']}")
    elif drift['status'] == 'OK':
        st.success(drift['message'])
    else:
        st.info(drift['message'])

    st.markdown("---")

    # Current prediction
    prediction = predict_current()

    if 'error' in prediction:
        st.error(prediction['error'])
        return

    # Main prediction display
    col1, col2, col3 = st.columns(3)

    with col1:
        direction_color = "green" if prediction['direction'] == 'UP' else "red"
        direction_arrow = "^" if prediction['direction'] == 'UP' else "v"
        st.metric(
            label="7-Day Direction",
            value=prediction['direction'],
            delta=f"{prediction['predicted_return']:+.2%}",
        )

    with col2:
        st.metric(
            label="Current BTC Price",
            value=f"${prediction['btc_price']:,.0f}",
        )

    with col3:
        st.metric(
            label="Predicted Price (7d)",
            value=f"${prediction['predicted_price']:,.0f}",
            delta=f"${prediction['predicted_price'] - prediction['btc_price']:+,.0f}",
        )

    st.markdown("---")

    # Confidence and details
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Prediction Details")

        confidence_colors = {'HIGH': 'green', 'MEDIUM': 'orange', 'LOW': 'red'}
        confidence_descriptions = {
            'HIGH': 'Large predicted move — historically 95% accurate on direction',
            'MEDIUM': 'Moderate predicted move — historically 75% accurate',
            'LOW': 'Small predicted move — historically 66% accurate',
        }

        conf = prediction['confidence']
        st.markdown(f"**Confidence:** :{confidence_colors[conf]}[{conf}]")
        st.caption(confidence_descriptions[conf])

        st.markdown(f"**Prediction date:** {prediction['prediction_date']}")
        st.markdown(f"**Target date:** {prediction['target_date']}")
        st.markdown(f"**Predicted return:** {prediction['predicted_return']:+.2%}")

    with col_right:
        st.subheader("How It Works")
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

    # Log prediction button
    st.markdown("---")
    if st.button("Log this prediction to paper trading"):
        log_prediction(prediction)
        update_outcomes()
        st.success(f"Prediction logged for {prediction['prediction_date']}")
        st.rerun()
