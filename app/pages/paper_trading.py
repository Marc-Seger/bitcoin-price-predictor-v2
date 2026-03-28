"""
Paper Trading page — Prediction log, accuracy tracking, simulated P&L.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from predict import load_trade_log, update_outcomes


def render():
    st.title("Paper Trading Log")
    st.caption("Track model predictions against actual outcomes — no real money involved.")

    # Update outcomes for any resolved predictions
    log = update_outcomes()

    if log.empty:
        st.info(
            "No predictions logged yet. Go to the Dashboard and click "
            "'Log this prediction' to start tracking."
        )
        return

    # Summary metrics
    resolved = log.dropna(subset=['correct'])
    pending = log[log['correct'].isna()]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Predictions", len(log))
    with col2:
        st.metric("Resolved", len(resolved))
    with col3:
        st.metric("Pending", len(pending))
    with col4:
        if len(resolved) > 0:
            accuracy = resolved['correct'].mean()
            st.metric("Direction Accuracy", f"{accuracy:.0%}")
        else:
            st.metric("Direction Accuracy", "—")

    st.markdown("---")

    # Simulated portfolio
    if len(resolved) > 0:
        st.subheader("Simulated Portfolio")
        st.caption(
            "Starting with $10,000. Goes long when model predicts UP, "
            "stays in cash when model predicts DOWN. No leverage, no shorts."
        )

        portfolio_value = 10000.0
        portfolio_history = [{'date': 'Start', 'value': portfolio_value}]

        for _, row in resolved.iterrows():
            if row['direction'] == 'UP':
                # Went long — portfolio moves with actual return
                portfolio_value *= (1 + row['actual_return'])
            # If DOWN, stayed in cash — no change

            portfolio_history.append({
                'date': row['target_date'].strftime('%Y-%m-%d') if hasattr(row['target_date'], 'strftime') else str(row['target_date']),
                'value': portfolio_value,
            })

        portfolio_df = pd.DataFrame(portfolio_history)

        col_port, col_stats = st.columns([2, 1])

        with col_port:
            st.line_chart(portfolio_df.set_index('date')['value'])

        with col_stats:
            total_return = (portfolio_value / 10000 - 1) * 100
            st.metric("Portfolio Value", f"${portfolio_value:,.0f}")
            st.metric("Total Return", f"{total_return:+.1f}%")

            # Buy & hold comparison
            if len(resolved) > 0:
                first_price = resolved.iloc[0]['btc_price_at_prediction']
                last_price = resolved.iloc[-1]['actual_price']
                bh_return = (last_price / first_price - 1) * 100
                st.metric("Buy & Hold Return", f"{bh_return:+.1f}%")

                # Win/loss streaks
                correct_series = resolved['correct'].astype(bool)
                current_streak = 0
                for val in reversed(correct_series.values):
                    if val == correct_series.values[-1]:
                        current_streak += 1
                    else:
                        break
                streak_type = "win" if correct_series.values[-1] else "loss"
                st.metric("Current Streak", f"{current_streak} {streak_type}")

    st.markdown("---")

    # Prediction log table
    st.subheader("Full Prediction Log")

    display_log = log.copy()
    display_log['predicted_return'] = display_log['predicted_return'].apply(lambda x: f"{x:+.2%}")
    display_log['actual_return'] = display_log['actual_return'].apply(
        lambda x: f"{x:+.2%}" if pd.notna(x) else "Pending"
    )
    display_log['btc_price_at_prediction'] = display_log['btc_price_at_prediction'].apply(
        lambda x: f"${x:,.0f}"
    )
    display_log['actual_price'] = display_log['actual_price'].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
    )
    display_log['correct'] = display_log['correct'].apply(
        lambda x: "Yes" if x == True else ("No" if x == False else "Pending")
    )

    # Show most recent first
    display_cols = [
        'prediction_date', 'target_date', 'direction', 'confidence',
        'predicted_return', 'actual_return', 'btc_price_at_prediction',
        'actual_price', 'correct',
    ]
    st.dataframe(
        display_log[display_cols].iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )
