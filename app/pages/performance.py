"""
Model Performance page — Evaluation results, charts, regime analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from config import RESULTS_DIR


def load_results(filename: str) -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return pd.DataFrame()


def render():
    st.title("Model Performance")
    st.caption("Walk-forward validation results — every prediction made without seeing future data.")

    # ─── Model comparison table ───
    st.subheader("Model Comparison")
    st.markdown(
        "All models evaluated with walk-forward validation on 7-day returns. "
        "Non-overlapping predictions (every 7th day) for honest metrics."
    )

    summary_path = os.path.join(RESULTS_DIR, 'phase3_full_summary.csv')
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path, index_col=0)
        st.dataframe(summary, use_container_width=True)
    else:
        st.warning("No summary file found. Run model evaluation first.")

    st.markdown("---")

    # ─── XGBoost deep dive ───
    st.subheader("XGBoost 7-Day Predictions vs Actual")

    xgb = load_results('XGB_7d_walkforward_results.csv')
    if xgb.empty:
        st.warning("No XGBoost 7-day results found.")
        return

    # Predictions vs actual chart
    chart_data = xgb[['actual', 'predicted']].copy()
    chart_data.columns = ['Actual Return', 'Predicted Return']
    st.line_chart(chart_data)

    # ─── Direction accuracy over time (rolling) ───
    st.subheader("Rolling Direction Accuracy")
    st.caption("30-day rolling window — shows how accuracy varies over time.")

    xgb['correct'] = ((xgb['actual'] > 0) == (xgb['predicted'] > 0)).astype(int)
    rolling_acc = xgb['correct'].rolling(30).mean()
    st.line_chart(rolling_acc, y_label="Accuracy", x_label="Date")

    st.markdown("---")

    # ─── Market regime breakdown ───
    st.subheader("Performance by Market Phase")

    def classify_phase(date):
        d = pd.Timestamp(date)
        if d < pd.Timestamp('2024-04-15'):
            return 'ETF Rally (Jan-Apr 2024)'
        elif d < pd.Timestamp('2024-11-01'):
            return 'Consolidation (May-Oct 2024)'
        elif d < pd.Timestamp('2025-02-01'):
            return 'Post-Election (Nov 24-Jan 25)'
        else:
            return 'Maturation (Feb-Jul 2025)'

    xgb['phase'] = xgb.index.map(classify_phase)

    phase_stats = []
    for phase in sorted(xgb['phase'].unique()):
        data = xgb[xgb['phase'] == phase]
        phase_stats.append({
            'Phase': phase,
            'Predictions': len(data),
            'Avg Return': f"{data['actual'].mean():.2%}",
            'Direction Accuracy': f"{data['correct'].mean():.1%}",
            'R²': f"{np.corrcoef(data['actual'], data['predicted'])[0,1]**2:.3f}",
        })

    st.dataframe(pd.DataFrame(phase_stats), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ─── Confidence vs accuracy ───
    st.subheader("Confidence vs Accuracy")
    st.markdown(
        "When the model predicts a **larger move**, it's significantly more accurate. "
        "This means high-confidence predictions are much more reliable."
    )

    xgb['abs_pred'] = xgb['predicted'].abs()
    q33 = xgb['abs_pred'].quantile(0.33)
    q66 = xgb['abs_pred'].quantile(0.66)

    confidence_data = []
    for label, mask in [
        ('Low (small move)', xgb['abs_pred'] <= q33),
        ('Medium', (xgb['abs_pred'] > q33) & (xgb['abs_pred'] <= q66)),
        ('High (large move)', xgb['abs_pred'] > q66),
    ]:
        subset = xgb[mask]
        confidence_data.append({
            'Confidence': label,
            'Avg Predicted Move': f"{subset['abs_pred'].mean():.2%}",
            'Direction Accuracy': f"{subset['correct'].mean():.1%}",
            'Count': len(subset),
        })

    st.dataframe(pd.DataFrame(confidence_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ─── Why LSTM/GRU failed ───
    st.subheader("Why Deep Learning Failed Here")
    st.markdown(
        """
        LSTM and GRU models performed at coin-flip level (~50% direction accuracy).
        This is a valuable negative result:

        - **Insufficient data**: ~1,000 training rows is too few for recurrent neural networks
          to learn meaningful temporal patterns.
        - **Noise dominance**: Financial returns are extremely noisy. Tree-based models
          handle this better because they make discrete split decisions rather than trying
          to learn smooth continuous mappings.
        - **XGBoost wins** because it naturally handles mixed feature types, is robust to
          noise, and doesn't require thousands of training sequences.

        This informed our decision to use XGBoost as the production model.
        """
    )
