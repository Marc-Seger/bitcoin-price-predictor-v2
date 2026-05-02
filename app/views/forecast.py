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
from performance import _build_prediction_log


def render():
    st.markdown("<h1 style='margin-bottom:4px;'>7-Day Bitcoin Forecast</h1>", unsafe_allow_html=True)

    # Current prediction
    drift = check_drift()
    prediction = predict_current()

    if 'error' in prediction:
        st.error(prediction['error'])
        return

    days_stale = prediction.get('days_stale', 0)
    if days_stale > 2:
        st.warning(
            f"Prediction is {days_stale} day{'s' if days_stale != 1 else ''} old "
            f"(based on data from {prediction['prediction_date']}) — "
            f"the pipeline may not have run recently."
        )

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

    # Drift banner
    if drift['status'] == 'WARNING':
        st.error(f"Model drift detected: {drift['message']}")
    elif drift['status'] == 'OK':
        st.success(drift['message'])
    else:
        st.info(drift['message'])

    # ─── Prediction log ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Past Predictions")
    st.caption(
        "Every prediction the model has made — walk-forward historical (2019 → early 2026) "
        "plus live production predictions (March 2026 → now).<br>"
        "<span style='color:#3b82f6;'>●</span> = live &nbsp; ⏳ = pending &nbsp; ✅ = correct &nbsp; ❌ = wrong",
        unsafe_allow_html=True,
    )

    log = _build_prediction_log()
    if log.empty:
        st.warning("No prediction data found.")
    else:
        n_options = {'Last 30': 30, 'Last 90': 90, 'Last 180': 180, 'All': len(log)}
        n_choice  = st.selectbox("Show", list(n_options.keys()), index=0, key='forecast_pred_log_n')
        display_log = log.iloc[::-1].head(n_options[n_choice])

        # ── Mini KPI row ─────────────────────────────────────────────────────
        resolved   = display_log[~display_log['pending']]
        n_up       = int((display_log['direction'] == 'UP').sum())
        n_down     = int((display_log['direction'] == 'DOWN').sum())
        n_correct  = int(resolved['correct'].sum()) if not resolved.empty else 0
        n_resolved = len(resolved)
        sr_label   = f"{n_correct}/{n_resolved} — {n_correct/n_resolved*100:.0f}%" if n_resolved else "—"
        avg_move   = display_log['predicted_return'].abs().mean()
        avg_label  = f"{avg_move*100:.1f}%" if pd.notna(avg_move) else "—"

        def _mini_kpi(label, value, color='#8899b4'):
            return (
                f"<div style='background:#171f30;border:1px solid #263354;border-radius:8px;"
                f"padding:10px 14px;text-align:center;'>"
                f"<div style='font-size:10px;text-transform:uppercase;letter-spacing:0.7px;"
                f"color:#56657e;font-weight:600;margin-bottom:4px;'>{label}</div>"
                f"<div style='font-size:17px;font-weight:700;color:{color};"
                f"font-family:JetBrains Mono,monospace;'>{value}</div>"
                f"</div>"
            )

        kpi_html = (
            f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px;'>"
            + _mini_kpi("↑ Up predictions",   str(n_up),   '#10b981')
            + _mini_kpi("↓ Down predictions", str(n_down), '#f43f5e')
            + _mini_kpi("Success rate",        sr_label,    '#3b82f6')
            + _mini_kpi("Avg predicted move",  avg_label,   '#f59e0b')
            + "</div>"
        )
        st.markdown(kpi_html, unsafe_allow_html=True)

        conf_colors = {'HIGH': '#10b981', 'MEDIUM': '#f59e0b', 'LOW': '#56657e'}

        def _result_cell(row):
            if row['pending']:
                return "<td style='text-align:center;font-size:15px;'>⏳</td>"
            return "<td style='text-align:center;font-size:15px;'>✅</td>" if row['correct'] else "<td style='text-align:center;font-size:15px;'>❌</td>"

        def _pct_cell(val):
            if pd.isna(val):
                return "<td style='text-align:right;color:#56657e;'>—</td>"
            clr = '#10b981' if val >= 0 else '#f43f5e'
            return f"<td style='text-align:right;color:{clr};font-weight:600;'>{val*100:+.1f}%</td>"

        def _price_cell(val):
            if pd.isna(val):
                return "<td style='text-align:right;color:#56657e;'>—</td>"
            return f"<td style='text-align:right;color:#e8edf5;'>${val:,.0f}</td>"

        def _delta_cell(row):
            if row['pending'] or pd.isna(row['actual_return']) or pd.isna(row['predicted_return']):
                return "<td style='text-align:right;color:#56657e;'>—</td>"
            delta = (row['actual_return'] - row['predicted_return']) * 100
            clr = '#10b981' if delta >= 0 else '#f43f5e'
            return f"<td style='text-align:right;color:{clr};'>{delta:+.1f}%</td>"

        headers = ['Date', 'Target d+7', 'Direction', 'Predicted', 'Confidence',
                   'BTC at Pred.', 'BTC at d+7', 'Actual', 'Delta', 'Result']
        th = "".join(
            f"<th style='text-align:{'center' if h in ('Direction','Confidence','Result') else 'right' if h not in ('Date','Target d+7') else 'left'}"
            f";color:#56657e;padding:6px 10px;font-size:11px;text-transform:uppercase;letter-spacing:0.6px;"
            f"background:#0f1520;position:sticky;top:0;z-index:1;'>{h}</th>"
            for h in headers
        )

        body = ""
        for pred_date, row in display_log.iterrows():
            dir_color  = '#10b981' if row['direction'] == 'UP' else '#f43f5e'
            dir_arrow  = '↑' if row['direction'] == 'UP' else '↓'
            conf_color = conf_colors.get(str(row['confidence']), '#56657e')
            src_dot    = "<span style='font-size:8px;color:#3b82f6;vertical-align:super;'>●</span>" if row['source'] == 'live' else ""
            _td        = pd.to_datetime(row['target_date'], errors='coerce')
            target_str = _td.strftime('%Y-%m-%d') if pd.notna(_td) else '—'

            body += (
                f"<tr style='border-top:1px solid #1e2940;'>"
                f"<td style='padding:6px 10px;color:#8899b4;white-space:nowrap;'>{pred_date.strftime('%Y-%m-%d')}{src_dot}</td>"
                f"<td style='padding:6px 10px;color:#56657e;white-space:nowrap;'>{target_str}</td>"
                f"<td style='text-align:center;color:{dir_color};font-weight:700;'>{dir_arrow} {row['direction']}</td>"
                + _pct_cell(row['predicted_return'])
                + f"<td style='text-align:center;color:{conf_color};font-weight:600;font-size:11px;'>{row['confidence']}</td>"
                + _price_cell(row['btc_at_prediction'])
                + _price_cell(row['btc_at_target'])
                + _pct_cell(row['actual_return'])
                + _delta_cell(row)
                + _result_cell(row)
                + "</tr>"
            )

        st.markdown(
            f"<div style='overflow-x:auto;overflow-y:auto;max-height:420px;border-radius:8px;'>"
            f"<table style='width:100%;border-collapse:collapse;background:#171f30;"
            f"font-family:JetBrains Mono,monospace;font-size:12px;'>"
            f"<thead><tr>{th}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>",
            unsafe_allow_html=True,
        )
