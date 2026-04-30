"""
Model Performance page — Evaluation results, charts, regime analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from config import RESULTS_DIR, MASTER_DF_PATH
from components import DARK_LAYOUT, styled_metric

APP_DIR = os.path.join(os.path.dirname(__file__), '..')
TRADE_LOG_PATH = os.path.join(APP_DIR, 'data', 'trade_log.csv')


def load_results(filename: str) -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def _build_prediction_log():
    """Build a unified prediction log combining walk-forward history and live trade log."""
    prices = pd.Series(dtype=float)
    if os.path.exists(MASTER_DF_PATH):
        prices = pd.read_csv(
            MASTER_DF_PATH, index_col=0, parse_dates=True,
            usecols=['date', 'Close_BTC'], low_memory=False
        )['Close_BTC'].dropna()

    def _btc_price(date):
        if prices.empty:
            return float('nan')
        ts = pd.Timestamp(date)
        if ts in prices.index:
            return float(prices[ts])
        try:
            return float(prices.asof(ts))
        except Exception:
            return float('nan')

    # ── Walk-forward history ─────────────────────────────────────────────
    wf_path = os.path.join(RESULTS_DIR, 'XGB_7d_walkforward_results.csv')
    if not os.path.exists(wf_path):
        wf = pd.DataFrame()
    else:
        wf_raw = pd.read_csv(wf_path, index_col=0, parse_dates=True)
        rows = []
        for pred_date, row in wf_raw.iterrows():
            target_date = pred_date + pd.Timedelta(days=7)
            btc_pred   = _btc_price(pred_date)
            btc_target = _btc_price(target_date) if target_date <= prices.index.max() else float('nan')
            pred_ret   = float(row['predicted'])
            actual_ret = float(row['actual'])
            correct    = 1 if (pred_ret > 0) == (actual_ret > 0) else 0
            abs_p = abs(pred_ret)
            conf = 'HIGH' if abs_p > 0.05 else ('MEDIUM' if abs_p > 0.02 else 'LOW')
            rows.append({
                'prediction_date':   pred_date,
                'target_date':       target_date,
                'predicted_return':  pred_ret,
                'direction':         'UP' if pred_ret > 0 else 'DOWN',
                'confidence':        conf,
                'btc_at_prediction': btc_pred,
                'btc_at_target':     btc_target,
                'actual_return':     actual_ret,
                'correct':           correct,
                'pending':           False,
                'source':            'walk-forward',
            })
        wf = pd.DataFrame(rows).set_index('prediction_date')

    # ── Live production log ──────────────────────────────────────────────
    if not os.path.exists(TRADE_LOG_PATH):
        live = pd.DataFrame()
    else:
        tl = pd.read_csv(TRADE_LOG_PATH, parse_dates=['prediction_date', 'target_date'])
        tl = tl.set_index('prediction_date')
        rows = []
        for pred_date, row in tl.iterrows():
            pending  = pd.isna(row.get('actual_price')) or str(row.get('actual_price')) == ''
            pred_ret = float(row['predicted_return'])

            raw_btc_pred = row.get('btc_price_at_prediction')
            btc_pred = float(raw_btc_pred) if pd.notna(raw_btc_pred) else _btc_price(pred_date)

            btc_target = float(row['actual_price']) if not pending and pd.notna(row.get('actual_price')) else float('nan')

            raw_ret = row.get('actual_return')
            if pd.notna(raw_ret) and str(raw_ret) != '':
                actual_ret = float(raw_ret)
            elif not pending and not pd.isna(btc_pred) and btc_pred > 0:
                actual_ret = (btc_target / btc_pred) - 1
            else:
                actual_ret = float('nan')

            raw_correct = row.get('correct')
            if pd.notna(raw_correct) and str(raw_correct) != '':
                correct = int(float(raw_correct))
            elif not pending and not pd.isna(actual_ret):
                correct = 1 if (pred_ret > 0) == (actual_ret > 0) else 0
            else:
                correct = None

            rows.append({
                'prediction_date':   pred_date,
                'target_date':       row['target_date'],
                'predicted_return':  pred_ret,
                'direction':         str(row.get('direction', 'UP' if pred_ret > 0 else 'DOWN')),
                'confidence':        str(row.get('confidence', '—')),
                'btc_at_prediction': btc_pred,
                'btc_at_target':     btc_target,
                'actual_return':     actual_ret,
                'correct':           correct,
                'pending':           pending,
                'source':            'live',
            })
        live = pd.DataFrame(rows).set_index('prediction_date')

    # ── Merge — live rows take priority ─────────────────────────────────
    if wf.empty and live.empty:
        return pd.DataFrame()
    elif wf.empty:
        combined = live
    elif live.empty:
        combined = wf
    else:
        combined = wf[~wf.index.isin(live.index)]
        combined = pd.concat([combined, live])

    combined.index = pd.to_datetime(combined.index, errors='coerce')
    combined = combined[combined.index.notna()].sort_index()
    return combined


def render():
    st.markdown("<h1 style='margin-bottom:4px;'>Model Performance</h1>", unsafe_allow_html=True)
    st.caption("Walk-forward validation results — every prediction made without seeing future data.")

    # ─── Model comparison table ───
    st.markdown("### Model Comparison")
    st.markdown(
        "All models evaluated with walk-forward validation on 7-day returns. "
        "Non-overlapping predictions (every 7th day) for honest metrics."
    )

    xgb_path = os.path.join(RESULTS_DIR, 'XGB_7d_walkforward_results.csv')
    if os.path.exists(xgb_path):
        _r = pd.read_csv(xgb_path, index_col=0, parse_dates=True)
        _r['correct'] = ((_r['actual'] > 0) == (_r['predicted'] > 0)).astype(int)
        _no = _r.iloc[::7]
        _r2 = 1 - ((_no['actual'] - _no['predicted'])**2).sum() / ((_no['actual'] - _no['actual'].mean())**2).sum()
        _naive = (_r['actual'] > 0).mean()
        summary = pd.DataFrame([{
            'Model': 'XGBoost (tuned)', 'Target': '7-day return',
            'R²': round(_r2, 3),
            'Direction Accuracy': f"{_r['correct'].mean():.1%}",
            'Naive Baseline': f"{_naive:.1%}",
            'Predictions': len(_no),
            'Data range': f"{_r.index.min().date()} → {_r.index.max().date()}",
        }, {
            'Model': 'LSTM', 'Target': '7-day return',
            'R²': -1.14, 'Direction Accuracy': '50.0%',
            'Naive Baseline': f"{_naive:.1%}", 'Predictions': 74,
            'Data range': '2022–2025 only',
        }, {
            'Model': 'GRU', 'Target': '7-day return',
            'R²': -2.01, 'Direction Accuracy': '54.1%',
            'Naive Baseline': f"{_naive:.1%}", 'Predictions': 74,
            'Data range': '2022–2025 only',
        }])
        model_colors = {'XGBoost (tuned)': '#3b82f6', 'LSTM': '#56657e', 'GRU': '#56657e'}
        for _, row in summary.iterrows():
            color = model_colors.get(row['Model'], '#56657e')
            r2_val = float(row['R²'])
            r2_color = '#10b981' if r2_val > 0 else '#f43f5e'
            dir_color = '#10b981' if float(row['Direction Accuracy'].strip('%')) > float(row['Naive Baseline'].strip('%')) else '#f43f5e'
            st.markdown(f"""
            <div style="background:#171f30;border:1px solid #263354;border-left:3px solid {color};
                        border-radius:8px;padding:12px 16px;margin-bottom:8px;
                        display:flex;gap:32px;align-items:center;">
                <div style="min-width:140px;">
                    <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.8px;color:#56657e;font-weight:600;">Model</div>
                    <div style="font-size:15px;font-weight:700;color:#e8edf5;font-family:JetBrains Mono,monospace;">{row['Model']}</div>
                    <div style="font-size:11px;color:#56657e;">{row['Data range']}</div>
                </div>
                <div style="min-width:80px;">
                    <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.8px;color:#56657e;font-weight:600;">R²</div>
                    <div style="font-size:18px;font-weight:700;color:{r2_color};font-family:JetBrains Mono,monospace;">{row['R²']}</div>
                </div>
                <div style="min-width:120px;">
                    <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.8px;color:#56657e;font-weight:600;">Direction Accuracy</div>
                    <div style="font-size:18px;font-weight:700;color:{dir_color};font-family:JetBrains Mono,monospace;">{row['Direction Accuracy']}</div>
                </div>
                <div style="min-width:100px;">
                    <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.8px;color:#56657e;font-weight:600;">Naive Baseline</div>
                    <div style="font-size:18px;font-weight:700;color:#8899b4;font-family:JetBrains Mono,monospace;">{row['Naive Baseline']}</div>
                </div>
                <div style="min-width:80px;">
                    <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.8px;color:#56657e;font-weight:600;">Predictions</div>
                    <div style="font-size:18px;font-weight:700;color:#8899b4;font-family:JetBrains Mono,monospace;">{row['Predictions']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No results file found. Run model evaluation first.")

    st.markdown("---")

    # ─── XGBoost deep dive ───
    st.markdown("### XGBoost 7-Day Predictions vs Actual")

    xgb = load_results('XGB_7d_walkforward_results.csv')
    if xgb.empty:
        st.warning("No XGBoost 7-day results found.")
        return

    # KPI summary
    xgb['correct'] = ((xgb['actual'] > 0) == (xgb['predicted'] > 0)).astype(int)
    non_overlap = xgb.iloc[::7]
    r2 = 1 - ((non_overlap['actual'] - non_overlap['predicted'])**2).sum() / ((non_overlap['actual'] - non_overlap['actual'].mean())**2).sum()
    dir_acc = non_overlap['correct'].mean()

    cols = st.columns(4)
    with cols[0]:
        styled_metric("R-squared", f"{r2:.3f}", color='blue')
    with cols[1]:
        styled_metric("Direction Accuracy", f"{dir_acc:.1%}", color='emerald')
    with cols[2]:
        styled_metric("Total Predictions", f"{len(xgb):,}", color='violet')
    with cols[3]:
        styled_metric("Non-overlapping", f"{len(non_overlap)}", color='amber')

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Predictions vs actual chart
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=xgb.index, y=xgb['actual'],
        name='Actual Return', line=dict(color='#e8edf5', width=1.5),
    ))
    fig_pred.add_trace(go.Scatter(
        x=xgb.index, y=xgb['predicted'],
        name='Predicted Return', line=dict(color='#3b82f6', width=1.5),
    ))
    fig_pred.add_hline(y=0, line_dash="dash", line_color="#56657e", opacity=0.5)
    fig_pred.update_layout(height=300, yaxis_title='7-Day Return', **DARK_LAYOUT)
    st.plotly_chart(fig_pred, use_container_width=True)

    # ─── Rolling direction accuracy ───
    st.markdown("### Rolling Direction Accuracy")
    st.caption("30-day rolling window — shows how accuracy varies over time.")

    rolling_acc = xgb['correct'].rolling(30).mean()
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=rolling_acc.index, y=rolling_acc.values,
        fill='tozeroy', fillcolor='rgba(16,185,129,0.15)',
        line=dict(color='#10b981', width=1.5),
        name='Accuracy',
    ))
    fig_roll.add_hline(y=0.5, line_dash="dash", line_color="#f43f5e", opacity=0.5,
                        annotation_text="Coin flip")
    fig_roll.update_layout(height=250, yaxis_title='Accuracy', yaxis_range=[0, 1], **DARK_LAYOUT)
    st.plotly_chart(fig_roll, use_container_width=True)

    st.markdown("---")

    # ─── Market regime breakdown ───
    st.markdown("### Performance by Market Phase")

    def classify_phase(date):
        d = pd.Timestamp(date)
        if d < pd.Timestamp('2019-01-01'):
            return 'Bear Market (2018)'
        elif d < pd.Timestamp('2020-03-01'):
            return 'Recovery (2019)'
        elif d < pd.Timestamp('2020-11-01'):
            return 'COVID Crash & Recovery (2020)'
        elif d < pd.Timestamp('2022-01-01'):
            return 'Bull Run (2021)'
        elif d < pd.Timestamp('2023-01-01'):
            return 'Bear Market (2022)'
        elif d < pd.Timestamp('2024-04-15'):
            return 'Recovery + ETF Rally (2023–Apr 2024)'
        elif d < pd.Timestamp('2024-11-01'):
            return 'Consolidation (May–Oct 2024)'
        else:
            return 'Post-Election & Correction (Nov 2024–2026)'

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
    st.markdown("### Confidence vs Accuracy")
    st.markdown(
        "When the model predicts a **larger move**, it's significantly more accurate. "
        "High-confidence predictions are much more reliable."
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

    # ─── Prediction summary KPIs ─────────────────────────────────────────────
    st.markdown("### Prediction Summary")
    st.caption("Direction accuracy across all historical walk-forward predictions and live production runs. Full prediction log is on the Forecast page.")

    log = _build_prediction_log()
    if log.empty:
        st.warning("No prediction data found.")
    else:
        resolved = log[~log['pending']]
        last30   = resolved.tail(30)

        def _streak(df):
            if df.empty:
                return 0, None
            vals = df['correct'].iloc[::-1].tolist()
            first = vals[0]
            count = 0
            for v in vals:
                if v == first:
                    count += 1
                else:
                    break
            return count, bool(first)

        streak_n, streak_win = _streak(resolved)

        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            overall_acc = resolved['correct'].mean() if len(resolved) > 0 else 0
            styled_metric("All-Time Accuracy", f"{overall_acc:.1%}",
                          f"{len(resolved):,} predictions", color='blue')
        with kpi_cols[1]:
            last30_acc = last30['correct'].mean() if len(last30) > 0 else 0
            styled_metric("Last 30 Accuracy", f"{last30_acc:.1%}",
                          color='emerald' if last30_acc >= 0.5 else 'rose')
        with kpi_cols[2]:
            live_count = len(log[log['source'] == 'live'])
            live_res   = resolved[resolved['source'] == 'live']
            live_acc   = live_res['correct'].mean() if len(live_res) > 0 else 0
            styled_metric("Live Production Accuracy", f"{live_acc:.1%}",
                          f"{live_count} live predictions", color='violet')
        with kpi_cols[3]:
            if streak_n and streak_win is not None:
                streak_label = f"{'✓' if streak_win else '✗'} {streak_n} in a row"
                streak_color = 'emerald' if streak_win else 'rose'
            else:
                streak_label, streak_color = "—", 'amber'
            styled_metric("Current Streak", streak_label, color=streak_color)

