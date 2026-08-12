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

# Two walk-forward result files, and the difference between them is the point.
#
# LEAKY_WF is the original evaluation. It trained on df.iloc[:i] to predict row i,
# but Target_Return_7d is close.shift(-7)/close - 1, so rows i-6..i-1 carried
# targets computed from prices on days i..i+5. The model was trained on the answer.
#
# LEAKFREE_WF re-runs the identical evaluation with those 7 rows purged
# (scripts/evaluate_leakfree.py). It is the number that matches reality: 48.5%
# against a 52.6% baseline, which the live production log independently confirms.
#
# Everything on this page is computed from LEAKFREE_WF. LEAKY_WF appears only in
# the comparison table, labelled as invalid, because deleting the mistake would
# hide it.
LEAKY_WF    = 'XGB_7d_walkforward_results.csv'
LEAKFREE_WF = 'XGB_7d_walkforward_leakfree.csv'


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

    # ── Walk-forward history (leak-free — see the note at the top of the file) ──
    wf_path = os.path.join(RESULTS_DIR, LEAKFREE_WF)
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
        # date_format='mixed' — see the note in predict.load_trade_log()
        tl = pd.read_csv(TRADE_LOG_PATH, parse_dates=['prediction_date', 'target_date'],
                         date_format='mixed')
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


def _metrics(df: pd.DataFrame) -> tuple:
    """Direction accuracy and R² over a set of non-overlapping windows."""
    correct = ((df['actual'] > 0) == (df['predicted'] > 0)).mean()
    ss_res = ((df['actual'] - df['predicted']) ** 2).sum()
    ss_tot = ((df['actual'] - df['actual'].mean()) ** 2).sum()
    return correct, 1 - ss_res / ss_tot


def render():
    st.markdown("<h1 style='margin-bottom:4px;'>Model Performance</h1>", unsafe_allow_html=True)
    st.caption("Walk-forward validation results, and the audit that corrected them.")

    st.error(
        "**This model has no demonstrated edge.** The original evaluation reported "
        "76.7% direction accuracy. It was wrong twice: overlapping windows were "
        "counted as independent, and the walk-forward loop trained on rows whose "
        "7-day targets were built from prices *after* the prediction date. Removing "
        "the leak takes accuracy to **48.5%** and the correlation between prediction "
        "and outcome from 0.77 to −0.03. The live log agrees. Every number below is "
        "computed leak-free; the original is kept, labelled, for comparison."
    )

    # ─── Model comparison table ───
    st.markdown("### Model Comparison")
    st.markdown(
        "All figures on non-overlapping 7-day windows. Reproduce with "
        "`scripts/evaluate_leakfree.py`."
    )

    leakfree_path = os.path.join(RESULTS_DIR, LEAKFREE_WF)
    leaky_path = os.path.join(RESULTS_DIR, LEAKY_WF)

    if os.path.exists(leakfree_path):
        _lf = pd.read_csv(leakfree_path, index_col=0, parse_dates=True)
        _lf_acc, _lf_r2 = _metrics(_lf)
        _naive = (_lf['actual'] > 0).mean()

        rows = [{
            'Model': 'XGBoost (leak-free)', 'Target': '7-day return',
            'R²': round(_lf_r2, 3),
            'Direction Accuracy': f"{_lf_acc:.1%}",
            'Naive Baseline': f"{_naive:.1%}",
            'Predictions': len(_lf),
            'Data range': f"{_lf.index.min().date()} → {_lf.index.max().date()}",
        }]

        # The original, kept visible on purpose. Hiding a retracted number is just
        # a quieter version of the mistake that produced it.
        if os.path.exists(leaky_path):
            _lk = pd.read_csv(leaky_path, index_col=0, parse_dates=True)
            _lk_no = _lk.iloc[::7]
            _lk_acc, _lk_r2 = _metrics(_lk_no)
            rows.append({
                'Model': 'XGBoost (original, INVALID)', 'Target': '7-day return',
                'R²': round(_lk_r2, 3),
                'Direction Accuracy': f"{_lk_acc:.1%}",
                'Naive Baseline': f"{_naive:.1%}",
                'Predictions': len(_lk_no),
                'Data range': 'target leakage — retracted',
            })

        # Deep-learning figures are static: they come from evaluate_dl.py, which
        # already purged the 7 peeking rows. That is exactly why the original
        # comparison was unfair — the networks ran the honest race, XGBoost did not.
        rows += [{
            'Model': 'LSTM', 'Target': '7-day return',
            'R²': -1.14, 'Direction Accuracy': '50.0%',
            'Naive Baseline': f"{_naive:.1%}", 'Predictions': 74,
            'Data range': '2022–2025, leak-free',
        }, {
            'Model': 'GRU', 'Target': '7-day return',
            'R²': -2.01, 'Direction Accuracy': '54.1%',
            'Naive Baseline': f"{_naive:.1%}", 'Predictions': 74,
            'Data range': '2022–2025, leak-free',
        }]
        summary = pd.DataFrame(rows)
        model_colors = {
            'XGBoost (leak-free)': '#3b82f6',
            'XGBoost (original, INVALID)': '#f43f5e',
        }
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

    # ─── XGBoost deep dive (leak-free) ───
    st.markdown("### XGBoost 7-Day Predictions vs Actual (leak-free)")

    xgb = load_results(LEAKFREE_WF)
    if xgb.empty:
        st.warning("No leak-free results found. Run `scripts/evaluate_leakfree.py`.")
        return

    # The leak-free file is already one row per non-overlapping window, so there
    # is no iloc[::7] step here and no way to mix window counts with daily counts.
    xgb['correct'] = ((xgb['actual'] > 0) == (xgb['predicted'] > 0)).astype(int)
    dir_acc, r2 = _metrics(xgb)
    naive_acc = (xgb['actual'] > 0).mean()

    cols = st.columns(4)
    with cols[0]:
        styled_metric("R-squared", f"{r2:.3f}", color='rose' if r2 < 0 else 'blue')
    with cols[1]:
        styled_metric("Direction Accuracy", f"{dir_acc:.1%}",
                      color='emerald' if dir_acc > naive_acc else 'rose')
    with cols[2]:
        styled_metric("Naive Baseline", f"{naive_acc:.1%}", color='violet')
    with cols[3]:
        styled_metric("Independent Windows", f"{len(xgb)}", color='amber')

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
    st.caption("26-window rolling average (~6 months) — each point is one independent "
               "7-day window, so this tracks the coin-flip line rather than sitting above it.")

    rolling_acc = xgb['correct'].rolling(26).mean()
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
        # "Avg Weekly BTC Return" describes the market in that phase, not the model.
        # It was previously labelled "Avg Return", which read as model performance.
        # R² is the real thing, not squared correlation — squaring hides the sign,
        # so a model fitting worse than the mean would still show a tidy positive number.
        acc, r2 = _metrics(data)
        phase_stats.append({
            'Phase': phase,
            'Windows': len(data),
            'Avg Weekly BTC Return': f"{data['actual'].mean():.2%}",
            'Direction Accuracy': f"{acc:.1%}",
            'R²': f"{r2:.3f}",
        })

    st.dataframe(pd.DataFrame(phase_stats), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ─── Confidence vs accuracy ───
    st.markdown("### Confidence vs Accuracy")
    st.markdown(
        "The original evaluation showed accuracy climbing to ~90% on the largest "
        "predicted moves, and that became the app's headline claim. It was the leak: "
        "leak-free, the same bucket sits at ~44%, below the smallest-move bucket. "
        "**Predicted move size carries no information about whether the call is right.** "
        "The live log shows the same thing."
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

        # Historical baseline: leak-free walk-forward, already one row per
        # independent window (no iloc[::7] needed).
        _wf_acc_val, _wf_n_val = None, 0
        _wf_path = os.path.join(RESULTS_DIR, LEAKFREE_WF)
        if os.path.exists(_wf_path):
            _wf = pd.read_csv(_wf_path, index_col=0, parse_dates=True)
            _wf_acc_val = ((_wf['actual'] > 0) == (_wf['predicted'] > 0)).mean()
            _wf_n_val   = len(_wf)

        kpi_cols = st.columns(5)
        with kpi_cols[0]:
            if _wf_acc_val is not None:
                styled_metric("Backtest (leak-free)", f"{_wf_acc_val:.1%}",
                              f"{_wf_n_val:,} independent windows", color='blue')
            else:
                styled_metric("Backtest (leak-free)", "—", color='blue')
        with kpi_cols[1]:
            overall_acc = resolved['correct'].mean() if len(resolved) > 0 else 0
            styled_metric("All-Time Accuracy", f"{overall_acc:.1%}",
                          f"{len(resolved):,} predictions", color='blue')
        with kpi_cols[2]:
            last30_acc = last30['correct'].mean() if len(last30) > 0 else 0
            styled_metric("Last 30 Accuracy", f"{last30_acc:.1%}",
                          color='emerald' if last30_acc >= 0.5 else 'rose')
        with kpi_cols[3]:
            live_count = len(log[log['source'] == 'live'])
            live_res   = resolved[resolved['source'] == 'live']
            live_acc   = live_res['correct'].mean() if len(live_res) > 0 else 0
            styled_metric("Live Production Accuracy", f"{live_acc:.1%}",
                          f"{live_count} live predictions", color='violet')
        with kpi_cols[4]:
            if streak_n and streak_win is not None:
                streak_label = f"{'✓' if streak_win else '✗'} {streak_n} in a row"
                streak_color = 'emerald' if streak_win else 'rose'
            else:
                streak_label, streak_color = "—", 'amber'
            styled_metric("Current Streak", streak_label, color=streak_color)

