"""
Page 3 — Strategy Lab

Backtest trading strategies with adjustable leverage, stop loss, take profit.
Uses historical walk-forward predictions to simulate realistic trading outcomes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from config import RESULTS_DIR, MASTER_DF_PATH
from components import CARD_COLORS, DARK_LAYOUT, styled_metric


PRESETS = {
    'Conservative': {
        'strategy': 'Model Direction', 'confidence_filter': 'All',
        'stop_loss_pct': 5.0, 'take_profit_pct': 15.0,
        'leverage': 1.0, 'fees_pct': 0.2,
    },
    'Moderate': {
        'strategy': 'High Confidence Only', 'confidence_filter': 'HIGH',
        'stop_loss_pct': 8.0, 'take_profit_pct': 20.0,
        'leverage': 2.0, 'fees_pct': 0.2,
    },
    'Aggressive': {
        'strategy': 'Model Direction', 'confidence_filter': 'HIGH',
        'stop_loss_pct': 10.0, 'take_profit_pct': 30.0,
        'leverage': 5.0, 'fees_pct': 0.2,
    },
    'Custom': {
        'strategy': 'Model Direction', 'confidence_filter': 'All',
        'stop_loss_pct': 5.0, 'take_profit_pct': 15.0,
        'leverage': 1.0, 'fees_pct': 0.2,
    },
}

_SS_KEYS = ['strat_strategy', 'strat_confidence', 'strat_sl', 'strat_tp',
            'strat_leverage', 'strat_fees']


def _init_session_state():
    if 'strat_leverage' not in st.session_state:
        p = PRESETS['Conservative']
        st.session_state['strat_strategy']   = p['strategy']
        st.session_state['strat_confidence'] = p['confidence_filter']
        st.session_state['strat_sl']         = p['stop_loss_pct']
        st.session_state['strat_tp']         = p['take_profit_pct']
        st.session_state['strat_leverage']   = p['leverage']
        st.session_state['strat_fees']       = p['fees_pct']
        st.session_state['_strat_last_preset'] = 'Conservative'


def _sync_preset(preset: str):
    """When preset changes to a non-Custom value, push its values into session state."""
    if st.session_state.get('_strat_last_preset') != preset:
        if preset != 'Custom':
            p = PRESETS[preset]
            st.session_state['strat_strategy']   = p['strategy']
            st.session_state['strat_confidence'] = p['confidence_filter']
            st.session_state['strat_sl']         = p['stop_loss_pct']
            st.session_state['strat_tp']         = p['take_profit_pct']
            st.session_state['strat_leverage']   = p['leverage']
            st.session_state['strat_fees']       = p['fees_pct']
        st.session_state['_strat_last_preset'] = preset


@st.cache_data(ttl=23 * 3600)
def load_predictions():
    xgb_path = os.path.join(RESULTS_DIR, 'XGB_7d_walkforward_results.csv')
    if not os.path.exists(xgb_path):
        return pd.DataFrame(), pd.DataFrame()
    preds = pd.read_csv(xgb_path, index_col=0, parse_dates=True)
    prices = pd.read_csv(MASTER_DF_PATH, index_col=0, parse_dates=True,
                         usecols=['date', 'Close_BTC', 'High_BTC', 'Low_BTC'],
                         low_memory=False)
    return preds, prices


@st.cache_data(ttl=23 * 3600)
def load_rsi_sma():
    return pd.read_csv(MASTER_DF_PATH, index_col=0, parse_dates=True,
                       usecols=['date', 'RSI_Close_BTC', 'SMA_50_Close_BTC', 'Close_BTC'],
                       low_memory=False)


def simulate_strategy(preds: pd.DataFrame, prices: pd.DataFrame,
                      strategy: str, leverage: float,
                      stop_loss_pct: float, take_profit_pct: float,
                      confidence_filter: str, fees_pct: float = 0.002,
                      start_date=None) -> pd.DataFrame:
    trades = []
    initial_capital = 10000.0
    capital = initial_capital

    filter_df = load_rsi_sma()
    all_pred_dates = preds.index[::7]

    if start_date is not None:
        pred_dates = all_pred_dates[all_pred_dates >= pd.Timestamp(start_date)]
    else:
        pred_dates = all_pred_dates

    for pred_date in pred_dates:
        if capital <= 0:
            break

        predicted_return = preds.loc[pred_date, 'predicted']
        abs_pred = abs(predicted_return)

        if abs_pred > 0.05:
            confidence = 'HIGH'
        elif abs_pred > 0.02:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        take_trade = False

        if strategy == 'Model Direction':
            take_trade = predicted_return > 0

        elif strategy == 'High Confidence Only':
            take_trade = predicted_return > 0 and confidence == 'HIGH'

        elif strategy == 'Model + RSI Filter':
            if pred_date in filter_df.index:
                rsi_val = filter_df.loc[pred_date, 'RSI_Close_BTC']
                take_trade = predicted_return > 0 and (pd.isna(rsi_val) or rsi_val < 70)
            else:
                take_trade = predicted_return > 0

        elif strategy == 'Trend Following':
            if pred_date in filter_df.index:
                close_val = filter_df.loc[pred_date, 'Close_BTC']
                sma_val = filter_df.loc[pred_date, 'SMA_50_Close_BTC']
                above_sma = close_val > sma_val if pd.notna(sma_val) else True
                take_trade = predicted_return > 0 and above_sma
            else:
                take_trade = predicted_return > 0

        if confidence_filter != 'All' and confidence != confidence_filter:
            take_trade = False

        if not take_trade:
            trades.append({
                'date': pred_date, 'action': 'SKIP',
                'entry_price': None, 'exit_price': None,
                'return_pct': 0, 'leveraged_return_pct': 0, 'pnl': 0,
                'capital_after': capital, 'exit_reason': 'No signal',
                'confidence': confidence, 'predicted_return': predicted_return,
            })
            continue

        if pred_date not in prices.index:
            continue

        entry_price = prices.loc[pred_date, 'Close_BTC']
        future_dates = prices.index[prices.index > pred_date][:7]
        if len(future_dates) == 0:
            continue

        exit_price = entry_price
        exit_date = future_dates[-1]
        exit_reason = 'End of window'
        liquidated = False
        liquidation_price = entry_price * (1 - 1 / leverage) if leverage > 1 else 0

        for day in future_dates:
            if day not in prices.index:
                continue

            day_high = prices.loc[day, 'High_BTC']
            day_low  = prices.loc[day, 'Low_BTC']
            day_close = prices.loc[day, 'Close_BTC']

            if leverage > 1 and day_low <= liquidation_price:
                exit_price = liquidation_price
                exit_date = day
                exit_reason = f'LIQUIDATED at ${liquidation_price:,.0f}'
                liquidated = True
                break

            if stop_loss_pct > 0:
                sl_price = entry_price * (1 - stop_loss_pct / 100)
                if day_low <= sl_price:
                    exit_price = sl_price
                    exit_date = day
                    exit_reason = f'Stop loss at ${sl_price:,.0f}'
                    break

            if take_profit_pct > 0:
                tp_price = entry_price * (1 + take_profit_pct / 100)
                if day_high >= tp_price:
                    exit_price = tp_price
                    exit_date = day
                    exit_reason = f'Take profit at ${tp_price:,.0f}'
                    break

            exit_price = day_close
            exit_date = day

        raw_return = (exit_price / entry_price) - 1
        raw_return -= fees_pct
        leveraged_return = raw_return * leverage

        if liquidated:
            pnl = -capital
            capital = 0
        else:
            pnl = capital * leveraged_return
            capital += pnl

        trades.append({
            'date': pred_date, 'action': 'LONG',
            'exit_date': exit_date,
            'entry_price': entry_price, 'exit_price': exit_price,
            'return_pct': raw_return * 100, 'leveraged_return_pct': leveraged_return * 100,
            'pnl': pnl, 'capital_after': capital,
            'exit_reason': exit_reason, 'confidence': confidence,
            'predicted_return': predicted_return,
        })

    return pd.DataFrame(trades)


def render():
    st.markdown("<h1 style='margin-bottom:4px;'>Strategy Lab</h1>", unsafe_allow_html=True)
    st.caption("Backtest trading strategies using historical model predictions. No real money — educational only.")

    preds, prices = load_predictions()

    if preds.empty:
        st.error("No prediction results found. Run model evaluation first.")
        return

    _init_session_state()

    # ─── Top row: preset + start date ───────────────────────────────────────
    col_preset, col_date, _ = st.columns([2, 2, 1])

    with col_preset:
        preset = st.selectbox(
            "Profile",
            list(PRESETS.keys()),
            key='strategy_preset',
            help="Conservative: 1x, SL 5%, TP 15% — Moderate: 2x, SL 8%, TP 20%, high-confidence only — Aggressive: 5x, SL 10%, TP 30%",
        )

    _sync_preset(preset)

    with col_date:
        min_date = preds.index.min().date()
        max_date = (preds.index.max() - pd.Timedelta(days=30)).date()
        start_date = st.date_input(
            "Backtest start",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            help="Move this forward for a more realistic out-of-sample test. Earlier dates include the 2019–2021 bull run.",
        )

    # ─── Customize expander ─────────────────────────────────────────────────
    with st.expander("Customize"):
        st.markdown("**Strategy**")
        c1, c2 = st.columns(2)
        with c1:
            strategy = st.selectbox(
                "Strategy",
                ["Model Direction", "High Confidence Only", "Model + RSI Filter", "Trend Following"],
                key='strat_strategy',
                help="Model Direction: long when UP. High Confidence: only large predicted moves. "
                     "RSI Filter: skip overbought entries. Trend Following: only above SMA50.",
            )
        with c2:
            confidence_filter = st.selectbox(
                "Confidence Filter", ["All", "HIGH", "MEDIUM", "LOW"],
                key='strat_confidence',
                help="Only take trades matching this confidence level.",
            )

        st.markdown("**Risk Management**")
        c3, c4 = st.columns(2)
        with c3:
            stop_loss_pct = st.slider(
                "Stop Loss (%)", 0.0, 20.0, key='strat_sl', step=0.5,
                help="0 = disabled. Closes position if price drops by this %.",
            )
        with c4:
            take_profit_pct = st.slider(
                "Take Profit (%)", 0.0, 50.0, key='strat_tp', step=1.0,
                help="0 = disabled. Closes position if price rises by this %.",
            )

        st.markdown("**Execution**")
        c5, c6 = st.columns(2)
        with c5:
            leverage = st.slider(
                "Leverage", 1.0, 40.0, key='strat_leverage', step=0.5,
                help="1x = no leverage. Higher = more risk and reward.",
            )
            if leverage > 10:
                st.warning(f"At {leverage}x, a {100/leverage:.1f}% drop = liquidation.")
        with c6:
            fees_pct = st.slider(
                "Trading Fees (%)", 0.0, 1.0, key='strat_fees', step=0.05,
                help="Round-trip cost per trade (entry + exit). 0.2% is typical for spot crypto.",
            ) / 100

    # ─── Run simulation ─────────────────────────────────────────────────────
    trades = simulate_strategy(
        preds, prices, strategy, leverage,
        stop_loss_pct, take_profit_pct,
        confidence_filter, fees_pct, start_date,
    )

    if trades.empty:
        st.warning("No trades to display with the current settings.")
        return

    # ─── Honest framing callout ─────────────────────────────────────────────
    st.warning(
        "Results are optimistic: the 52 model features were chosen on the full dataset, "
        "inflating historical accuracy vs. true out-of-sample. "
        "Move the **Backtest start** date forward for a more realistic picture.",
        icon="⚠️",
    )

    # ─── Summary metrics ────────────────────────────────────────────────────
    active_trades = trades[trades['action'] == 'LONG']
    skipped       = trades[trades['action'] == 'SKIP']

    final_capital = trades['capital_after'].iloc[-1]
    total_return  = (final_capital / 10000 - 1) * 100

    equity_curve = trades['capital_after']
    peak    = equity_curve.expanding().max()
    drawdown = ((equity_curve - peak) / peak * 100)
    max_dd  = drawdown.min()

    liquidations = active_trades['exit_reason'].str.contains('LIQUIDATED').sum() if len(active_trades) > 0 else 0

    cols = st.columns(5)
    with cols[0]:
        styled_metric("Final Capital", f"${final_capital:,.0f}", f"{total_return:+.1f}%",
                      color='emerald' if total_return >= 0 else 'rose')
    with cols[1]:
        wr = f"{(active_trades['leveraged_return_pct'] > 0).mean():.0%}" if len(active_trades) > 0 else "—"
        styled_metric("Win Rate", wr, color='blue')
    with cols[2]:
        styled_metric("Trades", f"{len(active_trades)}/{len(trades)}", color='violet')
    with cols[3]:
        styled_metric("Liquidations", str(liquidations),
                      color='rose' if liquidations > 0 else 'emerald')
    with cols[4]:
        styled_metric("Max Drawdown", f"{max_dd:.1f}%", color='amber')

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ─── Tabs ───────────────────────────────────────────────────────────────
    tab_equity, tab_trades, tab_stats = st.tabs(["Equity Curve", "Trade Log", "Statistics"])

    with tab_equity:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=trades['date'], y=trades['capital_after'],
            name=f'{strategy} ({leverage}x)',
            line=dict(color='#3b82f6', width=2),
        ))

        first_price = prices.loc[trades['date'].iloc[0], 'Close_BTC']
        bh_values = []
        for _, trade in trades.iterrows():
            if trade['date'] in prices.index:
                bh_values.append(10000 * prices.loc[trade['date'], 'Close_BTC'] / first_price)
            else:
                bh_values.append(bh_values[-1] if bh_values else 10000)

        fig.add_trace(go.Scatter(
            x=trades['date'], y=bh_values,
            name='Buy & Hold', line=dict(color='#f59e0b', width=2, dash='dash'),
        ))

        liq_trades = active_trades[active_trades['exit_reason'].str.contains('LIQUIDATED')]
        if len(liq_trades) > 0:
            fig.add_trace(go.Scatter(
                x=liq_trades['date'], y=liq_trades['capital_after'],
                mode='markers', name='Liquidation',
                marker=dict(color='#f43f5e', size=12, symbol='x'),
            ))

        sl_trades = active_trades[active_trades['exit_reason'].str.contains('Stop loss')]
        if len(sl_trades) > 0:
            fig.add_trace(go.Scatter(
                x=sl_trades['date'], y=sl_trades['capital_after'],
                mode='markers', name='Stop Loss',
                marker=dict(color='#f43f5e', size=8, symbol='triangle-down'),
            ))

        tp_trades = active_trades[active_trades['exit_reason'].str.contains('Take profit')]
        if len(tp_trades) > 0:
            fig.add_trace(go.Scatter(
                x=tp_trades['date'], y=tp_trades['capital_after'],
                mode='markers', name='Take Profit',
                marker=dict(color='#10b981', size=8, symbol='triangle-up'),
            ))

        fig.update_layout(height=400, yaxis_title='Portfolio Value ($)', **DARK_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with tab_trades:
        if len(active_trades) > 0:
            rows = []
            for _, t in active_trades.iterrows():
                rows.append({
                    'Date':             t['date'].strftime('%Y-%m-%d'),
                    'Action':           'OPEN',
                    'Confidence':       t['confidence'],
                    'Price':            f"${t['entry_price']:,.0f}",
                    'Return':           '',
                    'Leveraged Return': '',
                    'P&L':              '',
                    'Capital':          '',
                    'Exit Reason':      '',
                })
                rows.append({
                    'Date':             t['exit_date'].strftime('%Y-%m-%d'),
                    'Action':           'CLOSE',
                    'Confidence':       t['confidence'],
                    'Price':            f"${t['exit_price']:,.0f}",
                    'Return':           f"{t['return_pct']:+.2f}%",
                    'Leveraged Return': f"{t['leveraged_return_pct']:+.2f}%",
                    'P&L':              f"${t['pnl']:+,.0f}",
                    'Capital':          f"${t['capital_after']:,.0f}",
                    'Exit Reason':      t['exit_reason'],
                })
            display = pd.DataFrame(rows)
            st.dataframe(display.iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.info("No trades taken with current strategy settings.")

    with tab_stats:
        if len(active_trades) > 0:
            col_stats1, col_stats2 = st.columns(2)

            with col_stats1:
                st.markdown("**Return Statistics**")
                wins   = active_trades[active_trades['leveraged_return_pct'] > 0]
                losses = active_trades[active_trades['leveraged_return_pct'] <= 0]

                stats = {
                    'Total Trades': len(active_trades),
                    'Winning':  len(wins),
                    'Losing':   len(losses),
                    'Win Rate': f"{len(wins)/len(active_trades):.0%}" if len(active_trades) > 0 else "—",
                    'Avg Win':  f"{wins['leveraged_return_pct'].mean():+.2f}%" if len(wins) > 0 else "—",
                    'Avg Loss': f"{losses['leveraged_return_pct'].mean():+.2f}%" if len(losses) > 0 else "—",
                    'Best Trade':  f"{active_trades['leveraged_return_pct'].max():+.2f}%",
                    'Worst Trade': f"{active_trades['leveraged_return_pct'].min():+.2f}%",
                }
                st.dataframe(pd.DataFrame(stats.items(), columns=['Metric', 'Value']),
                             use_container_width=True, hide_index=True)

            with col_stats2:
                st.markdown("**Risk Metrics**")
                returns_series = active_trades['leveraged_return_pct'] / 100
                sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(52) if returns_series.std() > 0 else 0

                gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
                gross_loss   = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

                risk_stats = {
                    'Sharpe Ratio':    f"{sharpe:.2f}",
                    'Profit Factor':   f"{profit_factor:.2f}",
                    'Max Drawdown':    f"{max_dd:.1f}%",
                    'Liquidations':    liquidations,
                    'Stop Losses Hit': len(sl_trades) if len(active_trades) > 0 else 0,
                    'Take Profits Hit': len(tp_trades) if len(active_trades) > 0 else 0,
                    'Trades Skipped':  len(skipped),
                }
                st.dataframe(pd.DataFrame(risk_stats.items(), columns=['Metric', 'Value']),
                             use_container_width=True, hide_index=True)

            st.markdown("**Return Distribution**")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=active_trades['leveraged_return_pct'],
                nbinsx=30, marker_color='#3b82f6', opacity=0.7,
                name='Trade Returns',
            ))
            fig_hist.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
            fig_hist.update_layout(
                height=280, xaxis_title='Return (%)', yaxis_title='Count',
                **DARK_LAYOUT,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    date_from = preds.index.min().strftime('%b %Y')
    date_to   = preds.index.max().strftime('%b %Y')
    st.info(
        f"Backtest period: **{start_date} → {date_to}** (full data: {date_from} → {date_to}) — "
        f"walk-forward predictions only (model trained on past data at each point). "
        f"Fees: **{fees_pct*100:.2f}% per trade**."
    )
