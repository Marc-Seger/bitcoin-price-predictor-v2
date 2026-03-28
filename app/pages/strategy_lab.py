"""
Page 3 — Strategy Lab

Backtest trading strategies with adjustable leverage, stop loss, take profit.
Uses historical walk-forward predictions to simulate realistic trading outcomes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from config import RESULTS_DIR, MASTER_DF_PATH


@st.cache_data
def load_predictions():
    """Load XGBoost 7-day walk-forward predictions and BTC prices."""
    xgb_path = os.path.join(RESULTS_DIR, 'XGB_7d_walkforward_results.csv')
    if not os.path.exists(xgb_path):
        return pd.DataFrame(), pd.DataFrame()

    preds = pd.read_csv(xgb_path, index_col=0, parse_dates=True)

    prices = pd.read_csv(MASTER_DF_PATH, index_col=0, parse_dates=True,
                         usecols=['date', 'Close_BTC', 'High_BTC', 'Low_BTC'],
                         low_memory=False)
    return preds, prices


def simulate_strategy(preds: pd.DataFrame, prices: pd.DataFrame,
                      strategy: str, leverage: float,
                      stop_loss_pct: float, take_profit_pct: float,
                      confidence_filter: str, rsi_filter: bool) -> pd.DataFrame:
    """
    Simulate a trading strategy on historical predictions.

    For each 7-day prediction window:
    1. Check if the strategy signals a trade
    2. If yes, open a position with given leverage
    3. Check daily within the 7-day window for SL/TP/liquidation
    4. Close at end of window if SL/TP not hit

    Returns a DataFrame with one row per trade.
    """
    trades = []
    initial_capital = 10000.0
    capital = initial_capital
    peak_capital = initial_capital

    # Non-overlapping: trade every 7th prediction
    pred_dates = preds.index[::7]

    for pred_date in pred_dates:
        if capital <= 0:
            break

        predicted_return = preds.loc[pred_date, 'predicted']
        abs_pred = abs(predicted_return)

        # Confidence classification
        if abs_pred > 0.05:
            confidence = 'HIGH'
        elif abs_pred > 0.02:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        # ─── Strategy filters ───
        take_trade = False

        if strategy == 'Model Direction':
            take_trade = predicted_return > 0

        elif strategy == 'High Confidence Only':
            take_trade = predicted_return > 0 and confidence == 'HIGH'

        elif strategy == 'Model + RSI Filter':
            # Need RSI data
            rsi_col = 'RSI_Close_BTC'
            rsi_df = pd.read_csv(MASTER_DF_PATH, index_col=0, parse_dates=True,
                                  usecols=['date', rsi_col], low_memory=False)
            if pred_date in rsi_df.index:
                rsi_val = rsi_df.loc[pred_date, rsi_col]
                take_trade = predicted_return > 0 and rsi_val < 70
            else:
                take_trade = predicted_return > 0

        elif strategy == 'Trend Following':
            sma_col = 'SMA_50_Close_BTC'
            close_col = 'Close_BTC'
            trend_df = pd.read_csv(MASTER_DF_PATH, index_col=0, parse_dates=True,
                                    usecols=['date', sma_col, close_col], low_memory=False)
            if pred_date in trend_df.index:
                above_sma = trend_df.loc[pred_date, close_col] > trend_df.loc[pred_date, sma_col]
                take_trade = predicted_return > 0 and above_sma
            else:
                take_trade = predicted_return > 0

        # Confidence filter (applies on top of strategy)
        if confidence_filter != 'All' and confidence != confidence_filter:
            take_trade = False

        if not take_trade:
            trades.append({
                'date': pred_date,
                'action': 'SKIP',
                'entry_price': None,
                'exit_price': None,
                'return_pct': 0,
                'leveraged_return_pct': 0,
                'pnl': 0,
                'capital_after': capital,
                'exit_reason': 'No signal',
                'confidence': confidence,
                'predicted_return': predicted_return,
            })
            continue

        # ─── Execute trade ───
        if pred_date not in prices.index:
            continue

        entry_price = prices.loc[pred_date, 'Close_BTC']

        # Simulate daily price action over the next 7 days
        future_dates = prices.index[prices.index > pred_date][:7]
        if len(future_dates) == 0:
            continue

        exit_price = entry_price
        exit_reason = 'End of window'
        liquidated = False

        # Liquidation price: price at which leveraged loss = 100% of capital
        # For a long: entry * (1 - 1/leverage)
        liquidation_price = entry_price * (1 - 1 / leverage) if leverage > 1 else 0

        for day in future_dates:
            if day not in prices.index:
                continue

            day_high = prices.loc[day, 'High_BTC']
            day_low = prices.loc[day, 'Low_BTC']
            day_close = prices.loc[day, 'Close_BTC']

            # Check liquidation (low touches liquidation price)
            if leverage > 1 and day_low <= liquidation_price:
                exit_price = liquidation_price
                exit_reason = f'LIQUIDATED at ${liquidation_price:,.0f}'
                liquidated = True
                break

            # Check stop loss
            if stop_loss_pct > 0:
                sl_price = entry_price * (1 - stop_loss_pct / 100)
                if day_low <= sl_price:
                    exit_price = sl_price
                    exit_reason = f'Stop loss at ${sl_price:,.0f}'
                    break

            # Check take profit
            if take_profit_pct > 0:
                tp_price = entry_price * (1 + take_profit_pct / 100)
                if day_high >= tp_price:
                    exit_price = tp_price
                    exit_reason = f'Take profit at ${tp_price:,.0f}'
                    break

            exit_price = day_close

        # Calculate returns
        raw_return = (exit_price / entry_price) - 1
        leveraged_return = raw_return * leverage

        if liquidated:
            pnl = -capital  # lose everything
            capital = 0
        else:
            pnl = capital * leveraged_return
            capital += pnl

        peak_capital = max(peak_capital, capital)

        trades.append({
            'date': pred_date,
            'action': 'LONG',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return_pct': raw_return * 100,
            'leveraged_return_pct': leveraged_return * 100,
            'pnl': pnl,
            'capital_after': capital,
            'exit_reason': exit_reason,
            'confidence': confidence,
            'predicted_return': predicted_return,
        })

    return pd.DataFrame(trades)


def render():
    st.title("Strategy Lab")
    st.caption("Backtest trading strategies using historical model predictions. No real money — educational only.")

    preds, prices = load_predictions()

    if preds.empty:
        st.error("No prediction results found. Run model evaluation first.")
        return

    # ─── Strategy controls ───
    st.sidebar.markdown("### Strategy Settings")

    strategy = st.sidebar.selectbox(
        "Strategy",
        ["Model Direction", "High Confidence Only", "Model + RSI Filter", "Trend Following"],
        help="Model Direction: long when UP. High Confidence: only large predicted moves. "
             "RSI Filter: skip overbought entries. Trend Following: only above SMA50."
    )

    confidence_filter = st.sidebar.selectbox(
        "Confidence Filter", ["All", "HIGH", "MEDIUM", "LOW"],
        help="Only take trades when model confidence matches."
    )

    leverage = st.sidebar.slider("Leverage", 1.0, 40.0, 1.0, 0.5,
                                  help="1x = no leverage. Higher = more risk and reward. "
                                       "At 40x, a 2.5% move against you = liquidation.")

    stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 0.0, 20.0, 0.0, 0.5,
                                       help="0 = disabled. Closes position if price drops by this %.")

    take_profit_pct = st.sidebar.slider("Take Profit (%)", 0.0, 50.0, 0.0, 1.0,
                                         help="0 = disabled. Closes position if price rises by this %.")

    if leverage > 10:
        st.sidebar.warning(f"At {leverage}x leverage, a {100/leverage:.1f}% drop = liquidation.")

    # ─── Run simulation ───
    trades = simulate_strategy(preds, prices, strategy, leverage,
                                stop_loss_pct, take_profit_pct,
                                confidence_filter, False)

    if trades.empty:
        st.warning("No trades to display.")
        return

    # ─── Summary metrics ───
    active_trades = trades[trades['action'] == 'LONG']
    skipped = trades[trades['action'] == 'SKIP']

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        final_capital = trades['capital_after'].iloc[-1]
        total_return = (final_capital / 10000 - 1) * 100
        st.metric("Final Capital", f"${final_capital:,.0f}", f"{total_return:+.1f}%")

    with col2:
        if len(active_trades) > 0:
            win_rate = (active_trades['leveraged_return_pct'] > 0).mean()
            st.metric("Win Rate", f"{win_rate:.0%}")
        else:
            st.metric("Win Rate", "—")

    with col3:
        st.metric("Trades Taken", f"{len(active_trades)}/{len(trades)}")

    with col4:
        liquidations = active_trades['exit_reason'].str.contains('LIQUIDATED').sum() if len(active_trades) > 0 else 0
        st.metric("Liquidations", liquidations)

    with col5:
        # Max drawdown
        equity_curve = trades['capital_after']
        peak = equity_curve.expanding().max()
        drawdown = ((equity_curve - peak) / peak * 100)
        max_dd = drawdown.min()
        st.metric("Max Drawdown", f"{max_dd:.1f}%")

    st.markdown("---")

    # ─── Tabs for detailed analysis ───
    tab_equity, tab_trades, tab_stats = st.tabs(["Equity Curve", "Trade Log", "Statistics"])

    with tab_equity:
        # Equity curve vs buy & hold
        fig = go.Figure()

        # Strategy equity
        fig.add_trace(go.Scatter(
            x=trades['date'], y=trades['capital_after'],
            name=f'{strategy} ({leverage}x)',
            line=dict(color='#2196f3', width=2),
        ))

        # Buy & hold comparison
        first_price = prices.loc[trades['date'].iloc[0], 'Close_BTC']
        bh_values = []
        for _, trade in trades.iterrows():
            if trade['date'] in prices.index:
                current_price = prices.loc[trade['date'], 'Close_BTC']
                bh_values.append(10000 * current_price / first_price)
            else:
                bh_values.append(bh_values[-1] if bh_values else 10000)

        fig.add_trace(go.Scatter(
            x=trades['date'], y=bh_values,
            name='Buy & Hold', line=dict(color='#ff9800', width=2, dash='dash'),
        ))

        # Mark liquidations
        liq_trades = active_trades[active_trades['exit_reason'].str.contains('LIQUIDATED')]
        if len(liq_trades) > 0:
            fig.add_trace(go.Scatter(
                x=liq_trades['date'], y=liq_trades['capital_after'],
                mode='markers', name='Liquidation',
                marker=dict(color='red', size=12, symbol='x'),
            ))

        # Mark SL/TP hits
        sl_trades = active_trades[active_trades['exit_reason'].str.contains('Stop loss')]
        if len(sl_trades) > 0:
            fig.add_trace(go.Scatter(
                x=sl_trades['date'], y=sl_trades['capital_after'],
                mode='markers', name='Stop Loss',
                marker=dict(color='#ef5350', size=8, symbol='triangle-down'),
            ))

        tp_trades = active_trades[active_trades['exit_reason'].str.contains('Take profit')]
        if len(tp_trades) > 0:
            fig.add_trace(go.Scatter(
                x=tp_trades['date'], y=tp_trades['capital_after'],
                mode='markers', name='Take Profit',
                marker=dict(color='#26a69a', size=8, symbol='triangle-up'),
            ))

        fig.update_layout(
            height=500, template='plotly_dark',
            yaxis_title='Portfolio Value ($)',
            legend=dict(orientation='h', y=1.05),
            margin=dict(l=50, r=20, t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_trades:
        if len(active_trades) > 0:
            display = active_trades[[
                'date', 'confidence', 'entry_price', 'exit_price',
                'return_pct', 'leveraged_return_pct', 'pnl', 'capital_after', 'exit_reason',
            ]].copy()
            display['entry_price'] = display['entry_price'].apply(lambda x: f"${x:,.0f}")
            display['exit_price'] = display['exit_price'].apply(lambda x: f"${x:,.0f}")
            display['return_pct'] = display['return_pct'].apply(lambda x: f"{x:+.2f}%")
            display['leveraged_return_pct'] = display['leveraged_return_pct'].apply(lambda x: f"{x:+.2f}%")
            display['pnl'] = display['pnl'].apply(lambda x: f"${x:+,.0f}")
            display['capital_after'] = display['capital_after'].apply(lambda x: f"${x:,.0f}")

            display.columns = ['Date', 'Confidence', 'Entry', 'Exit', 'Return',
                               'Leveraged Return', 'P&L', 'Capital', 'Exit Reason']
            st.dataframe(display.iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.info("No trades taken with current strategy settings.")

    with tab_stats:
        if len(active_trades) > 0:
            col_stats1, col_stats2 = st.columns(2)

            with col_stats1:
                st.markdown("**Return Statistics**")
                wins = active_trades[active_trades['leveraged_return_pct'] > 0]
                losses = active_trades[active_trades['leveraged_return_pct'] <= 0]

                stats = {
                    'Total Trades': len(active_trades),
                    'Winning Trades': len(wins),
                    'Losing Trades': len(losses),
                    'Win Rate': f"{len(wins)/len(active_trades):.0%}" if len(active_trades) > 0 else "—",
                    'Avg Win': f"{wins['leveraged_return_pct'].mean():+.2f}%" if len(wins) > 0 else "—",
                    'Avg Loss': f"{losses['leveraged_return_pct'].mean():+.2f}%" if len(losses) > 0 else "—",
                    'Best Trade': f"{active_trades['leveraged_return_pct'].max():+.2f}%",
                    'Worst Trade': f"{active_trades['leveraged_return_pct'].min():+.2f}%",
                    'Avg Trade': f"{active_trades['leveraged_return_pct'].mean():+.2f}%",
                }
                st.dataframe(pd.DataFrame(stats.items(), columns=['Metric', 'Value']),
                             use_container_width=True, hide_index=True)

            with col_stats2:
                st.markdown("**Risk Metrics**")
                returns_series = active_trades['leveraged_return_pct'] / 100

                # Sharpe ratio (annualized, assuming ~52 trades per year with weekly)
                if returns_series.std() > 0:
                    sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(52)
                else:
                    sharpe = 0

                # Profit factor
                gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
                gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

                risk_stats = {
                    'Sharpe Ratio': f"{sharpe:.2f}",
                    'Profit Factor': f"{profit_factor:.2f}",
                    'Max Drawdown': f"{max_dd:.1f}%",
                    'Liquidations': liquidations,
                    'Stop Losses Hit': len(sl_trades) if len(active_trades) > 0 else 0,
                    'Take Profits Hit': len(tp_trades) if len(active_trades) > 0 else 0,
                    'Trades Skipped': len(skipped),
                }
                st.dataframe(pd.DataFrame(risk_stats.items(), columns=['Metric', 'Value']),
                             use_container_width=True, hide_index=True)

            # Return distribution
            st.markdown("**Return Distribution**")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=active_trades['leveraged_return_pct'],
                nbinsx=30, marker_color='#2196f3', opacity=0.7,
                name='Trade Returns',
            ))
            fig_hist.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
            fig_hist.update_layout(
                height=300, template='plotly_dark',
                xaxis_title='Return (%)', yaxis_title='Count',
                margin=dict(l=50, r=20, t=10, b=30),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
