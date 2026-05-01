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
        'min_magnitude': 0.0,
        'rsi_enabled': False, 'rsi_max': 70.0,
        'trend_enabled': False, 'sma_period': 50,
        'macd_enabled': False,
        'volume_enabled': False,
        'stop_loss_pct': 5.0, 'take_profit_pct': 15.0,
        'leverage': 1.0, 'fees_pct': 0.2, 'slippage_pct': 0.1, 'funding_rate': 0.0,
        'reentry': False,
    },
    'Moderate': {
        'min_magnitude': 5.0,
        'rsi_enabled': True, 'rsi_max': 70.0,
        'trend_enabled': False, 'sma_period': 50,
        'macd_enabled': False,
        'volume_enabled': False,
        'stop_loss_pct': 8.0, 'take_profit_pct': 20.0,
        'leverage': 2.0, 'fees_pct': 0.2, 'slippage_pct': 0.1, 'funding_rate': 0.03,
        'reentry': False,
    },
    'Aggressive': {
        'min_magnitude': 5.0,
        'rsi_enabled': True, 'rsi_max': 70.0,
        'trend_enabled': True, 'sma_period': 50,
        'macd_enabled': True,
        'volume_enabled': False,
        'stop_loss_pct': 10.0, 'take_profit_pct': 30.0,
        'leverage': 5.0, 'fees_pct': 0.2, 'slippage_pct': 0.1, 'funding_rate': 0.03,
        'reentry': False,
    },
    'Custom': {
        'min_magnitude': 0.0,
        'rsi_enabled': False, 'rsi_max': 70.0,
        'trend_enabled': False, 'sma_period': 50,
        'macd_enabled': False,
        'volume_enabled': False,
        'stop_loss_pct': 5.0, 'take_profit_pct': 15.0,
        'leverage': 1.0, 'fees_pct': 0.2, 'slippage_pct': 0.1, 'funding_rate': 0.0,
        'reentry': False,
    },
}

PROFILE_DESCRIPTIONS = {
    'Conservative': (
        'Enters a long whenever the model predicts any positive 7-day return. '
        'No extra filters — the purest test of the model\'s directional edge.\n\n'
        '- **Leverage:** 1× (no liquidation risk)\n'
        '- **Take profit:** 15%\n'
        '- **Stop loss:** 5%\n'
        '- **Slippage:** 0.1% applied on fills\n'
        '- **Funding fees:** N/A (spot position)\n\n'
        'If TP/SL isn\'t triggered, the trade closes automatically at day 7.'
    ),
    'Moderate': (
        'Enters when the model predicts a return above **5%** and RSI is below **70** '
        '(avoids buying into overbought conditions).\n\n'
        '- **Leverage:** 2×\n'
        '- **Take profit:** 20%\n'
        '- **Stop loss:** 8%\n'
        '- **Slippage:** 0.1% applied on fills\n'
        '- **Funding fees:** 0.03%/day on the leveraged position\n\n'
        'If TP/SL isn\'t triggered, the trade closes automatically at day 7.'
    ),
    'Aggressive': (
        'Enters when the model predicts >5%, RSI is below 70, price is above the 50-day SMA '
        '(uptrend confirmed), and MACD histogram is positive (momentum aligns).\n\n'
        '- **Leverage:** 5× (a 20% adverse move = full liquidation)\n'
        '- **Take profit:** 30%\n'
        '- **Stop loss:** 10%\n'
        '- **Slippage:** 0.1% applied on fills\n'
        '- **Funding fees:** 0.03%/day on the leveraged position\n\n'
        'Full-history results are dominated by the 2019–2024 bull run — use a recent start date for a realistic picture.'
    ),
}


def _get_confidence(predicted_return: float) -> str:
    abs_pred = abs(predicted_return)
    if abs_pred > 0.05:
        return 'HIGH'
    elif abs_pred > 0.02:
        return 'MEDIUM'
    return 'LOW'


def _init_session_state():
    if 'strat_leverage' not in st.session_state:
        _push_preset(PRESETS['Conservative'])
        st.session_state['_strat_last_preset'] = 'Conservative'


def _push_preset(p: dict):
    st.session_state['strat_min_magnitude']  = p['min_magnitude']
    st.session_state['strat_rsi_enabled']    = p['rsi_enabled']
    st.session_state['strat_rsi_max']        = p['rsi_max']
    st.session_state['strat_trend_enabled']  = p['trend_enabled']
    st.session_state['strat_sma_period']     = p['sma_period']
    st.session_state['strat_macd_enabled']   = p['macd_enabled']
    st.session_state['strat_volume_enabled'] = p['volume_enabled']
    st.session_state['strat_sl']             = p['stop_loss_pct']
    st.session_state['strat_tp']             = p['take_profit_pct']
    st.session_state['strat_leverage']       = p['leverage']
    st.session_state['strat_fees']           = p['fees_pct']
    st.session_state['strat_slippage']       = p['slippage_pct']
    st.session_state['strat_funding']        = p['funding_rate']
    st.session_state['strat_reentry']        = p['reentry']


def _sync_preset(preset: str):
    if st.session_state.get('_strat_last_preset') != preset:
        if preset != 'Custom':
            _push_preset(PRESETS[preset])
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
def load_filter_data():
    wanted = ['date', 'Close_BTC', 'RSI_Close_BTC', 'SMA_50_Close_BTC',
              'SMA_200_Close_BTC', 'MACD_Histogram_D_BTC', 'Volume_Percentile_BTC']
    try:
        df = pd.read_csv(MASTER_DF_PATH, index_col=0, parse_dates=True,
                         usecols=wanted, low_memory=False)
    except ValueError:
        # SMA_200_Close_BTC missing — load without it and compute
        fallback = [c for c in wanted if c != 'SMA_200_Close_BTC']
        df = pd.read_csv(MASTER_DF_PATH, index_col=0, parse_dates=True,
                         usecols=fallback, low_memory=False)
        df['SMA_200_Close_BTC'] = df['Close_BTC'].rolling(200).mean()
    return df


def _eval_filters(pred_date, predicted_return: float, filter_df: pd.DataFrame,
                  filters: dict) -> bool:
    """Returns True only if all active filters pass."""
    if predicted_return <= filters['min_magnitude'] / 100:
        return False

    if pred_date not in filter_df.index:
        return True

    row = filter_df.loc[pred_date]

    if filters['rsi_enabled']:
        rsi = row['RSI_Close_BTC']
        if pd.notna(rsi) and rsi > filters['rsi_max']:
            return False

    if filters['trend_enabled']:
        sma_col = 'SMA_50_Close_BTC' if filters['sma_period'] == 50 else 'SMA_200_Close_BTC'
        sma   = row.get(sma_col, np.nan)
        close = row['Close_BTC']
        if pd.notna(sma) and pd.notna(close) and close <= sma:
            return False

    if filters['macd_enabled']:
        macd = row['MACD_Histogram_D_BTC']
        if pd.notna(macd) and macd <= 0:
            return False

    if filters['volume_enabled']:
        vol_pct = row['Volume_Percentile_BTC']
        if pd.notna(vol_pct) and vol_pct < 0.5:
            return False

    return True


def _skip_record(pred_date, predicted_return: float, capital: float) -> dict:
    return {
        'date': pred_date, 'action': 'SKIP',
        'entry_price': None, 'exit_price': None,
        'return_pct': 0.0, 'leveraged_return_pct': 0.0, 'pnl': 0.0,
        'capital_after': capital, 'exit_reason': 'No signal',
        'confidence': _get_confidence(predicted_return),
        'predicted_return': predicted_return,
    }


def _execute_trade_record(pred_date, predicted_return: float, prices: pd.DataFrame,
                          leverage: float, stop_loss_pct: float, take_profit_pct: float,
                          fees_pct: float, slippage_pct: float, funding_rate_daily: float,
                          capital: float):
    """Execute a single trade from pred_date. Returns (trade_dict, exit_date)."""
    future_dates = prices.index[prices.index > pred_date][:7]
    if len(future_dates) == 0:
        return _skip_record(pred_date, predicted_return, capital), None

    entry_price  = prices.loc[pred_date, 'Close_BTC']
    actual_entry = entry_price * (1 + slippage_pct)

    liq_trigger = actual_entry * (1 - 1 / leverage) if leverage > 1 else None
    sl_trigger  = actual_entry * (1 - stop_loss_pct / 100) if stop_loss_pct > 0 else None
    tp_trigger  = actual_entry * (1 + take_profit_pct / 100) if take_profit_pct > 0 else None

    actual_exit = actual_entry
    exit_date   = future_dates[-1]
    exit_reason = 'End of window'
    liquidated  = False

    for day in future_dates:
        if day not in prices.index:
            continue
        day_high  = prices.loc[day, 'High_BTC']
        day_low   = prices.loc[day, 'Low_BTC']
        day_close = prices.loc[day, 'Close_BTC']

        if liq_trigger and day_low <= liq_trigger:
            actual_exit = liq_trigger * (1 - slippage_pct)
            exit_date   = day
            exit_reason = f'LIQUIDATED at ${liq_trigger:,.0f}'
            liquidated  = True
            break
        if sl_trigger and day_low <= sl_trigger:
            actual_exit = sl_trigger * (1 - slippage_pct)
            exit_date   = day
            exit_reason = f'Stop loss at ${sl_trigger:,.0f}'
            break
        if tp_trigger and day_high >= tp_trigger:
            actual_exit = tp_trigger * (1 - slippage_pct)
            exit_date   = day
            exit_reason = f'Take profit at ${tp_trigger:,.0f}'
            break
        actual_exit = day_close * (1 - slippage_pct)
        exit_date   = day

    raw_return = (actual_exit / actual_entry) - 1 - fees_pct
    leveraged_return = raw_return * leverage

    if leverage > 1 and funding_rate_daily > 0:
        days_held = max((exit_date - pred_date).days, 1)
        leveraged_return -= leverage * funding_rate_daily * days_held

    if liquidated:
        pnl         = -capital
        new_capital = 0.0
    else:
        pnl         = capital * leveraged_return
        new_capital = capital + pnl

    return {
        'date': pred_date, 'action': 'LONG',
        'exit_date': exit_date,
        'entry_price': actual_entry, 'exit_price': actual_exit,
        'return_pct': raw_return * 100,
        'leveraged_return_pct': leveraged_return * 100,
        'pnl': pnl, 'capital_after': new_capital,
        'exit_reason': exit_reason,
        'confidence': _get_confidence(predicted_return),
        'predicted_return': predicted_return,
    }, exit_date


def simulate_strategy(preds: pd.DataFrame, prices: pd.DataFrame,
                      filter_df: pd.DataFrame, filters: dict,
                      leverage: float, stop_loss_pct: float, take_profit_pct: float,
                      fees_pct: float, slippage_pct: float, funding_rate_daily: float,
                      start_date, reentry: bool) -> pd.DataFrame:
    trades  = []
    capital = 10000.0

    if start_date is not None:
        valid_preds = preds.index[preds.index >= pd.Timestamp(start_date)]
    else:
        valid_preds = preds.index

    if len(valid_preds) == 0:
        return pd.DataFrame()

    window_dates = valid_preds[::7]

    for pred_date in window_dates:
        if capital <= 0:
            break

        predicted_return = preds.loc[pred_date, 'predicted']
        take_trade = _eval_filters(pred_date, predicted_return, filter_df, filters)

        if not take_trade or pred_date not in prices.index:
            trades.append(_skip_record(pred_date, predicted_return, capital))
            continue

        record, exit_date = _execute_trade_record(
            pred_date, predicted_return, prices,
            leverage, stop_loss_pct, take_profit_pct,
            fees_pct, slippage_pct, funding_rate_daily, capital,
        )
        capital = record['capital_after']
        trades.append(record)

        # Re-entry: if TP/SL hit before window end, try one more trade from exit_date
        if reentry and exit_date is not None and capital > 0:
            window_end = pred_date + pd.Timedelta(days=7)
            if exit_date < window_end:
                re_candidates = preds.index[
                    (preds.index >= exit_date) & (preds.index < window_end)
                ]
                for reentry_date in re_candidates:
                    if capital <= 0:
                        break
                    if reentry_date not in prices.index:
                        continue
                    re_predicted = preds.loc[reentry_date, 'predicted']
                    if _eval_filters(reentry_date, re_predicted, filter_df, filters):
                        re_record, _ = _execute_trade_record(
                            reentry_date, re_predicted, prices,
                            leverage, stop_loss_pct, take_profit_pct,
                            fees_pct, slippage_pct, funding_rate_daily, capital,
                        )
                        capital = re_record['capital_after']
                        trades.append(re_record)
                        break  # one re-entry per window

    return pd.DataFrame(trades)


def render():
    st.markdown("<h1 style='margin-bottom:4px;'>Strategy Lab</h1>", unsafe_allow_html=True)
    st.caption("Backtest trading strategies using historical model predictions. No real money — educational only.")

    preds, prices = load_predictions()
    if preds.empty:
        st.error("No prediction results found. Run model evaluation first.")
        return

    filter_df = load_filter_data()
    _init_session_state()

    # ─── Top row: preset + start date + re-entry ────────────────────────────
    col_preset, col_date, col_reentry = st.columns([2, 2, 2])

    with col_preset:
        preset = st.selectbox("Profile", list(PRESETS.keys()), key='strategy_preset')

    _sync_preset(preset)

    with col_date:
        min_date = preds.index.min().date()
        max_date = (preds.index.max() - pd.Timedelta(days=30)).date()
        start_date = st.date_input(
            "Backtest start",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            help="Move forward for a more realistic out-of-sample test. Earlier dates include the 2019–2021 bull run.",
        )

    with col_reentry:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        reentry = st.toggle(
            "Re-enter on close",
            value=st.session_state.get('strat_reentry', False),
            help="When a trade closes early (TP/SL hit), immediately look for a new entry "
                 "within the remaining window days — instead of waiting for the next 7-day window.",
        )

    # ─── Controls: custom filters or preset description ─────────────────────
    if preset == 'Custom':
        with st.expander("Customize", expanded=True):
            st.markdown("**Signal Filters** — a trade opens only when all active filters pass")

            min_magnitude = st.slider(
                "Min predicted return (%)", 0.0, 15.0,
                key='strat_min_magnitude', step=0.5,
                help="Only enter when the model predicts at least this % return over 7 days. "
                     "0 = any positive prediction.",
            )

            c1, c2 = st.columns(2)
            with c1:
                trend_enabled = st.toggle(
                    "Trend filter (above SMA)", key='strat_trend_enabled',
                    help="Only enter when BTC is trading above its moving average (uptrend regime).",
                )
            with c2:
                macd_enabled = st.toggle(
                    "MACD filter (histogram > 0)", key='strat_macd_enabled',
                    help="Only enter when the daily MACD histogram is positive — "
                         "momentum is building in the bullish direction.",
                )

            c3, c4 = st.columns(2)
            with c3:
                rsi_enabled = st.toggle(
                    "RSI filter", key='strat_rsi_enabled',
                    help="Skip trade if BTC RSI is above the threshold (overbought).",
                )
            with c4:
                volume_enabled = st.toggle(
                    "Volume filter (above 30-day average)", key='strat_volume_enabled',
                    help="Only enter on above-average volume days — avoids low-conviction moves.",
                )

            if trend_enabled:
                sma_period = st.selectbox("SMA period", [50, 200], key='strat_sma_period')
            else:
                sma_period = st.session_state.get('strat_sma_period', 50)

            if rsi_enabled:
                rsi_max = st.slider(
                    "RSI max threshold", 50.0, 90.0, key='strat_rsi_max', step=1.0,
                    help="Enter only when RSI is below this value.",
                )
            else:
                rsi_max = st.session_state.get('strat_rsi_max', 70.0)

            st.markdown("---")
            st.markdown("**Risk Management**")
            c5, c6 = st.columns(2)
            with c5:
                stop_loss_pct = st.slider(
                    "Stop Loss (%)", 0.0, 20.0, key='strat_sl', step=0.5,
                    help="0 = disabled. Closes position if price drops by this %.",
                )
            with c6:
                take_profit_pct = st.slider(
                    "Take Profit (%)", 0.0, 50.0, key='strat_tp', step=1.0,
                    help="0 = disabled. Closes position if price rises by this %.",
                )

            st.markdown("**Execution**")
            c7, c8 = st.columns(2)
            with c7:
                leverage = st.slider(
                    "Leverage", 1.0, 40.0, key='strat_leverage', step=0.5,
                    help="1x = no leverage (spot). Higher = more risk and reward.",
                )
                if leverage > 10:
                    st.warning(f"At {leverage}x, a {100/leverage:.1f}% drop = liquidation.")
            with c8:
                funding_rate = st.slider(
                    "Funding Rate (%/day)", 0.0, 0.10, key='strat_funding', step=0.005,
                    help="Daily cost of holding a leveraged position (perpetual futures). "
                         "0.03%/day = standard Binance/Bybit rate. Only applies above 1x.",
                    format="%.3f",
                    disabled=(leverage == 1.0),
                ) / 100

            c9, c10 = st.columns(2)
            with c9:
                slippage_pct = st.slider(
                    "Slippage (%)", 0.0, 0.5, key='strat_slippage', step=0.05,
                    help="Market impact on fill price. Entry costs more, exits pay less.",
                ) / 100
            with c10:
                fees_pct = st.slider(
                    "Trading Fees (%)", 0.0, 1.0, key='strat_fees', step=0.05,
                    help="Round-trip cost per trade (entry + exit). 0.2% is typical for spot crypto.",
                ) / 100

        # Active filter summary
        active = ["Bullish prediction"]
        if min_magnitude > 0:
            active.append(f"≥{min_magnitude:.1f}% predicted return")
        if rsi_enabled:
            active.append(f"RSI < {rsi_max:.0f}")
        if trend_enabled:
            active.append(f"Above {sma_period}-day SMA")
        if macd_enabled:
            active.append("MACD histogram > 0")
        if volume_enabled:
            active.append("Volume above average")
        st.info("**Active filters:** " + " · ".join(active))

    else:
        p               = PRESETS[preset]
        min_magnitude   = p['min_magnitude']
        rsi_enabled     = p['rsi_enabled']
        rsi_max         = p['rsi_max']
        trend_enabled   = p['trend_enabled']
        sma_period      = p['sma_period']
        macd_enabled    = p['macd_enabled']
        volume_enabled  = p['volume_enabled']
        stop_loss_pct   = p['stop_loss_pct']
        take_profit_pct = p['take_profit_pct']
        leverage        = p['leverage']
        fees_pct        = p['fees_pct'] / 100
        slippage_pct    = p['slippage_pct'] / 100
        funding_rate    = p['funding_rate'] / 100

        st.info(PROFILE_DESCRIPTIONS[preset])

    filters = {
        'min_magnitude':  min_magnitude,
        'rsi_enabled':    rsi_enabled,
        'rsi_max':        rsi_max,
        'trend_enabled':  trend_enabled,
        'sma_period':     sma_period,
        'macd_enabled':   macd_enabled,
        'volume_enabled': volume_enabled,
    }

    # ─── Run simulation ─────────────────────────────────────────────────────
    trades = simulate_strategy(
        preds, prices, filter_df, filters,
        leverage, stop_loss_pct, take_profit_pct,
        fees_pct, slippage_pct, funding_rate,
        start_date, reentry,
    )

    if trades.empty:
        st.warning("No trades to display with the current settings.")
        return

    # ─── Honest framing callout (only when backtest covers > 1 year of history) ──
    from datetime import date as _date, timedelta as _timedelta
    if start_date < (_date.today() - _timedelta(days=365)):
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
    drawdown = (equity_curve - peak) / peak * 100
    max_dd  = drawdown.min()

    liquidations = (
        active_trades['exit_reason'].str.contains('LIQUIDATED').sum()
        if len(active_trades) > 0 else 0
    )

    cols = st.columns(5)
    with cols[0]:
        styled_metric("Final Capital", f"${final_capital:,.0f}", f"{total_return:+.1f}%",
                      color='emerald' if total_return >= 0 else 'rose')
    with cols[1]:
        wr = f"{(active_trades['leveraged_return_pct'] > 0).mean():.0%}" if len(active_trades) > 0 else "—"
        styled_metric("Win Rate", wr, color='blue')
    with cols[2]:
        styled_metric("Trades", str(len(active_trades)), f"{len(skipped)} skipped", color='violet')
    with cols[3]:
        styled_metric("Liquidations", str(liquidations),
                      color='rose' if liquidations > 0 else 'emerald')
    with cols[4]:
        styled_metric("Max Drawdown", f"{max_dd:.1f}%", color='amber')

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ─── Tabs ───────────────────────────────────────────────────────────────
    tab_equity, tab_trades, tab_stats = st.tabs(["Equity Curve", "Trade Log", "Statistics"])

    preset_label = preset if preset != 'Custom' else 'Custom filters'

    with tab_equity:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=trades['date'], y=trades['capital_after'],
            name=f'{preset_label} ({leverage:.0f}x)',
            line=dict(color='#3b82f6', width=2),
            mode='lines',
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
            mode='lines',
        ))

        # Entry markers — capital before the trade opened
        if len(active_trades) > 0:
            capital_at_entry = active_trades['capital_after'] - active_trades['pnl']
            fig.add_trace(go.Scatter(
                x=active_trades['date'], y=capital_at_entry,
                mode='markers', name='Entry',
                marker=dict(color='#60a5fa', size=8, symbol='circle-open',
                            line=dict(width=2, color='#60a5fa')),
            ))

        # Exit markers — plotted at exit_date with post-trade capital
        liq_trades = active_trades[active_trades['exit_reason'].str.contains('LIQUIDATED')]
        if len(liq_trades) > 0:
            fig.add_trace(go.Scatter(
                x=liq_trades['exit_date'], y=liq_trades['capital_after'],
                mode='markers', name='Liquidation',
                marker=dict(color='#f43f5e', size=12, symbol='x'),
            ))

        sl_trades = active_trades[active_trades['exit_reason'].str.contains('Stop loss')]
        if len(sl_trades) > 0:
            fig.add_trace(go.Scatter(
                x=sl_trades['exit_date'], y=sl_trades['capital_after'],
                mode='markers', name='Stop Loss',
                marker=dict(color='#f43f5e', size=9, symbol='triangle-down'),
            ))

        tp_trades = active_trades[active_trades['exit_reason'].str.contains('Take profit')]
        if len(tp_trades) > 0:
            fig.add_trace(go.Scatter(
                x=tp_trades['exit_date'], y=tp_trades['capital_after'],
                mode='markers', name='Take Profit',
                marker=dict(color='#10b981', size=9, symbol='triangle-up'),
            ))

        eow_trades = active_trades[active_trades['exit_reason'] == 'End of window']
        if len(eow_trades) > 0:
            fig.add_trace(go.Scatter(
                x=eow_trades['exit_date'], y=eow_trades['capital_after'],
                mode='markers', name='Window close',
                marker=dict(color='#94a3b8', size=7, symbol='circle'),
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
            st.info("No trades taken with current settings.")

    with tab_stats:
        if len(active_trades) > 0:
            col_stats1, col_stats2 = st.columns(2)

            with col_stats1:
                st.markdown("**Return Statistics**")
                wins   = active_trades[active_trades['leveraged_return_pct'] > 0]
                losses = active_trades[active_trades['leveraged_return_pct'] <= 0]

                stats = {
                    'Total Trades': len(active_trades),
                    'Winning':      len(wins),
                    'Losing':       len(losses),
                    'Win Rate':     f"{len(wins)/len(active_trades):.0%}",
                    'Avg Win':      f"{wins['leveraged_return_pct'].mean():+.2f}%" if len(wins) > 0 else "—",
                    'Avg Loss':     f"{losses['leveraged_return_pct'].mean():+.2f}%" if len(losses) > 0 else "—",
                    'Best Trade':   f"{active_trades['leveraged_return_pct'].max():+.2f}%",
                    'Worst Trade':  f"{active_trades['leveraged_return_pct'].min():+.2f}%",
                }
                st.dataframe(pd.DataFrame(stats.items(), columns=['Metric', 'Value']),
                             use_container_width=True, hide_index=True)

            with col_stats2:
                st.markdown("**Risk Metrics**")
                returns_series = active_trades['leveraged_return_pct'] / 100
                sharpe = (
                    (returns_series.mean() / returns_series.std()) * np.sqrt(52)
                    if returns_series.std() > 0 else 0
                )
                gross_profit  = wins['pnl'].sum() if len(wins) > 0 else 0
                gross_loss    = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

                risk_stats = {
                    'Sharpe Ratio':     f"{sharpe:.2f}",
                    'Profit Factor':    f"{profit_factor:.2f}",
                    'Max Drawdown':     f"{max_dd:.1f}%",
                    'Liquidations':     liquidations,
                    'Stop Losses Hit':  len(sl_trades) if len(active_trades) > 0 else 0,
                    'Take Profits Hit': len(tp_trades) if len(active_trades) > 0 else 0,
                    'Trades Skipped':   len(skipped),
                }
                st.dataframe(pd.DataFrame(risk_stats.items(), columns=['Metric', 'Value']),
                             use_container_width=True, hide_index=True)

            st.markdown("**Return Distribution**")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=active_trades['leveraged_return_pct'],
                nbinsx=30, marker_color='#3b82f6', opacity=0.7, name='Trade Returns',
            ))
            fig_hist.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
            fig_hist.update_layout(
                height=280, xaxis_title='Return (%)', yaxis_title='Count', **DARK_LAYOUT,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    date_from = preds.index.min().strftime('%b %Y')
    date_to   = preds.index.max().strftime('%b %Y')
    reentry_note = " · Re-entry on early close enabled." if reentry else ""
    st.info(
        f"Backtest period: **{start_date} → {date_to}** (full data: {date_from} → {date_to}) — "
        f"walk-forward predictions only (model trained on past data at each point). "
        f"Fees: **{fees_pct*100:.2f}% per trade**.{reentry_note}"
    )
