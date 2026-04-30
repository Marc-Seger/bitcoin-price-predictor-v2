"""
Page 1 — Financial Dashboard

Interactive charts and indicators for BTC and correlated assets.
TradingView-inspired: price chart with overlay indicators, RSI, MACD,
volume, sentiment, on-chain metrics, and cross-asset analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from config import MASTER_DF_PATH, ASSETS, COL_FEAR_GREED, COL_GOOGLE_TRENDS, COL_ETF_FLOW
from components import CARD_COLORS, ASSET_COLORS, DARK_LAYOUT, CHART_BG, TEXT_COLOR, styled_metric


@st.cache_data(ttl=60)
def load_data():
    """Load master_df. Cache invalidates if file is modified."""
    return pd.read_csv(MASTER_DF_PATH, index_col=0, parse_dates=True, low_memory=False)


def render():
    st.markdown("<h1 style='margin-bottom:4px;'>Financial Dashboard</h1>", unsafe_allow_html=True)

    df = load_data()

    ASSET_LABELS = {'BTC': 'Bitcoin (BTC)', 'SP500': 'S&P 500', 'NASDAQ': 'NASDAQ',
                    'GOLD': 'Gold', 'DXY': 'Dollar Index'}

    # ─── Asset selector — drives KPI cards and all tabs ───
    asset = st.selectbox(
        "Asset",
        list(ASSET_LABELS.keys()),
        format_func=lambda a: ASSET_LABELS[a],
        key='dash_asset',
    )
    close_col = f'Close_{asset}'

    # ─── KPI cards ───
    valid = df.dropna(subset=[close_col])
    if valid.empty:
        st.warning(f"No data available for {ASSET_LABELS.get(asset, asset)}.")
        return

    latest = valid.iloc[-1]
    price = latest[close_col]
    prev_7d = valid.iloc[-8][close_col] if len(valid) > 8 else price
    prev_30d = valid.iloc[-31][close_col] if len(valid) > 31 else price
    ath = df[close_col].max()

    data_date = valid.index[-1].strftime("%d %b %Y")
    st.caption(f"Data as of {data_date}")

    price_fmt = f"${price:,.0f}" if asset != 'DXY' else f"{price:,.2f}"

    cols = st.columns(5)
    with cols[0]:
        styled_metric(f"{asset} Price", price_fmt, color='blue')
    with cols[1]:
        chg_7d = (price / prev_7d - 1)
        styled_metric("7d Change", f"{chg_7d:+.1%}", color='emerald' if chg_7d >= 0 else 'rose')
    with cols[2]:
        chg_30d = (price / prev_30d - 1)
        styled_metric("30d Change", f"{chg_30d:+.1%}", color='emerald' if chg_30d >= 0 else 'rose')
    with cols[3]:
        ath_dist = (price / ath - 1)
        styled_metric("From ATH", f"{ath_dist:+.1%}", color='emerald' if ath_dist >= 0 else 'rose')
    with cols[4]:
        rsi_col = f'RSI_Close_{asset}'
        rsi_val = latest.get(rsi_col) if rsi_col in df.columns else None
        if pd.notna(rsi_val):
            if rsi_val > 70:
                rsi_label, rsi_color = "Overbought", 'emerald'
            elif rsi_val < 30:
                rsi_label, rsi_color = "Oversold", 'rose'
            else:
                rsi_label, rsi_color = "Neutral", 'amber'
            styled_metric("RSI (14)", f"{rsi_val:.0f} — {rsi_label}", color=rsi_color)
        else:
            styled_metric("RSI (14)", "—", color='violet')

    # ─── Dynamic tabs — Sentiment and On-Chain are BTC-only ───
    tab_names = ["Price & Indicators"]
    if asset == 'BTC':
        tab_names += ["Sentiment", "On-Chain"]
    tab_names += ["Cross-Asset", "Technical Analysis"]
    tabs = st.tabs(tab_names)
    tab_map = {name: tabs[i] for i, name in enumerate(tab_names)}

    # ══════════════════════════════════════════
    # TAB 1: Price & Indicators
    # ══════════════════════════════════════════
    with tab_map["Price & Indicators"]:
        # Row 1: Overlays + Subplots
        col_overlay, col_sub = st.columns(2)
        with col_overlay:
            overlays = st.multiselect(
                "Overlays",
                ["SMA 9", "SMA 20", "SMA 50", "SMA 200", "Bollinger Bands"],
                default=["SMA 50", "SMA 200"]
            )
        with col_sub:
            subplots_on = st.multiselect(
                "Subplots", ["RSI", "MACD"], default=["RSI"]
            )

        # Row 2: Timeframe + Log scale + Volume toggle
        col_tf, col_log, col_vol = st.columns([1, 0.5, 0.5])
        with col_tf:
            timeframe = st.selectbox("Timeframe", ["1M", "3M", "6M", "1Y", "2Y", "All"], index=3)
        with col_log:
            log_scale = st.checkbox("Log scale", value=False)
        with col_vol:
            show_volume = st.checkbox("Volume", value=True)

        # Full dataset passed to Plotly — initial view window set via xaxis.range
        # so scroll-out reveals data beyond the selected timeframe
        plot_df = df.copy()
        tf_days_map = {'1M': 30, '3M': 90, '6M': 180, '1Y': 365, '2Y': 730}
        if timeframe in tf_days_map:
            x_start = df.index[-1] - pd.Timedelta(days=tf_days_map[timeframe])
        else:
            x_start = df.index[0]
        x_end = df.index[-1]

        # Build chart — volume overlays on price with secondary y-axis
        n_subs = 1 + len(subplots_on)
        row_heights = [0.6] + [0.2] * len(subplots_on) if subplots_on else [1.0]
        total = sum(row_heights)
        row_heights = [h / total for h in row_heights]

        # Use secondary_y for volume overlay on row 1
        fig = make_subplots(
            rows=n_subs, cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=row_heights,
            specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]] * (n_subs - 1),
        )

        # ─── Volume bars (behind price, on secondary y-axis) ───
        vol_col = f'Volume_{asset}'
        open_col = f'Open_{asset}'
        n_visible = (plot_df.index >= x_start).sum()

        if show_volume and vol_col in plot_df.columns:
            vol_series = plot_df[vol_col].fillna(0)
            open_series = plot_df.get(open_col, plot_df[close_col]).fillna(0)
            close_series = plot_df[close_col].fillna(0)

            # Resample to weekly when >600 visible points to avoid sub-pixel bars
            if n_visible > 600:
                # W-FRI: bars land on Fridays (trading days for all assets)
                # avoids Sunday labels being hidden by rangebreaks on non-BTC assets
                vol_plot = vol_series.resample('W-FRI').sum()
                o_plot = open_series.resample('W-FRI').first().fillna(0)
                c_plot = close_series.resample('W-FRI').last().fillna(0)
            else:
                vol_plot = vol_series
                o_plot = open_series
                c_plot = close_series

            bar_colors = ['#10b981' if c >= o else '#f43f5e'
                          for c, o in zip(c_plot, o_plot)]
            fig.add_trace(go.Bar(
                x=vol_plot.index, y=vol_plot,
                name='Volume', marker_color=bar_colors, opacity=0.4,
            ), row=1, col=1, secondary_y=True)
            # Scale to 95th percentile of visible window
            visible_vol = vol_plot.loc[x_start:x_end]
            ref_vol = (visible_vol if not visible_vol.empty else vol_plot).quantile(0.95)
            fig.update_yaxes(
                secondary_y=True, row=1, col=1,
                range=[0, ref_vol * 4],
                showgrid=False, showticklabels=False, title_text="",
            )

        # ─── Candlestick / line chart (primary y-axis) ───
        high_col = f'High_{asset}'
        low_col = f'Low_{asset}'

        if all(c in plot_df.columns for c in [open_col, high_col, low_col, close_col]):
            if n_visible > 600:
                candle_df = pd.DataFrame({
                    open_col:  plot_df[open_col].resample('W-FRI').first(),
                    high_col:  plot_df[high_col].resample('W-FRI').max(),
                    low_col:   plot_df[low_col].resample('W-FRI').min(),
                    close_col: plot_df[close_col].resample('W-FRI').last(),
                }).dropna()
            else:
                candle_df = plot_df
            fig.add_trace(go.Candlestick(
                x=candle_df.index,
                open=candle_df[open_col], high=candle_df[high_col],
                low=candle_df[low_col], close=candle_df[close_col],
                name=asset, increasing_line_color='#10b981', decreasing_line_color='#f43f5e',
            ), row=1, col=1, secondary_y=False)
        elif close_col in plot_df.columns:
            src = plot_df[close_col].resample('W-FRI').last() if n_visible > 600 else plot_df[close_col]
            fig.add_trace(go.Scatter(
                x=src.index, y=src,
                name=asset, line=dict(color=ASSET_COLORS.get(asset, '#3b82f6'), width=2),
            ), row=1, col=1, secondary_y=False)

        # ─── Overlays (on primary y-axis) ───
        sma_colors = {'SMA 9': '#f59e0b', 'SMA 20': '#3b82f6', 'SMA 50': '#8b5cf6', 'SMA 200': '#f43f5e'}
        for overlay in overlays:
            if overlay.startswith('SMA'):
                window = int(overlay.split(' ')[1])
                col_name = f'SMA_{window}_Close_{asset}'
                if col_name in plot_df.columns:
                    fig.add_trace(go.Scatter(
                        x=plot_df.index, y=plot_df[col_name],
                        name=overlay, line=dict(width=1, color=sma_colors.get(overlay, 'gray')),
                    ), row=1, col=1, secondary_y=False)

            elif overlay == 'Bollinger Bands':
                upper = f'Upper_Band_Close_{asset}'
                lower = f'Lower_Band_Close_{asset}'
                if upper in plot_df.columns and lower in plot_df.columns:
                    fig.add_trace(go.Scatter(
                        x=plot_df.index, y=plot_df[upper], name='BB Upper',
                        line=dict(width=1, color='rgba(139,92,246,0.5)'),
                    ), row=1, col=1, secondary_y=False)
                    fig.add_trace(go.Scatter(
                        x=plot_df.index, y=plot_df[lower], name='BB Lower',
                        line=dict(width=1, color='rgba(139,92,246,0.5)'),
                        fill='tonexty', fillcolor='rgba(139,92,246,0.08)',
                    ), row=1, col=1, secondary_y=False)

        # ─── Subplots (RSI, MACD) ───
        subplot_row = 2
        for subplot in subplots_on:
            rsi_col_sub = f'RSI_Close_{asset}'
            macd_col = f'MACD_D_{asset}'

            if subplot == 'RSI' and rsi_col_sub in plot_df.columns:
                fig.add_trace(go.Scatter(
                    x=plot_df.index, y=plot_df[rsi_col_sub],
                    name='RSI', line=dict(color='#8b5cf6', width=1.5),
                ), row=subplot_row, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="#f43f5e", opacity=0.5, row=subplot_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="#10b981", opacity=0.5, row=subplot_row, col=1)
                fig.update_yaxes(title_text="RSI", range=[0, 100], row=subplot_row, col=1)
                subplot_row += 1

            elif subplot == 'MACD' and macd_col in plot_df.columns:
                signal_col = f'Signal_Line_D_{asset}'
                hist_col = f'MACD_Histogram_D_{asset}'
                fig.add_trace(go.Scatter(
                    x=plot_df.index, y=plot_df[macd_col],
                    name='MACD', line=dict(color='#3b82f6', width=1.5),
                ), row=subplot_row, col=1)
                if signal_col in plot_df.columns:
                    fig.add_trace(go.Scatter(
                        x=plot_df.index, y=plot_df[signal_col],
                        name='Signal', line=dict(color='#f59e0b', width=1.5),
                    ), row=subplot_row, col=1)
                if hist_col in plot_df.columns:
                    hist = plot_df[hist_col]
                    colors_h = ['#10b981' if v >= 0 else '#f43f5e' for v in hist]
                    fig.add_trace(go.Bar(
                        x=plot_df.index, y=hist,
                        name='Histogram', marker_color=colors_h, opacity=0.5,
                    ), row=subplot_row, col=1)
                fig.update_yaxes(title_text="MACD", row=subplot_row, col=1)
                subplot_row += 1

        # Weekend rangebreaks for non-crypto assets (markets closed Sat–Mon)
        if asset != 'BTC':
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

        # Set price y-axis range to visible window (skip when log scale — let Plotly auto-range)
        hi_col = f'High_{asset}' if f'High_{asset}' in plot_df.columns else close_col
        lo_col = f'Low_{asset}' if f'Low_{asset}' in plot_df.columns else close_col
        window = plot_df.loc[x_start:x_end]
        if log_scale:
            if not window.empty:
                p_min = max(window[lo_col].dropna().min(), 1)
                p_max = window[hi_col].dropna().max()
                # Plotly log-axis range is in log10 units
                fig.update_layout(yaxis=dict(
                    type='log',
                    range=[np.log10(p_min * 0.97), np.log10(p_max * 1.03)],
                ))
            else:
                fig.update_layout(yaxis=dict(type='log'))
        else:
            if not window.empty:
                p_min = window[lo_col].min()
                p_max = window[hi_col].max()
                pad = (p_max - p_min) * 0.05
                fig.update_layout(yaxis=dict(range=[p_min - pad, p_max + pad]))

        fig.update_xaxes(range=[x_start, x_end])

        fig.update_layout(
            height=150 + 220 * n_subs,
            xaxis_rangeslider_visible=False,
            barmode='overlay',
            title=dict(text=ASSET_LABELS.get(asset, asset), font=dict(size=14, color=TEXT_COLOR), x=0),
            **DARK_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ══════════════════════════════════════════
    # TAB 2: Sentiment (BTC only)
    # ══════════════════════════════════════════
    if "Sentiment" in tab_map:
     with tab_map["Sentiment"]:
        sent_tf = st.selectbox("Period", ["3M", "6M", "1Y", "2Y", "All"], index=2, key="sent_tf")
        sent_days = {'3M': 90, '6M': 180, '1Y': 365, '2Y': 730, 'All': len(df)}
        sent_n = sent_days[sent_tf]

        col_gauge, col_timeline = st.columns([1, 2])

        fg_col = COL_FEAR_GREED

        with col_gauge:
            st.markdown("### Fear & Greed Index")
            if fg_col in df.columns:
                fg_val = df[fg_col].dropna().iloc[-1]
                fg_label_map = {
                    range(0, 25): ("Extreme Fear", "#f43f5e"),
                    range(25, 45): ("Fear", "#f59e0b"),
                    range(45, 55): ("Neutral", "#fbbf24"),
                    range(55, 75): ("Greed", "#10b981"),
                    range(75, 101): ("Extreme Greed", "#06b6d4"),
                }
                fg_label, fg_color = "Unknown", "gray"
                for rng, (lbl, clr) in fg_label_map.items():
                    if int(fg_val) in rng:
                        fg_label, fg_color = lbl, clr
                        break

                # Label rendered outside Plotly to avoid overlapping the arc
                st.markdown(
                    f"<p style='text-align:center; font-size:15px; font-weight:600; "
                    f"color:{fg_color}; margin-bottom:0;'>{fg_label}</p>",
                    unsafe_allow_html=True,
                )
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fg_val,
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': TEXT_COLOR},
                        'bar': {'color': fg_color},
                        'bgcolor': CHART_BG,
                        'steps': [
                            {'range': [0, 25], 'color': 'rgba(244,63,94,0.2)'},
                            {'range': [25, 45], 'color': 'rgba(245,158,11,0.2)'},
                            {'range': [45, 55], 'color': 'rgba(251,191,36,0.2)'},
                            {'range': [55, 75], 'color': 'rgba(16,185,129,0.2)'},
                            {'range': [75, 100], 'color': 'rgba(6,182,212,0.2)'},
                        ],
                    },
                ))
                gauge_fig.update_layout(height=200, **{k: v for k, v in DARK_LAYOUT.items() if k != 'legend'})
                st.plotly_chart(gauge_fig, use_container_width=True)
            else:
                st.info("Fear & Greed data not available.")

        with col_timeline:
            st.markdown("### Fear & Greed Over Time")
            if fg_col in df.columns:
                fg_data = df[fg_col].dropna().tail(sent_n)
                fig_fg = go.Figure()
                fig_fg.add_trace(go.Scatter(
                    x=fg_data.index, y=fg_data.values,
                    fill='tozeroy', fillcolor='rgba(139,92,246,0.15)',
                    line=dict(color='#8b5cf6', width=1.5),
                    name='Fear & Greed',
                ))
                fig_fg.add_hline(y=25, line_dash="dash", line_color="#f43f5e", opacity=0.4)
                fig_fg.add_hline(y=75, line_dash="dash", line_color="#10b981", opacity=0.4)
                fig_fg.update_layout(height=260, yaxis_range=[0, 100], **DARK_LAYOUT)
                st.plotly_chart(fig_fg, use_container_width=True)

        # Google Trends
        st.markdown("### Google Trends — Bitcoin Search Interest")
        trends_col = COL_GOOGLE_TRENDS
        if trends_col in df.columns:
            trends = df[trends_col].dropna().tail(sent_n)
            if not trends.empty:
                fig_trends = go.Figure()
                fig_trends.add_trace(go.Scatter(
                    x=trends.index, y=trends.values,
                    fill='tozeroy', fillcolor='rgba(59,130,246,0.15)',
                    line=dict(color='#3b82f6', width=1.5),
                    name='Search Interest',
                ))
                fig_trends.update_layout(height=200, **DARK_LAYOUT)
                st.plotly_chart(fig_trends, use_container_width=True)
            else:
                st.info("No Google Trends data available for the selected period.")

        # ETF Flows
        st.markdown("### Bitcoin ETF Daily Flows")
        st.caption("Net institutional inflow/outflow across all spot Bitcoin ETFs ($M). Available from Jan 2024.")
        etf_col = COL_ETF_FLOW
        if etf_col in df.columns:
            etf_data = df[etf_col].dropna().tail(sent_n)
            if not etf_data.empty:
                colors = ['#10b981' if v >= 0 else '#f43f5e' for v in etf_data.values]
                fig_etf = go.Figure()
                fig_etf.add_trace(go.Bar(
                    x=etf_data.index, y=etf_data.values,
                    marker_color=colors,
                    name='Net Flow ($M)',
                ))
                fig_etf.add_hline(y=0, line_color='#56657e', opacity=0.5)
                fig_etf.update_layout(height=220, yaxis_title='$M', **DARK_LAYOUT)
                st.plotly_chart(fig_etf, use_container_width=True)
            else:
                st.info("ETF flow data not available for the selected period (data starts Jan 2024).")

    # ══════════════════════════════════════════
    # TAB 3: On-Chain (BTC only)
    # ══════════════════════════════════════════
    if "On-Chain" in tab_map:
     with tab_map["On-Chain"]:
        oc_tf = st.selectbox("Period", ["3M", "6M", "1Y", "2Y", "All"], index=2, key="oc_tf")
        oc_n = {'3M': 90, '6M': 180, '1Y': 365, '2Y': 730, 'All': len(df)}[oc_tf]
        st.markdown("### On-Chain Metrics")

        onchain_metrics = {
            'OnChain_Hash_Rate': ('Hash Rate', 'Network security — higher = more miners', 'blue'),
            'OnChain_Active_Addresses': ('Active Addresses', 'Daily unique addresses — user activity proxy', 'emerald'),
            'OnChain_MVRV_Ratio': ('MVRV Ratio', '>3.5 = historically overbought, <1 = undervalued', 'amber'),
            'OnChain_Transaction_Count': ('Transactions', 'Daily on-chain transaction count', 'violet'),
        }

        for col_name, (display_name, description, color) in onchain_metrics.items():
            if col_name not in df.columns:
                continue
            data = df[col_name].dropna().tail(oc_n)
            if data.empty:
                continue

            col_val, col_chart = st.columns([1, 3])
            with col_val:
                current = data.iloc[-1]
                prev = data.iloc[-8] if len(data) > 8 else data.iloc[0]
                change = (current / prev - 1) if prev != 0 else 0
                styled_metric(display_name, f"{current:,.0f}", f"{change:+.1%} (7d)", color=color)
                st.caption(description)
            with col_chart:
                fig_oc = go.Figure()
                fig_oc.add_trace(go.Scatter(
                    x=data.index, y=data.values,
                    fill='tozeroy', fillcolor=f"rgba({','.join(str(int(CARD_COLORS[color][i:i+2], 16)) for i in (1, 3, 5))},0.1)",
                    line=dict(color=CARD_COLORS[color], width=1.5),
                    name=display_name,
                ))
                fig_oc.update_layout(height=140, showlegend=False, **DARK_LAYOUT)
                st.plotly_chart(fig_oc, use_container_width=True)

    # ══════════════════════════════════════════
    # TAB 4: Cross-Asset
    # ══════════════════════════════════════════
    with tab_map["Cross-Asset"]:
        st.markdown("### Cross-Asset Comparison")

        timeframe_cross = st.selectbox("Period", ["3M", "6M", "1Y", "2Y", "All"], index=2, key="cross_tf")
        days_cross = {'3M': 90, '6M': 180, '1Y': 365, '2Y': 730, 'All': len(df)}
        cross_df = df.tail(days_cross[timeframe_cross])

        # DXY excluded here — it's a model input / index, not an investable asset.
        # It remains available in the Price & Indicators asset selector.
        CROSS_ASSETS = ['BTC', 'SP500', 'NASDAQ', 'GOLD']

        # Normalized performance
        st.markdown("**Normalized Performance (% change from start)**")
        perf_data = pd.DataFrame()
        for a in CROSS_ASSETS:
            cc = f'Close_{a}'
            if cc in cross_df.columns:
                series = cross_df[cc].dropna()
                if len(series) > 0:
                    perf_data[a] = (series / series.iloc[0] - 1) * 100

        if not perf_data.empty:
            fig_perf = go.Figure()
            for a in perf_data.columns:
                fig_perf.add_trace(go.Scatter(
                    x=perf_data.index, y=perf_data[a],
                    name=a, line=dict(color=ASSET_COLORS.get(a, '#e8edf5'), width=2),
                ))
            fig_perf.update_layout(height=350, yaxis_title="% Change", **DARK_LAYOUT)
            st.plotly_chart(fig_perf, use_container_width=True)

        # ── Period return table ───────────────────────────────────────────
        st.markdown("**Period Returns (%)**")
        close_cols = {a: f'Close_{a}' for a in CROSS_ASSETS if f'Close_{a}' in cross_df.columns}
        if close_cols:
            periods = {'1W': 7, '1M': 30, '3M': 90, '6M': 180, '1Y': 365, '3Y': 1095, '5Y': 1825, '10Y': 3650}
            rows = []
            for a, cc in close_cols.items():
                series = df[cc].dropna()
                row = {'Asset': a}
                for label, days in periods.items():
                    if len(series) > days:
                        ret = (series.iloc[-1] / series.iloc[-1 - days] - 1) * 100
                        row[label] = ret
                    else:
                        row[label] = None
                rows.append(row)
            ret_df = pd.DataFrame(rows).set_index('Asset')

            # Build styled HTML table
            def _cell(val):
                if val is None:
                    return "<td style='text-align:right;color:#56657e;'>—</td>"
                color = '#10b981' if val >= 0 else '#f43f5e'
                return f"<td style='text-align:right;color:{color};font-weight:600;'>{val:+.1f}%</td>"

            header = "<tr><th style='text-align:left;color:#56657e;padding:6px 12px;'>Asset</th>"
            header += "".join(
                f"<th style='text-align:right;color:#56657e;padding:6px 12px;'>{p}</th>"
                for p in periods
            ) + "</tr>"

            body = ""
            for a, row_data in ret_df.iterrows():
                asset_color = ASSET_COLORS.get(a, '#e8edf5')
                body += (
                    f"<tr><td style='padding:6px 12px;font-weight:600;color:{asset_color};'>{a}</td>"
                    + "".join(_cell(row_data.get(p)) for p in periods)
                    + "</tr>"
                )

            st.markdown(
                f"<table style='width:100%;border-collapse:collapse;"
                f"background:#171f30;border-radius:8px;font-family:JetBrains Mono,monospace;"
                f"font-size:13px;'>{header}{body}</table>",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='margin-bottom:24px'></div>", unsafe_allow_html=True)

        # ── Volatility comparison ─────────────────────────────────────────
        st.markdown("**Rolling 30-Day Annualised Volatility**")
        if close_cols:
            fig_vol = go.Figure()
            for a, cc in close_cols.items():
                series = cross_df[cc].dropna()
                if series.empty:
                    continue
                daily_ret = series.pct_change()
                vol = daily_ret.rolling(30).std() * np.sqrt(365) * 100
                fig_vol.add_trace(go.Scatter(
                    x=vol.index, y=vol,
                    name=a, line=dict(color=ASSET_COLORS.get(a, '#e8edf5'), width=1.5),
                ))
            fig_vol.update_layout(
                height=260, yaxis_title="Annualised Vol %", **DARK_LAYOUT
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        # ── Drawdown chart ────────────────────────────────────────────────
        st.markdown("**Drawdown from Peak**")
        if close_cols:
            fig_dd = go.Figure()
            dd_min = 0.0
            for a, cc in close_cols.items():
                series = cross_df[cc].dropna()
                if series.empty:
                    continue
                rolling_max = series.cummax()
                drawdown = (series / rolling_max - 1) * 100
                dd_min = min(dd_min, float(drawdown.min()))
                fig_dd.add_trace(go.Scatter(
                    x=drawdown.index, y=drawdown,
                    name=a, line=dict(color=ASSET_COLORS.get(a, '#e8edf5'), width=1.5),
                ))
            fig_dd.add_hline(y=0, line_color='#56657e', opacity=0.3)
            fig_dd.update_layout(
                height=280, yaxis_title="Drawdown %",
                yaxis_range=[dd_min - 5, 0],
                **DARK_LAYOUT
            )
            st.plotly_chart(fig_dd, use_container_width=True)

    # ══════════════════════════════════════════
    # TAB 5: Technical Analysis
    # ══════════════════════════════════════════
    with tab_map["Technical Analysis"]:
        ta_tf = st.selectbox(
            "Timeframe", ["3M", "6M", "1Y", "2Y", "All"], index=3, key="ta_tf"
        )

        drawings = st.multiselect(
            "Chart Drawings",
            ["Support & Resistance", "Fibonacci Retracement", "Monthly Pivots", "ATH & Cycle Levels"],
            default=[],
            help="Select one or more drawings to overlay on the chart.",
        )

        # Prepare columns and slice data
        ta_close = f'Close_{asset}'
        ta_high  = f'High_{asset}'
        ta_low   = f'Low_{asset}'
        ta_open  = f'Open_{asset}'

        ta_days = {'3M': 90, '6M': 180, '1Y': 365, '2Y': 730, 'All': len(df)}
        ta_n    = ta_days[ta_tf]
        ta_df   = df.dropna(subset=[ta_close]).tail(ta_n)

        if ta_df.empty:
            st.warning(f"No data available for {ASSET_LABELS.get(asset, asset)}.")
        else:
            current_price = ta_df[ta_close].iloc[-1]
            hi_col = ta_high if ta_high in ta_df.columns else ta_close
            lo_col = ta_low  if ta_low  in ta_df.columns else ta_close

            # ── Base candlestick chart ──────────────────────────────────────
            fig_ta = go.Figure()

            if all(c in ta_df.columns for c in [ta_open, ta_high, ta_low, ta_close]):
                fig_ta.add_trace(go.Candlestick(
                    x=ta_df.index,
                    open=ta_df[ta_open], high=ta_df[ta_high],
                    low=ta_df[ta_low],  close=ta_df[ta_close],
                    name=asset,
                    increasing_line_color='#10b981', decreasing_line_color='#f43f5e',
                ))
            else:
                fig_ta.add_trace(go.Scatter(
                    x=ta_df.index, y=ta_df[ta_close],
                    name=asset,
                    line=dict(color=ASSET_COLORS.get(asset, '#3b82f6'), width=2),
                ))

            # ── Drawing 1: Support & Resistance zones ──────────────────────
            # Algorithm:
            #   1. Scan each bar for pivot highs/lows (extreme within n_pivot bars).
            #   2. Recency filter: only pivots whose last touch is ≤ 12 months ago.
            #   3. Cluster nearby pivots (within 1.5%) into zones.
            #   4. Score each zone with time-decay: each touch weighted by
            #      exp(-days_since_touch / 365) so recent touches score higher.
            #   5. Broken-level detection: if price has cleanly closed through the
            #      zone (3 consecutive closes past it), mark it as broken (dimmed).
            #   6. Draw top 10 active zones; broken zones shown dashed + dim.
            if "Support & Resistance" in drawings:
                high_s = ta_df[hi_col]
                low_s  = ta_df[lo_col]
                close_s = ta_df[ta_close]
                today_ts = ta_df.index[-1]
                cutoff_12m = today_ts - pd.Timedelta(days=365)

                n_pivot = max(5, len(ta_df) // 50)

                # Collect pivot (price, date) pairs
                pivot_points = []  # (price, date)
                for i in range(n_pivot, len(ta_df) - n_pivot):
                    bar_date = ta_df.index[i]
                    if high_s.iloc[i] == high_s.iloc[i - n_pivot: i + n_pivot + 1].max():
                        pivot_points.append((float(high_s.iloc[i]), bar_date))
                    if low_s.iloc[i] == low_s.iloc[i - n_pivot: i + n_pivot + 1].min():
                        pivot_points.append((float(low_s.iloc[i]), bar_date))

                # Recency filter: keep only pivots last touched ≤ 12 months ago
                pivot_points = [(p, d) for p, d in pivot_points if d >= cutoff_12m]

                def _cluster_with_dates(points, threshold=0.015):
                    if not points:
                        return []
                    sorted_pts = sorted(points, key=lambda x: x[0])
                    clusters = [[sorted_pts[0]]]
                    for pt in sorted_pts[1:]:
                        if (pt[0] - clusters[-1][0][0]) / clusters[-1][0][0] < threshold:
                            clusters[-1].append(pt)
                        else:
                            clusters.append([pt])
                    results = []
                    for c in clusters:
                        prices = [x[0] for x in c]
                        dates  = [x[1] for x in c]
                        center = float(np.mean(prices))
                        last_touch = max(dates)
                        # Time-decay score: sum of exp(-age_days/365) per touch
                        score = sum(
                            np.exp(-(today_ts - d).days / 365.0) for d in dates
                        )
                        results.append((center, len(c), last_touch, score))
                    return results

                levels_raw = sorted(
                    _cluster_with_dates(pivot_points),
                    key=lambda x: x[3], reverse=True
                )[:10]

                for level_price, touch_count, last_touch, score in levels_raw:
                    zone_half = level_price * 0.008
                    is_support = level_price < current_price

                    # Broken-level detection: 3 consecutive closes past the zone
                    zone_lo = level_price - zone_half
                    zone_hi = level_price + zone_half
                    recent_closes = close_s.iloc[-10:]
                    if is_support:
                        broken = (recent_closes < zone_lo).rolling(3).min().iloc[-1] == 1
                    else:
                        broken = (recent_closes > zone_hi).rolling(3).min().iloc[-1] == 1

                    if broken:
                        fill   = 'rgba(86,101,126,0.08)'
                        border = '#56657e'
                        label  = f"{'S' if is_support else 'R'}  {level_price:,.0f}  ({touch_count}✕ broken)"
                    else:
                        fill   = 'rgba(16,185,129,0.12)' if is_support else 'rgba(244,63,94,0.12)'
                        border = '#10b981' if is_support else '#f43f5e'
                        label  = f"{'S' if is_support else 'R'}  {level_price:,.0f}  ({touch_count}✕)"

                    fig_ta.add_hrect(
                        y0=zone_lo, y1=zone_hi,
                        fillcolor=fill,
                        line=dict(color=border, width=0.8, dash='dash' if broken else 'solid'),
                        opacity=1.0, layer='below',
                    )
                    fig_ta.add_annotation(
                        x=ta_df.index[-1], y=level_price,
                        text=label, xanchor='right', yanchor='middle',
                        font=dict(size=10, color=border),
                        showarrow=False, xref='x', yref='y',
                        bgcolor='rgba(15,21,32,0.6)',
                    )

            # ── Drawing 2: Fibonacci Retracement ──────────────────────────
            # Algorithm:
            #   Find the period's overall swing high and swing low.
            #   If the high came AFTER the low → uptrend: levels retrace downward.
            #   If the high came BEFORE the low → downtrend: levels retrace upward.
            #   Lines are clipped to start at the earlier swing point date so they
            #   don't extend back into price history before the swing was established.
            #   Swing start/end markers and date+price annotations are added.
            if "Fibonacci Retracement" in drawings:
                swing_high = float(ta_df[hi_col].max())
                swing_low  = float(ta_df[lo_col].min())
                swing_range = swing_high - swing_low

                high_idx = ta_df[hi_col].idxmax()
                low_idx  = ta_df[lo_col].idxmin()
                uptrend  = high_idx > low_idx  # high came after low

                # Swing start = earlier extreme, end = later extreme
                if uptrend:
                    swing_start_date, swing_start_price = low_idx,  swing_low
                    swing_end_date,   swing_end_price   = high_idx, swing_high
                else:
                    swing_start_date, swing_start_price = high_idx, swing_high
                    swing_end_date,   swing_end_price   = low_idx,  swing_low

                fib_levels = [
                    (0.000, '0.0%',   '#94a3b8'),
                    (0.236, '23.6%',  '#fbbf24'),
                    (0.382, '38.2%',  '#10b981'),
                    (0.500, '50.0%',  '#3b82f6'),
                    (0.618, '61.8%',  '#f59e0b'),
                    (0.786, '78.6%',  '#f43f5e'),
                    (1.000, '100.0%', '#94a3b8'),
                ]

                for ratio, lbl, color in fib_levels:
                    level_price = swing_high - ratio * swing_range if uptrend else swing_low + ratio * swing_range
                    # Clip line to start at the swing start date
                    fig_ta.add_shape(
                        type='line',
                        x0=swing_start_date, x1=ta_df.index[-1],
                        y0=level_price, y1=level_price,
                        line=dict(color=color, width=1, dash='dot'),
                        xref='x', yref='y',
                    )
                    fig_ta.add_annotation(
                        x=swing_start_date, y=level_price,
                        text=f"Fib {lbl}  {level_price:,.0f}",
                        xanchor='left', yanchor='bottom',
                        font=dict(size=10, color=color),
                        showarrow=False, xref='x', yref='y',
                        bgcolor='rgba(15,21,32,0.6)',
                    )

                # Swing start/end markers with date + price labels
                for date, price_pt, marker_label in [
                    (swing_start_date, swing_start_price, f"Swing start\n{swing_start_date.strftime('%d %b %Y')}  {swing_start_price:,.0f}"),
                    (swing_end_date,   swing_end_price,   f"Swing end\n{swing_end_date.strftime('%d %b %Y')}  {swing_end_price:,.0f}"),
                ]:
                    fig_ta.add_trace(go.Scatter(
                        x=[date], y=[price_pt],
                        mode='markers+text',
                        marker=dict(color='#e8edf5', size=8, symbol='diamond'),
                        text=[marker_label],
                        textposition='top center' if price_pt == swing_high else 'bottom center',
                        textfont=dict(size=9, color='#8899b4'),
                        showlegend=False,
                        hoverinfo='skip',
                    ))

            # ── Drawing 3: Monthly Pivot Points ───────────────────────────
            # Algorithm:
            #   Resample data to monthly bars (MS = month start). Take the last
            #   COMPLETE month (second-to-last row) to avoid in-progress data.
            #   Monthly pivots are more relevant for BTC's slow-moving timeframe
            #   than weekly pivots which flip too quickly.
            #   PP  = (High + Low + Close) / 3
            #   R1/R2/R3 = standard floor pivot resistance levels
            #   S1/S2/S3 = standard floor pivot support levels
            if "Monthly Pivots" in drawings:
                monthly = df[[hi_col, lo_col, ta_close]].dropna().resample('MS').agg({
                    hi_col:   'max',
                    lo_col:   'min',
                    ta_close: 'last',
                }).dropna()

                if len(monthly) >= 2:
                    pm = monthly.iloc[-2]  # last complete month
                    month_label = monthly.index[-2].strftime('%b %Y')
                    H, L, C = float(pm[hi_col]), float(pm[lo_col]), float(pm[ta_close])

                    PP = (H + L + C) / 3
                    R1 = 2 * PP - L
                    R2 = PP + (H - L)
                    R3 = H + 2 * (PP - L)
                    S1 = 2 * PP - H
                    S2 = PP - (H - L)
                    S3 = L - 2 * (H - PP)

                    pivot_lines = [
                        (PP, 'PP',  '#e8edf5', 'solid',  1.5),
                        (R1, 'R1',  '#fca5a5', 'dash',   1.0),
                        (R2, 'R2',  '#f43f5e', 'dash',   1.0),
                        (R3, 'R3',  '#dc2626', 'dash',   1.0),
                        (S1, 'S1',  '#6ee7b7', 'dash',   1.0),
                        (S2, 'S2',  '#10b981', 'dash',   1.0),
                        (S3, 'S3',  '#059669', 'dash',   1.0),
                    ]
                    for level_price, label, color, dash, width in pivot_lines:
                        fig_ta.add_hline(
                            y=level_price,
                            line=dict(color=color, width=width, dash=dash),
                            annotation_text=f"{label} ({month_label})  {level_price:,.0f}",
                            annotation_position="bottom right",
                            annotation_font=dict(size=10, color=color),
                        )

            # ── Drawing 4: ATH & BTC Cycle Levels ─────────────────────────
            # All-time high from the full dataset (not just visible window).
            # BTC cycle peaks are hardcoded historical reference levels:
            #   ~$20k Dec 2017, ~$69k Nov 2021.
            # Non-visible levels are silently skipped to avoid cluttering
            # the chart when zoomed into a range where they don't appear.
            if "ATH & Cycle Levels" in drawings:
                full_hi = df[hi_col] if hi_col in df.columns else df[ta_close]
                ath = float(full_hi.max())

                fig_ta.add_hline(
                    y=ath,
                    line=dict(color='#fbbf24', width=1.5, dash='dashdot'),
                    annotation_text=f"ATH  {ath:,.0f}",
                    annotation_position="top right",
                    annotation_font=dict(size=10, color='#fbbf24'),
                )

                if asset == 'BTC':
                    vis_min = float(ta_df[lo_col].min()) * 0.9
                    vis_max = float(ta_df[hi_col].max()) * 1.1
                    cycle_peaks = [
                        (20000, '2017 Cycle Peak', '#f59e0b'),
                        (69000, '2021 Cycle Peak', '#f59e0b'),
                    ]
                    for level_price, label, color in cycle_peaks:
                        if vis_min <= level_price <= vis_max:
                            fig_ta.add_hline(
                                y=level_price,
                                line=dict(color=color, width=1, dash='dot'),
                                annotation_text=f"{label}  {level_price:,.0f}",
                                annotation_position="top right",
                                annotation_font=dict(size=10, color=color),
                            )

            # ── Layout ─────────────────────────────────────────────────────
            if asset != 'BTC':
                fig_ta.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

            fig_ta.update_layout(
                height=620,
                xaxis_rangeslider_visible=False,
                title=dict(
                    text=f"{ASSET_LABELS.get(asset, asset)} — Technical Analysis",
                    font=dict(size=14, color=TEXT_COLOR), x=0,
                ),
                **DARK_LAYOUT,
            )
            st.plotly_chart(fig_ta, use_container_width=True, config={'displayModeBar': False})

            # ── Legends / explanations ─────────────────────────────────────
            if drawings:
                st.markdown("---")
                legend_map = {
                    "Support & Resistance": (
                        "**Support & Resistance zones**",
                        "Green = support (price is above), Red = resistance (price is below). "
                        "Only pivots touched within the last 12 months are included. "
                        "Zones are scored by recency-weighted touch count (recent touches count more). "
                        "Dashed grey zones = broken (price closed through them 3× recently).",
                    ),
                    "Fibonacci Retracement": (
                        "**Fibonacci Retracement**",
                        "Drawn from the period's swing start to swing end (diamond markers). "
                        "Lines are clipped to start at the earlier swing point. "
                        "Key levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%. "
                        "Direction (uptrend/downtrend) is inferred from which extreme came last.",
                    ),
                    "Monthly Pivots": (
                        "**Monthly Pivot Points**",
                        "Calculated from last month's High, Low, Close: PP = (H+L+C)÷3. "
                        "R1/R2/R3 are resistance levels above PP; S1/S2/S3 are support levels below. "
                        "Monthly timeframe is more meaningful for BTC than weekly pivots.",
                    ),
                    "ATH & Cycle Levels": (
                        "**ATH & Cycle Levels**",
                        "Gold line = all-time high from the full dataset. "
                        "For BTC: previous cycle peaks ($20k Dec 2017, $69k Nov 2021) are shown "
                        "when visible in the selected timeframe. These historical levels often act as "
                        "major support after being broken.",
                    ),
                }
                cols_leg = st.columns(len(drawings))
                for i, drawing in enumerate(drawings):
                    if drawing in legend_map:
                        title, body = legend_map[drawing]
                        with cols_leg[i]:
                            st.markdown(title)
                            st.caption(body)
