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
    last_date = valid.index[-1]
    _rows_7d  = valid[valid.index <= last_date - pd.Timedelta(days=7)]
    _rows_30d = valid[valid.index <= last_date - pd.Timedelta(days=30)]
    prev_7d  = _rows_7d[close_col].iloc[-1]  if not _rows_7d.empty  else price
    prev_30d = _rows_30d[close_col].iloc[-1] if not _rows_30d.empty else price
    ath = df[close_col].max()

    st.caption(f"Daily close for {valid.index[-1].strftime('%d/%m/%Y')}")

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
    # "Technical Analysis" retired 2026-08-13 (S/R zones, Fibonacci, pivots, ATH
    # levels). Chart-drawing tools rather than analysis, and off-message for a
    # portfolio aimed at product/analyst roles. The full block is recoverable
    # from git — see CLAUDE.md for the commit and how to restore it.
    tab_names += ["Cross-Asset"]
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
        etf_col = COL_ETF_FLOW
        if etf_col in df.columns:
            # dropna() here means the chart plots the last N *available* values, which
            # silently renders months-old data as if it were current once a source dies.
            # The farside.co.uk scraper has been returning 403 since ~April 2026 (the site
            # blocks datacenter IPs, so it fails from GitHub Actions but works from a
            # laptop). Say so rather than quietly showing stale bars.
            etf_series = df[etf_col].dropna()
            last_etf = etf_series.index.max() if not etf_series.empty else None
            stale_days = (df.index.max() - last_etf).days if last_etf is not None else None

            if stale_days is not None and stale_days > 7:
                st.warning(
                    f"**This chart stopped updating on {last_etf.strftime('%d %B %Y')}** "
                    f"({stale_days} days ago). The upstream source, farside.co.uk, blocks "
                    f"datacenter IP ranges, so the daily fetch returns 403 from GitHub "
                    f"Actions. Everything below is historical. ETF flows are not a model "
                    f"feature, so predictions are unaffected."
                )
            st.caption(
                "Net institutional inflow/outflow across all spot Bitcoin ETFs ($M). "
                "Available from Jan 2024."
            )

            etf_data = etf_series.tail(sent_n)
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
                _oc_last = data.index[-1]
                _oc_7d = data[data.index <= _oc_last - pd.Timedelta(days=7)]
                prev = _oc_7d.iloc[-1] if not _oc_7d.empty else data.iloc[0]
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

        timeframe_cross = st.selectbox("Period", ["3M", "6M", "1Y", "2Y", "All history"], index=2, key="cross_tf")
        days_cross = {'3M': 90, '6M': 180, '1Y': 365, '2Y': 730, 'All history': len(df)}
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
            fig_perf.update_layout(height=350, yaxis_title="% Change", **{**DARK_LAYOUT, 'legend': dict(orientation='h', y=-0.12, yanchor='top', font=dict(size=10))})
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
                height=260, yaxis_title="Annualised Vol %",
                **{**DARK_LAYOUT,
                   'legend': dict(orientation='h', y=-0.15, yanchor='top', font=dict(size=10)),
                   'margin': dict(l=50, r=20, t=20, b=50)}
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
                **{**DARK_LAYOUT,
                   'legend': dict(orientation='h', y=-0.15, yanchor='top', font=dict(size=10)),
                   'margin': dict(l=50, r=20, t=20, b=50)}
            )
            st.plotly_chart(fig_dd, use_container_width=True)
