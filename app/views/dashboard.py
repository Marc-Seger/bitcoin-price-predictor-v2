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
from components import CARD_COLORS, ASSET_COLORS, DARK_LAYOUT, CHART_BG, styled_metric


@st.cache_data(ttl=300)
def load_data():
    return pd.read_csv(MASTER_DF_PATH, index_col=0, parse_dates=True, low_memory=False)


def render():
    st.markdown("<h1 style='margin-bottom:4px;'>Financial Dashboard</h1>", unsafe_allow_html=True)

    df = load_data()

    # ─── Asset selector (top, controls everything) ───
    ASSET_LABELS = {'BTC': 'Bitcoin (BTC)', 'SP500': 'S&P 500', 'NASDAQ': 'NASDAQ',
                    'GOLD': 'Gold', 'DXY': 'Dollar Index'}
    asset = st.selectbox(
        "Asset", ASSETS, index=0, key="dash_asset",
        format_func=lambda x: ASSET_LABELS.get(x, x),
    )
    close_col = f'Close_{asset}'

    # ─── KPI cards (dynamic per asset, using latest available data) ───
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

    # Format price: $ for USD assets, plain for DXY index
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
        styled_metric("From ATH", f"{ath_dist:+.1%}", color='amber')
    with cols[4]:
        rsi_col = f'RSI_Close_{asset}'
        rsi_val = latest.get(rsi_col) if rsi_col in df.columns else None
        if pd.notna(rsi_val):
            label = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
            styled_metric("RSI (14)", f"{rsi_val:.0f} — {label}", color='violet')
        else:
            styled_metric("RSI (14)", "—", color='violet')

    # ─── Tabs ───
    tab_price, tab_sentiment, tab_onchain, tab_cross = st.tabs([
        "Price & Indicators", "Sentiment", "On-Chain", "Cross-Asset"
    ])

    # ══════════════════════════════════════════
    # TAB 1: Price & Indicators
    # ══════════════════════════════════════════
    with tab_price:
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
                "Subplots", ["RSI", "MACD"], default=[]
            )

        # Row 2: Timeframe + Log scale + Volume toggle
        col_tf, col_log, col_vol = st.columns([1, 0.5, 0.5])
        with col_tf:
            timeframe = st.selectbox("Timeframe", ["1M", "3M", "6M", "1Y", "2Y", "All"], index=3)
        with col_log:
            log_scale = st.checkbox("Log scale", value=False)
        with col_vol:
            show_volume = st.checkbox("Volume", value=True)

        tf_days = {'1M': 30, '3M': 90, '6M': 180, '1Y': 365, '2Y': 730, 'All': len(df)}
        plot_df = df.tail(tf_days[timeframe]).copy()

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
        if show_volume and vol_col in plot_df.columns:
            colors = ['rgba(16,185,129,0.3)' if c >= o else 'rgba(244,63,94,0.3)'
                      for c, o in zip(
                          plot_df[close_col].fillna(0),
                          plot_df.get(open_col, plot_df[close_col]).fillna(0)
                      )]
            fig.add_trace(go.Bar(
                x=plot_df.index, y=plot_df[vol_col],
                name='Volume', marker_color=colors, opacity=0.4,
            ), row=1, col=1, secondary_y=True)

        # ─── Candlestick / line chart (primary y-axis) ───
        high_col = f'High_{asset}'
        low_col = f'Low_{asset}'

        if all(c in plot_df.columns for c in [open_col, high_col, low_col, close_col]):
            fig.add_trace(go.Candlestick(
                x=plot_df.index,
                open=plot_df[open_col], high=plot_df[high_col],
                low=plot_df[low_col], close=plot_df[close_col],
                name=asset, increasing_line_color='#10b981', decreasing_line_color='#f43f5e',
            ), row=1, col=1, secondary_y=False)
        elif close_col in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df[close_col],
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

        # Style volume axis (right side, no grid)
        fig.update_yaxes(
            secondary_y=True, row=1, col=1,
            title_text="Vol", showgrid=False,
            tickfont=dict(size=9, color='#56657e'),
            title_font=dict(size=9, color='#56657e'),
        )

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

        # Log scale for price axis
        if log_scale:
            fig.update_yaxes(type="log", row=1, col=1, secondary_y=False)

        fig.update_layout(
            height=150 + 220 * n_subs,
            xaxis_rangeslider_visible=False,
            **DARK_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════
    # TAB 2: Sentiment
    # ══════════════════════════════════════════
    with tab_sentiment:
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

                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fg_val,
                    title={'text': fg_label, 'font': {'size': 16, 'color': fg_color}},
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
                gauge_fig.update_layout(height=220, **{k: v for k, v in DARK_LAYOUT.items() if k != 'legend'})
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
    # TAB 3: On-Chain
    # ══════════════════════════════════════════
    with tab_onchain:
        oc_tf = st.selectbox("Period", ["3M", "6M", "1Y", "2Y", "All"], index=2, key="oc_tf")
        oc_n = {'3M': 90, '6M': 180, '1Y': 365, '2Y': 730, 'All': len(df)}[oc_tf]
        st.markdown("### On-Chain Metrics")

        onchain_metrics = {
            'OnChain_Hash_Rate': ('Hash Rate', 'Network security — higher = more miners', 'blue'),
            'OnChain_Active_Addresses': ('Active Addresses', 'Daily unique addresses — user activity proxy', 'emerald'),
            'OnChain_MVRV_Ratio': ('MVRV Ratio', '>3.5 = historically overbought, <1 = undervalued', 'amber'),
            'OnChain_Transaction_Count': ('Transactions', 'Daily on-chain transaction count', 'violet'),
            'OnChain_30d_ROI': ('30-Day ROI', 'Rolling 30-day return on investment', 'rose'),
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
    with tab_cross:
        st.markdown("### Cross-Asset Comparison")

        timeframe_cross = st.selectbox("Period", ["3M", "6M", "1Y", "2Y"], index=2, key="cross_tf")
        days_cross = {'3M': 90, '6M': 180, '1Y': 365, '2Y': 730}
        cross_df = df.tail(days_cross[timeframe_cross])

        # Normalized performance
        st.markdown("**Normalized Performance (% change from start)**")
        perf_data = pd.DataFrame()
        for a in ASSETS:
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

        # Correlation heatmap
        st.markdown(f"**{timeframe_cross} Return Correlation**")
        close_cols = {a: f'Close_{a}' for a in ASSETS if f'Close_{a}' in cross_df.columns}
        if len(close_cols) > 1:
            returns = pd.DataFrame({a: cross_df[c].pct_change() for a, c in close_cols.items()})
            corr = returns.corr()

            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale=[[0, '#f43f5e'], [0.5, '#0f1520'], [1, '#3b82f6']],
                zmid=0, zmin=-1, zmax=1,
                text=corr.values.round(2), texttemplate='%{text}',
                textfont={"size": 13, "color": "#e8edf5"},
            ))
            fig_corr.update_layout(height=350, **DARK_LAYOUT)
            st.plotly_chart(fig_corr, use_container_width=True)
