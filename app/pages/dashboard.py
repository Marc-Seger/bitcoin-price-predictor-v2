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
from config import MASTER_DF_PATH, ASSETS


@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(MASTER_DF_PATH, index_col=0, parse_dates=True, low_memory=False)
    return df


def render():
    st.title("Financial Dashboard")

    df = load_data()

    # ─── Key metrics cards ───
    latest = df.dropna(subset=['Close_BTC']).iloc[-1]
    prev_7d = df.dropna(subset=['Close_BTC']).iloc[-8] if len(df) > 8 else latest
    prev_30d = df.dropna(subset=['Close_BTC']).iloc[-31] if len(df) > 31 else latest
    ath = df['Close_BTC'].max()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("BTC Price", f"${latest['Close_BTC']:,.0f}")
    with col2:
        chg_7d = (latest['Close_BTC'] / prev_7d['Close_BTC'] - 1)
        st.metric("7d Change", f"{chg_7d:+.1%}")
    with col3:
        chg_30d = (latest['Close_BTC'] / prev_30d['Close_BTC'] - 1)
        st.metric("30d Change", f"{chg_30d:+.1%}")
    with col4:
        ath_dist = (latest['Close_BTC'] / ath - 1)
        st.metric("From ATH", f"{ath_dist:+.1%}")
    with col5:
        rsi_val = latest.get('RSI_Close_BTC', None)
        if pd.notna(rsi_val):
            label = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
            st.metric("RSI (14)", f"{rsi_val:.0f} — {label}")
        else:
            st.metric("RSI (14)", "—")

    st.markdown("---")

    # ─── Tab layout ───
    tab_price, tab_sentiment, tab_onchain, tab_cross = st.tabs([
        "Price & Indicators", "Sentiment", "On-Chain", "Cross-Asset"
    ])

    # ══════════════════════════════════════════
    # TAB 1: Price & Indicators
    # ══════════════════════════════════════════
    with tab_price:
        # Controls
        col_tf, col_indicators = st.columns([1, 3])

        with col_tf:
            timeframe = st.selectbox("Timeframe", ["1M", "3M", "6M", "1Y", "2Y", "All"], index=3)
            tf_days = {'1M': 30, '3M': 90, '6M': 180, '1Y': 365, '2Y': 730, 'All': len(df)}
            plot_df = df.tail(tf_days[timeframe]).copy()

        with col_indicators:
            overlays = st.multiselect(
                "Overlays",
                ["SMA 9", "SMA 20", "SMA 50", "SMA 200", "Bollinger Bands"],
                default=["SMA 50", "SMA 200"]
            )
            subplots_on = st.multiselect(
                "Subplots",
                ["Volume", "RSI", "MACD"],
                default=["Volume"]
            )

        # Build the chart
        n_subplots = 1 + len(subplots_on)
        row_heights = [0.5] + [0.2] * len(subplots_on) if subplots_on else [1.0]
        # Normalize heights
        total = sum(row_heights)
        row_heights = [h / total for h in row_heights]

        fig = make_subplots(
            rows=n_subplots, cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=row_heights,
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=plot_df.index,
            open=plot_df['Open_BTC'], high=plot_df['High_BTC'],
            low=plot_df['Low_BTC'], close=plot_df['Close_BTC'],
            name='BTC', increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        ), row=1, col=1)

        # Overlays
        sma_colors = {'SMA 9': '#ff9800', 'SMA 20': '#2196f3', 'SMA 50': '#9c27b0', 'SMA 200': '#f44336'}
        for overlay in overlays:
            if overlay.startswith('SMA'):
                window = int(overlay.split(' ')[1])
                col_name = f'SMA_{window}_Close_BTC'
                if col_name in plot_df.columns:
                    fig.add_trace(go.Scatter(
                        x=plot_df.index, y=plot_df[col_name],
                        name=overlay, line=dict(width=1, color=sma_colors.get(overlay, 'gray')),
                    ), row=1, col=1)

            elif overlay == 'Bollinger Bands':
                upper = 'Upper_Band_Close_BTC'
                lower = 'Lower_Band_Close_BTC'
                if upper in plot_df.columns and lower in plot_df.columns:
                    fig.add_trace(go.Scatter(
                        x=plot_df.index, y=plot_df[upper], name='BB Upper',
                        line=dict(width=1, color='rgba(150,150,150,0.5)'),
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=plot_df.index, y=plot_df[lower], name='BB Lower',
                        line=dict(width=1, color='rgba(150,150,150,0.5)'),
                        fill='tonexty', fillcolor='rgba(150,150,150,0.1)',
                    ), row=1, col=1)

        # Subplots
        subplot_row = 2
        for subplot in subplots_on:
            if subplot == 'Volume' and 'Volume_BTC' in plot_df.columns:
                colors = ['#26a69a' if c >= o else '#ef5350'
                          for c, o in zip(plot_df['Close_BTC'], plot_df['Open_BTC'])]
                fig.add_trace(go.Bar(
                    x=plot_df.index, y=plot_df['Volume_BTC'],
                    name='Volume', marker_color=colors, opacity=0.7,
                ), row=subplot_row, col=1)
                fig.update_yaxes(title_text="Volume", row=subplot_row, col=1)
                subplot_row += 1

            elif subplot == 'RSI' and 'RSI_Close_BTC' in plot_df.columns:
                fig.add_trace(go.Scatter(
                    x=plot_df.index, y=plot_df['RSI_Close_BTC'],
                    name='RSI', line=dict(color='#7e57c2', width=1.5),
                ), row=subplot_row, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red",
                              opacity=0.5, row=subplot_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green",
                              opacity=0.5, row=subplot_row, col=1)
                fig.update_yaxes(title_text="RSI", range=[0, 100], row=subplot_row, col=1)
                subplot_row += 1

            elif subplot == 'MACD' and 'MACD_D_BTC' in plot_df.columns:
                fig.add_trace(go.Scatter(
                    x=plot_df.index, y=plot_df['MACD_D_BTC'],
                    name='MACD', line=dict(color='#2196f3', width=1.5),
                ), row=subplot_row, col=1)
                fig.add_trace(go.Scatter(
                    x=plot_df.index, y=plot_df['Signal_Line_D_BTC'],
                    name='Signal', line=dict(color='#ff9800', width=1.5),
                ), row=subplot_row, col=1)
                hist = plot_df['MACD_Histogram_D_BTC']
                colors = ['#26a69a' if v >= 0 else '#ef5350' for v in hist]
                fig.add_trace(go.Bar(
                    x=plot_df.index, y=hist,
                    name='Histogram', marker_color=colors, opacity=0.5,
                ), row=subplot_row, col=1)
                fig.update_yaxes(title_text="MACD", row=subplot_row, col=1)
                subplot_row += 1

        fig.update_layout(
            height=200 + 300 * n_subplots, xaxis_rangeslider_visible=False,
            template='plotly_dark', margin=dict(l=50, r=20, t=30, b=30),
            legend=dict(orientation='h', y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════
    # TAB 2: Sentiment
    # ══════════════════════════════════════════
    with tab_sentiment:
        col_gauge, col_timeline = st.columns([1, 2])

        with col_gauge:
            st.subheader("Fear & Greed Index")
            fg_col = 'Sentiment_BTC_index_value'
            if fg_col in df.columns:
                fg_val = df[fg_col].dropna().iloc[-1]
                fg_label_map = {
                    range(0, 25): ("Extreme Fear", "#ef5350"),
                    range(25, 45): ("Fear", "#ff9800"),
                    range(45, 55): ("Neutral", "#ffeb3b"),
                    range(55, 75): ("Greed", "#66bb6a"),
                    range(75, 101): ("Extreme Greed", "#26a69a"),
                }
                fg_label, fg_color = "Unknown", "gray"
                for rng, (lbl, clr) in fg_label_map.items():
                    if int(fg_val) in rng:
                        fg_label, fg_color = lbl, clr
                        break

                # Gauge chart
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fg_val,
                    title={'text': fg_label, 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': fg_color},
                        'steps': [
                            {'range': [0, 25], 'color': 'rgba(239,83,80,0.3)'},
                            {'range': [25, 45], 'color': 'rgba(255,152,0,0.3)'},
                            {'range': [45, 55], 'color': 'rgba(255,235,59,0.3)'},
                            {'range': [55, 75], 'color': 'rgba(102,187,106,0.3)'},
                            {'range': [75, 100], 'color': 'rgba(38,166,154,0.3)'},
                        ],
                    },
                ))
                gauge_fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=10),
                                         template='plotly_dark')
                st.plotly_chart(gauge_fig, use_container_width=True)
            else:
                st.info("Fear & Greed data not available.")

        with col_timeline:
            st.subheader("Fear & Greed Over Time")
            if fg_col in df.columns:
                fg_data = df[fg_col].dropna().tail(365)
                fig_fg = go.Figure()
                fig_fg.add_trace(go.Scatter(
                    x=fg_data.index, y=fg_data.values,
                    fill='tozeroy', line=dict(color='#7e57c2'),
                    name='Fear & Greed',
                ))
                fig_fg.add_hline(y=25, line_dash="dash", line_color="red", opacity=0.5)
                fig_fg.add_hline(y=75, line_dash="dash", line_color="green", opacity=0.5)
                fig_fg.update_layout(height=300, template='plotly_dark',
                                      margin=dict(l=50, r=20, t=10, b=30),
                                      yaxis_range=[0, 100])
                st.plotly_chart(fig_fg, use_container_width=True)

    # ══════════════════════════════════════════
    # TAB 3: On-Chain
    # ══════════════════════════════════════════
    with tab_onchain:
        st.subheader("On-Chain Metrics")

        onchain_metrics = {
            'OnChain_Hash_Rate': ('Hash Rate', 'Network security — higher = more miners = more secure'),
            'OnChain_Active_Addresses': ('Active Addresses', 'Daily unique addresses — proxy for user activity'),
            'OnChain_MVRV_Ratio': ('MVRV Ratio', 'Market Value / Realized Value — >3.5 historically signals top, <1 signals bottom'),
            'OnChain_Transaction_Count': ('Transaction Count', 'Daily on-chain transactions'),
            'OnChain_30d_ROI': ('30-Day ROI', 'Rolling 30-day return on investment'),
        }

        for col_name, (display_name, description) in onchain_metrics.items():
            if col_name not in df.columns:
                continue
            data = df[col_name].dropna().tail(365)
            if data.empty:
                continue

            st.markdown(f"**{display_name}**")
            st.caption(description)

            current = data.iloc[-1]
            prev = data.iloc[-8] if len(data) > 8 else data.iloc[0]
            change = (current / prev - 1) if prev != 0 else 0

            col_val, col_chart = st.columns([1, 3])
            with col_val:
                st.metric(display_name, f"{current:,.0f}", f"{change:+.1%} (7d)")
            with col_chart:
                st.line_chart(data)

    # ══════════════════════════════════════════
    # TAB 4: Cross-Asset
    # ══════════════════════════════════════════
    with tab_cross:
        st.subheader("Cross-Asset Comparison")

        timeframe_cross = st.selectbox("Period", ["3M", "6M", "1Y", "2Y"], index=2,
                                        key="cross_tf")
        days_cross = {'3M': 90, '6M': 180, '1Y': 365, '2Y': 730}
        cross_df = df.tail(days_cross[timeframe_cross])

        # Normalized performance chart
        st.markdown("**Normalized Performance (% change from start)**")
        perf_data = pd.DataFrame()
        for asset in ASSETS:
            close_col = f'Close_{asset}'
            if close_col in cross_df.columns:
                series = cross_df[close_col].dropna()
                if len(series) > 0:
                    perf_data[asset] = (series / series.iloc[0] - 1) * 100

        if not perf_data.empty:
            fig_perf = go.Figure()
            colors = {'BTC': '#f7931a', 'SP500': '#2196f3', 'NASDAQ': '#00bcd4',
                      'GOLD': '#ffd700', 'DXY': '#9e9e9e'}
            for asset in perf_data.columns:
                fig_perf.add_trace(go.Scatter(
                    x=perf_data.index, y=perf_data[asset],
                    name=asset, line=dict(color=colors.get(asset, 'white'), width=2),
                ))
            fig_perf.update_layout(
                height=400, template='plotly_dark',
                yaxis_title="% Change", margin=dict(l=50, r=20, t=10, b=30),
                legend=dict(orientation='h', y=1.05),
            )
            st.plotly_chart(fig_perf, use_container_width=True)

        # Correlation heatmap
        st.markdown("**Rolling 30-Day Correlation**")
        close_cols = {a: f'Close_{a}' for a in ASSETS if f'Close_{a}' in cross_df.columns}
        if len(close_cols) > 1:
            returns = pd.DataFrame()
            for asset, col in close_cols.items():
                returns[asset] = cross_df[col].pct_change()

            corr = returns.tail(30).corr()

            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
                text=corr.values.round(2), texttemplate='%{text}',
                textfont={"size": 14},
            ))
            fig_corr.update_layout(height=400, template='plotly_dark',
                                    margin=dict(l=50, r=20, t=10, b=30))
            st.plotly_chart(fig_corr, use_container_width=True)
