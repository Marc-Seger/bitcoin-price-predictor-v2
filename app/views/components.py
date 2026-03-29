"""
Shared design system for all view pages.

Centralizes theme constants (colors, Plotly layout) and the styled_metric KPI card
so they don't need to be redefined in each view file.
"""

import streamlit as st

# ─── Theme colors (match app.py CSS) ───
CHART_BG   = "#0f1520"
GRID_COLOR = "#1e2940"
TEXT_COLOR = "#8899b4"

CARD_COLORS = {
    'blue':    '#3b82f6',
    'emerald': '#10b981',
    'amber':   '#f59e0b',
    'rose':    '#f43f5e',
    'violet':  '#8b5cf6',
    'cyan':    '#06b6d4',
}

ASSET_COLORS = {
    'BTC':    '#f7931a',
    'SP500':  '#3b82f6',
    'NASDAQ': '#06b6d4',
    'GOLD':   '#f59e0b',
    'DXY':    '#8b5cf6',
}

DARK_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor=CHART_BG,
    plot_bgcolor=CHART_BG,
    font=dict(family="JetBrains Mono, monospace", color=TEXT_COLOR, size=11),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    margin=dict(l=50, r=20, t=20, b=30),
    legend=dict(orientation='h', y=1.02, font=dict(size=10)),
)


def styled_metric(label, value, delta=None, color='blue', invert_delta=False):
    """
    KPI card with a colored left border.

    invert_delta: if True, positive delta is red and negative is green
    (used for metrics where higher is worse, e.g. drawdown).
    """
    border_color = CARD_COLORS.get(color, CARD_COLORS['blue'])
    delta_html = ""
    if delta is not None:
        is_positive = str(delta).lstrip().startswith("+")
        if invert_delta:
            delta_color = "#f43f5e" if is_positive else "#10b981"
        else:
            delta_color = "#10b981" if is_positive else "#f43f5e"
        delta_html = (
            f"<div style='font-size:12px;color:{delta_color};"
            f"font-family:JetBrains Mono,monospace;'>{delta}</div>"
        )
    st.markdown(f"""
    <div style="background:#171f30;border:1px solid #263354;border-left:3px solid {border_color};
                border-radius:8px;padding:12px 14px;">
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.8px;
                    color:#56657e;font-weight:600;margin-bottom:4px;">{label}</div>
        <div style="font-size:20px;font-weight:700;color:#e8edf5;
                    font-family:JetBrains Mono,monospace;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)
