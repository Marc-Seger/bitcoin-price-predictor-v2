"""
Bitcoin Price Predictor — Streamlit App
"""

import streamlit as st

st.set_page_config(
    page_title="BTC Price Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom dark theme CSS (inspired by shortfall analyzer) ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg0: #070b12; --bg1: #0f1520; --bg2: #171f30; --bg3: #1e2940;
    --brd: #263354; --brd2: #334572;
    --t1: #e8edf5; --t2: #8899b4; --t3: #56657e;
    --blue: #3b82f6; --cyan: #06b6d4; --emerald: #10b981;
    --amber: #f59e0b; --rose: #f43f5e; --violet: #8b5cf6;
}

/* Main background */
.stApp { background-color: var(--bg0) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg1) !important;
    border-right: 1px solid var(--brd) !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
}

/* KPI cards */
div[data-testid="stMetric"] {
    background: var(--bg2);
    border: 1px solid var(--brd);
    border-radius: 10px;
    padding: 12px 16px;
}
div[data-testid="stMetric"] label {
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--t3) !important;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 20px !important;
    font-weight: 700 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}

/* Dataframes */
.stDataFrame { font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; }

/* Headers */
h1, h2, h3 { font-family: 'DM Sans', sans-serif !important; }
h1 { font-size: 22px !important; font-weight: 700 !important; }
h2 { font-size: 16px !important; font-weight: 600 !important; }
h3 { font-size: 14px !important; font-weight: 600 !important; }

/* General text */
p, li, span { font-family: 'DM Sans', sans-serif !important; }

/* Selectbox / multiselect */
div[data-baseweb="select"] { font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Reduce top padding */
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───
st.sidebar.markdown(
    "<h1 style='font-family: JetBrains Mono, monospace; font-size: 15px; "
    "font-weight: 700; color: #3b82f6; margin-bottom: 2px;'>BTC Price Predictor</h1>"
    "<p style='font-size: 10px; color: #56657e; font-family: JetBrains Mono, monospace;'>"
    "ML-powered Bitcoin analysis & forecasting</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Forecast", "Strategy Lab", "Model Performance", "Documentation"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Built by [Marc Seger](https://github.com/Marc-Seger)  \n"
    "XGBoost · Walk-forward validation  \n"
    "Not financial advice."
)

# ─── Page routing ───
if page == "Dashboard":
    from views.dashboard import render
elif page == "Forecast":
    from views.forecast import render
elif page == "Strategy Lab":
    from views.strategy_lab import render
elif page == "Model Performance":
    from views.performance import render
elif page == "Documentation":
    from views.documentation import render

render()
