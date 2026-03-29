"""
Bitcoin Price Predictor — Streamlit App
"""

import os
import sys
import requests
import streamlit as st

# Make app/views/ importable so views can do `from components import ...`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'views'))

# ─── Ensure master_df.csv exists (download from GitHub Release if running on Streamlit Cloud) ───
_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'full_data', 'master_df.csv')
_RELEASE_URL = 'https://github.com/Marc-Seger/bitcoin-price-predictor-v2/releases/download/latest/master_df.csv'

if not os.path.exists(_DATA_PATH):
    os.makedirs(os.path.dirname(_DATA_PATH), exist_ok=True)
    with st.spinner('Downloading data (first run only, ~30s)...'):
        r = requests.get(_RELEASE_URL, timeout=120)
        r.raise_for_status()
        with open(_DATA_PATH, 'wb') as f:
            f.write(r.content)

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
    min-width: 220px !important;
    max-width: 220px !important;
}
section[data-testid="stSidebar"] > div:first-child {
    width: 220px !important;
}
/* Hide the radio group label */
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    display: none !important;
}

/* Nav menu — only target option labels (contain a radio input), not the group label */
section[data-testid="stSidebar"] .stRadio > div > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 3px !important;
}
section[data-testid="stSidebar"] .stRadio label:has(input) {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    border: 1px solid transparent !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: var(--t2) !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
    margin: 0 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
section[data-testid="stSidebar"] .stRadio label:has(input):hover {
    background: var(--bg2) !important;
    color: var(--t1) !important;
    border-color: var(--brd) !important;
}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: rgba(59,130,246,0.12) !important;
    border-color: rgba(59,130,246,0.35) !important;
    color: var(--blue) !important;
    font-weight: 600 !important;
}
/* Collapse radio circle and input to zero size — keeps input clickable unlike display:none */
section[data-testid="stSidebar"] .stRadio label:has(input) > *:not(:last-child) {
    position: absolute !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    pointer-events: none !important;
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

PAGES = ["Dashboard", "Forecast", "Strategy Lab", "Model Performance", "Documentation"]

# Persist selected page across refreshes via URL query param
_qp = st.query_params.get("page", "Dashboard")
_default_idx = PAGES.index(_qp) if _qp in PAGES else 0

page = st.sidebar.radio(
    "",
    PAGES,
    index=_default_idx,
    label_visibility="collapsed",
    key="nav_page",
)
st.query_params["page"] = page

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
