# Bitcoin Price Predictor v2 — Claude Briefing

## Project Summary
Marc is a data analyst (post-bootcamp) building a Bitcoin-focused data/ML portfolio for job hunting.
This is the main active project: a live market dashboard + 7-day price predictor app built with XGBoost, Streamlit, and automated daily data pipelines via GitHub Actions.

- GitHub: https://github.com/Marc-Seger/bitcoin-price-predictor-v2
- Stack: Python 3.11, XGBoost, Streamlit, Plotly, Optuna, GitHub Actions
- Local path: `/Users/marcseger/Code/GitHubMarc/bitcoin-price-predictor-v2/`

---

## Current Phase
**Phase 5 complete. Walk-forward eval with tuned model in progress. Pending: upload master_df, deploy to Streamlit Cloud, build monitoring workflow.**

---

## Completed Session 8 (2026-04-03)

### Production Model Retrained with Tuned Params
- `scripts/train_production.py` updated to load `xgboost_best_params_v2.json` (was pointing to v1)
- Production model retrained: 2,967 rows, 52 features, 1.8 MB
- Params: n_estimators=400, max_depth=8, lr=0.072, subsample=0.635, colsample_bytree=0.761, min_child_weight=8

### Walk-Forward Eval Running (in progress)
- Script: `/tmp/regen_xgb_7d.py` — XGBoost walk-forward on TARGET_7D, 2,467 predictions, tuned v2 params
- Output: `results/XGB_7d_walkforward_results.csv` (overwrites old baseline-params version)
- Performance tab reads this file dynamically — no code changes needed once eval finishes
- **Background task still running when session paused** — check output on resume

### Trade Log Cleared
- `app/data/trade_log.csv` — baseline-model prediction removed, header kept
- Ready for fresh predictions from tuned model

### Phase 7 (Power BI) Scratched
- Marc confirmed: Power BI work is cancelled. Remove from plan entirely.

### Monitoring Workflow Planned
- Marc wants a GitHub Actions weekly job that monitors model health and opens a GitHub Issue when action is needed
- Signals: rolling 30-day direction accuracy vs naive baseline, days since last retune (>90d threshold), pipeline failures
- GitHub auto-emails Marc when issues are created — no manual checking needed
- **Build this immediately after walk-forward eval completes**

---

## Completed Session 7 (2026-03-29)

### Dashboard Tab — Final Polish
All visual issues resolved this session. App is now production-ready pending model retrain.

- **Sidebar first-click bug fixed**: radio `<input>` hidden via `position:absolute; opacity:0; width:0; height:0; pointer-events:none`
- **Page persistence on refresh**: `st.query_params["page"]` used to write/read current page in URL
- **"Navigate" label removed**: `[data-testid="stWidgetLabel"] { display: none }`
- **Forecast page**: drift message moved to bottom. KPI cards `min-height:90px`. Delta color fix: `'+' in str(delta)`
- **Strategy Lab**: backtest period banner moved to bottom
- **Model Performance**: comparison table as KPI-style cards with colored left borders

---

## Completed Session 6 (2026-03-29)

### Dashboard Tab — Major Polish Pass

- Asset selector inside Price & Indicators tab; KPI cards read from `st.session_state`
- Full dataset in chart with initial visible range; toolbar pan/zoom reveals history
- Weekly resampling for dense timeframes (>600 points) using `resample('W-FRI')`
- Volume scaling at 95th percentile of visible window; `opacity=0.4`; `barmode='overlay'`
- Y-axis range fitted to visible window with 5% padding; log scale fixed
- RSI on by default; Plotly modebar hidden; weekend rangebreaks for non-BTC assets
- Fear & Greed label moved to `st.markdown` above gauge; height 200px

---

## Completed Session 5 (2026-03-29)

### Data Pipeline — Now Complete
- Dataset extended to 2017 via `scripts/backfill_history.py` (3,375 rows)
- Non-BTC price fix: real index tickers (^GSPC, ^IXIC, GC=F, DX-Y.NYB)
- Weekend NaN fix: non-BTC OHLCV forward-fill in `merge.py`
- ETF flows automated: `fetch_etf_flows()` scrapes farside.co.uk
- Google Trends backfill: two-window rescaling with overlap-based scale factor (~2.12)
- Honest model metrics: R²=-0.056, Direction=55.2% vs 52.3% naive (pre-tuning baseline)

---

## Full Phase Plan

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Cleanup & folder organization | Complete |
| 1 | Data audit | Complete |
| 2 | Data pipeline automation | Complete |
| 3 | Model audit & re-run | Complete |
| 4 | Model improvement (tuning) | Complete — best R²=0.5690 (v2 params) |
| 5 | Streamlit predictor app | Complete — app built, polished, model retrained |
| 6 | Deploy + monitoring | **In progress** |
| 7 | Dashboard tab improvements | Not started (optional backlog) |
| 8 | Portfolio website | Not started |

**Phase 7 (Power BI) permanently cancelled.**

---

## What's Next

### Immediate (resume here)
1. **Check walk-forward eval** — read `/private/tmp/claude-501/-Users-marcseger-Code-GitHubMarc-bitcoin-price-predictor-v2/a0fbe96c-3db1-436d-9d3b-585bbe8326b0/tasks/byja7decj.output` or just run fresh if task died:
   `/opt/homebrew/bin/python3.11 /tmp/regen_xgb_7d.py`
   (script is at `/tmp/regen_xgb_7d.py` — may not persist after reboot, recreate if needed)
2. **Build GitHub Actions monitoring workflow** — weekly job, opens GitHub Issue when:
   - Rolling 30-day direction accuracy < naive baseline
   - Days since last retune > 90
   - Daily pipeline has been failing
3. **Upload master_df to GitHub Release**: `bash scripts/setup_github_release.sh`
4. **Deploy to Streamlit Community Cloud**

### After deployment (optional backlog)
- Dashboard tab improvements (discuss before implementing):
  - Market summary strip: "BTC -5.9% this week | Fear & Greed: 39 (Fear) | RSI: Neutral"
  - Multi-asset KPI row: mini cards for all assets with 7d change
  - Key levels: SMA 200 price, recent high/low
  - Market regime badge: Bull/Bear/Consolidation from SMA crossovers
- Portfolio website (Phase 8): update links to v2 app

---

## Key Decisions

### Architecture
- **No Jupyter notebooks** — all logic in `.py` modules and scripts
- **bitcoin-market-dashboard** is an official archive — never modify again
- **bitcoin-price-predictor-v2** is the ONE app going forward
- **Actual project structure:**
  ```
  bitcoin-price-predictor-v2/
  ├── src/
  │   ├── config.py            # shared constants (paths, features, assets)
  │   ├── utils.py             # shared functions (load_featured_data, compute_metrics)
  │   ├── data/
  │   │   ├── fetch.py         # yfinance, FRED, Alt.me, pytrends, CoinMetrics, ETF flows
  │   │   ├── merge.py         # combine sources → master_df (with weekend forward-fill)
  │   │   └── features.py      # 269 technical indicators, targets
  │   └── models/
  │       ├── evaluate.py      # walk-forward eval for RF/XGBoost (1d target, baseline — not used in app)
  │       └── evaluate_dl.py   # expanding-window eval for LSTM/GRU
  ├── scripts/
  │   ├── update_data.py       # full data pipeline (fetch → merge → features → save)
  │   ├── backfill_history.py  # one-time: rebuild master_df from 2017 with GT + ETF flows
  │   ├── train_production.py  # train XGBoost on all data, save joblib (uses xgboost_best_params_v2.json)
  │   ├── tune_xgboost.py      # Optuna tuning (100 trials, resumable)
  │   └── setup_github_release.sh  # upload master_df to GitHub release
  ├── app/
  │   ├── app.py               # Streamlit entry point (CSS + routing)
  │   ├── model/
  │   │   └── predict.py       # prediction engine, paper trading, drift detection
  │   └── views/               # NOT pages/ (avoids Streamlit auto-detection)
  │       ├── dashboard.py     # financial dashboard (5 tabs)
  │       ├── forecast.py      # 7-day prediction
  │       ├── strategy_lab.py  # backtesting with leverage/SL/TP
  │       ├── performance.py   # model evaluation results
  │       └── documentation.py # how-to, data sources, methodology, limitations
  ├── data/                    # gitignored — master_df.csv stored here locally
  ├── models/                  # gitignored — xgboost_production.joblib
  ├── results/                 # metrics CSVs, walk-forward results, tuning DB
  ├── .github/workflows/       # daily_update.yml (07:00 UTC)
  ├── requirements.txt
  └── README.md
  ```

### GitHub Strategy
- Keep old repos as-is ("before" narrative for job hunting)
- master_df.csv stored as GitHub Release asset (too large for repo)
- GitHub Actions: daily 07:00 UTC — retrains model daily on latest data
- Tuning improvements: commit updated `xgboost_best_params_v2.json` → Actions picks up automatically

### Data Sources (free only)
**In use (fully automated):**
- BTC + assets (SP500, NASDAQ, Gold, DXY): `yfinance` (real index tickers: ^GSPC, ^IXIC, GC=F, DX-Y.NYB)
- Macro data (CPI, rates, M2, treasury yield, unemployment): FRED API
- BTC Fear & Greed: Alternative.me free API
- Google Trends: `pytrends`
- On-chain (5 metrics): CoinMetrics free community API
- ETF flows: farside.co.uk scraping (`ETF_Flow_Total`)

### Model
- Target: 7-day return (not absolute price — tree models can't extrapolate)
- Features: 52 curated from 269 (RF importance + analyst review)
- Evaluation: walk-forward, non-overlapping 7-day windows
- Tuned metrics (v2 params): R²=0.5690 (Optuna walk-forward score)
- Pre-tuning baseline: R²=-0.056, Direction=55.2% vs 52.3% naive
- Confidence levels: >5% = HIGH, >2% = MEDIUM, else LOW
- LSTM/GRU tried and failed — insufficient data (~3k rows) for neural nets
- `results/XGB_7d_walkforward_results.csv` — used by Performance tab (computed dynamically)
- `src/models/evaluate.py` — uses TARGET_1D with baseline params, not relevant to app

### Long-Term Maintenance Flow
- **Daily**: fully autonomous (GitHub Actions: fetch → retrain → predict → commit)
- **Every ~90 days**: retune with `tune_xgboost.py`, commit updated params JSON, push
- **When to retune early**: significant market regime shift, or monitoring workflow flags accuracy drop
- **Monitoring**: weekly GitHub Actions job opens Issues when thresholds breached

---

## Known Issues
- None blocking — production model is now tuned, trade log is clean

---

## Local Environment
- Python: 3.11 at `/opt/homebrew/bin/python3.11` (Apple Silicon / arm64)
- FRED API key: stored in `.env` in this project root
- TensorFlow: 2.16.2 (CPU only, tensorflow-metal uninstalled due to conflict)
- numpy: 1.26.4 (downgraded by TF from 2.4.3)

---

## How to Resume
Start each session with:
> "Read the CLAUDE.md in /Users/marcseger/Code/GitHubMarc/bitcoin-price-predictor-v2 and pick up where we left off"
