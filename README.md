# Bitcoin Price Predictor v2

An autonomous machine-learning pipeline that forecasts Bitcoin's 7-day return, grades its own
past calls, and publishes the record — including where it gets things wrong.

**[Live app →](https://bitcoin-price-predictor-v2.streamlit.app)**  ·  Python · XGBoost · Streamlit · GitHub Actions

---

## The headline, stated up front

**This model has no demonstrated edge.**

| | |
|---|---|
| Leak-free backtest, direction accuracy | **48.5%** |
| R² | **−0.308** |
| Correlation (prediction, outcome) | **−0.030** |
| Independent windows | 367 (2019-06 → 2026-06) |
| Naive baseline (always predict UP) | **52.6%** |
| Live production log | ~100 resolved predictions, **~49%** |

An earlier version of this README and the app reported **76.7% direction accuracy**. That figure
was wrong. It survived for months because the documents describing it copied each other instead
of re-deriving it, and because the monitoring built to catch exactly this had itself silently
broken. Both problems are described below, because they are the more interesting part of the
project.

## What went wrong, and how it was found

Two errors, both in the evaluation rather than the model.

**1. Overlapping windows counted as independent.** 2,467 daily predictions were reported as
"non-overlapping". They were not: consecutive 7-day windows share six of their seven days, so one
correct multi-week call was counted up to seven times. The genuinely independent count is 367.

**2. Target leakage — the dominant effect.** The target is
`Target_Return_7d = close.shift(-7) / close - 1`. The walk-forward loop trained on every row up to
the prediction date, which meant the last six training rows carried targets computed from prices
*after* that date. The model was being trained on the answer.

Removing the leak takes direction accuracy from 73.3% to **48.5%**, and the correlation between
prediction and outcome from 0.77 to −0.03. Not a weaker edge — no edge.

Two independent measurements agree on this. The corrected backtest says 48.5%; the live
production log, which has recorded a prediction before its outcome was known since March 2026,
says ~49%. Reproduce with [`scripts/evaluate_leakfree.py`](scripts/evaluate_leakfree.py).

**The same leak invalidated two other results** that had been believed: that the model was more
accurate on large predicted moves (leak-free, the largest-move bucket is the *worst* of the
three), and that tree models beat LSTM/GRU here (the deep-learning evaluation had correctly
purged the leaking rows, so the comparison was never like for like).

## Why nobody noticed for five months

A weekly job was supposed to compare live accuracy against the naive baseline and open a GitHub
issue when it slipped. It reported "all clear" with a green checkmark every week from April to
August while the model sat at a coin flip.

A one-off manual repair to the trade log left a single date in a different format from the rest.
The file read threw, a broad `except` caught it, and the failure was recorded as an informational
note rather than an alert. **A check that could not run was being reported as the thing it checked
passing.**

That is the actual lesson of this project, and it was not the date format. It was that the
monitoring could not distinguish "I checked and everything is fine" from "I could not check".
Every check now raises an alert when it cannot complete.

## What does work

The engineering around the model is sound, and it is what the project is actually worth showing:

- **Fully autonomous.** Every day at 22:00 UTC, after the US market close, a GitHub Actions
  workflow fetches data, validates it, retrains, predicts, logs and commits. No manual step.
- **Six data-quality checks gate every run.** Currency, frozen-value detection (a dead source plus
  forward-fill looks identical to a flat market), value ranges, feature completeness, date
  continuity, and API contract logging. A failure stops the pipeline, opens an issue, and leaves
  yesterday's model in place.
- **Predictions are logged before outcomes exist.** That log is the only reason the error was
  findable at all.
- **Weekly health monitoring** on pipeline liveness, prediction cadence and retune age. Accuracy is
  reported rather than alarmed, since ~50% is the expected state — and a sudden jump *above* the
  band is treated as a suspected leak rather than good news.
- **20 unit tests run in CI on every push**, covering the date arithmetic that once caused a
  17-day silent data gap, the accounting-format parser, and the merge logic.

## What would have to change

Three fixes, in order of importance, before any future number here should be trusted:

1. **Purge the last seven training rows** at every walk-forward step. That is the leak.
2. **Select features inside the walk-forward** rather than before it. The 52 features were chosen
   from 271 candidates by importance measured across the full eight years, then scored on those
   same eight years.
3. **Tune against a genuinely held-out period.** The hyperparameters came from 100 Optuna trials
   each scored on the entire history, keeping whichever scored highest — a maximum over 100
   attempts on the data being reported, not an out-of-sample result.

None of this is done, and none of it is expected to rescue the model. Seven-day Bitcoin direction
from free daily data may simply not be predictable, and a null result is the honest outcome rather
than a failure. What it would buy is a number worth standing behind.

The pipeline keeps running in the meantime, adding one more independently logged prediction to a
live test of exactly that question.

## The app

Four pages at [bitcoin-price-predictor-v2.streamlit.app](https://bitcoin-price-predictor-v2.streamlit.app):

| Page | What it shows |
|---|---|
| **Dashboard** | Price charts with overlay indicators, sentiment, on-chain metrics, cross-asset comparison |
| **Forecast** | The current 7-day prediction plus the full prediction log, wins and losses alike |
| **Model Performance** | Leak-free vs original figures side by side, market-phase breakdown, live accuracy |
| **Documentation** | Data sources with honest status, methodology, and the full audit |

## Data

All sources are free. Roughly 3,500 daily rows from January 2017, growing by one per day.

| Source | Data | Status |
|---|---|---|
| yfinance | BTC, S&P 500, NASDAQ, Gold, Dollar Index | Live |
| CoinMetrics | 5 on-chain metrics | Live |
| Alternative.me | Fear & Greed index | Live |
| pytrends | Google Trends | Live, dashboard only |
| FRED | 7 macro series | Fetched and stored, **not currently used** by the model or dashboard |
| farside.co.uk | Spot ETF flows | **Broken since May 2026** — the site blocks datacenter IPs, so the fetch returns 403 from CI. Never a model feature, so predictions are unaffected |

## Structure

```
src/
  config.py              shared constants (paths, the 52 selected features, assets)
  data/fetch.py          yfinance, FRED, Alternative.me, pytrends, CoinMetrics, ETF flows
  data/merge.py          combine sources → master_df
  data/features.py       indicator engineering → 271 candidate features + 5 targets
  models/evaluate.py     walk-forward for tree models  ⚠ still contains the leak
  models/evaluate_dl.py  expanding-window for LSTM/GRU (leak-free)
scripts/
  update_data.py         the daily pipeline
  evaluate_leakfree.py   the corrected Gate 1 evaluation ← use this one
  train_production.py    train on all data, save joblib
  tune_xgboost.py        Optuna tuning  ⚠ scores on the full history, no held-out period
  validate_pipeline.py   the six data-quality checks
app/                     Streamlit app (views/ not pages/, to avoid auto-detection)
tests/                   20 unit tests, run in CI on every push
.github/workflows/       daily_update.yml · model_monitor.yml · tests.yml
```

`master_df.csv` is stored as a GitHub Release asset rather than in the repo — it is too large.

## Running it locally

```bash
pip install -r requirements.txt
python scripts/update_data.py        # needs FRED_API_KEY in .env
streamlit run app/app.py
```

---

Built by [Marc Seger](https://www.linkedin.com/in/marcseger/). Not financial advice, and given
the numbers above, that should go without saying.
