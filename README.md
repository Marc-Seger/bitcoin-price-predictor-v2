# Bitcoin Price Predictor v2

An autonomous machine-learning pipeline that forecasts Bitcoin's 7-day return, grades its own
past calls, and publishes the record — including where it gets things wrong.

**[Live app →](https://bitcoin-price-predictor-v2.streamlit.app)**  ·  Python · XGBoost · Streamlit · GitHub Actions

---

## The headline, stated up front

**This model has no demonstrated edge.** It calls the direction of Bitcoin's next 7 days about as
well as a coin flip — slightly worse, in fact, than simply guessing "up" every single time.

| | | what it means |
|---|---|---|
| Direction accuracy | **48.5%** | how often it got up-vs-down right |
| Always guessing "up" | **52.6%** | the do-nothing benchmark it fails to beat |
| Correlation with reality | **−0.030** | 0 means no relationship at all. This is 0 |
| R² | **−0.308** | below 0 means worse than always predicting the average |
| Tested over | 367 independent weeks | June 2019 → June 2026 |
| Live track record | ~100 forecasts, **~49%** | logged daily since March 2026, before outcomes were known |

The last two rows are the important ones. The backtest and the live record were measured
completely independently and landed within a point of each other, which is what makes the
conclusion solid rather than a guess.

An earlier version of this README and the app reported **76.7% direction accuracy**. That figure
was wrong. It survived for months because the documents describing it copied each other instead
of re-deriving it, and because the monitoring built to catch exactly this had itself silently
broken. Both problems are described below, because they are the more interesting part of the
project.

## What went wrong, and how it was found

Neither error was in the model. Both were in how I *tested* it. No ML background needed for
what follows.

### Error 1: counting the same call many times

The model forecasts the next 7 days, and I ran it every single day. So Monday's forecast covers
Mon→Mon, Tuesday's covers Tue→Tue, and those two windows **share six of their seven days**.
They're nearly the same bet.

If Bitcoin climbs steadily for a fortnight and the model says "up", it's right on Monday, right
again Tuesday, right again Wednesday — fourteen ticks in the correct column. But that isn't
fourteen good calls. It's about two, counted over and over.

It's like grading a weather forecaster who says "rain this week" every morning during a wet
fortnight, then crediting them with fourteen correct forecasts.

So "2,467 predictions" was really **367** independent ones, and lucky streaks got multiplied.
Cost: about 3 percentage points.

### Error 2: the model was shown the answers

This is the one that mattered.

To teach the model, you give it thousands of worked examples: *here are the market conditions on
some day, and here is what Bitcoin actually did over the following week.* It learns by finding
patterns linking the two.

The catch is that the second half of each example — what actually happened next — **can only be
filled in a week later**. The answer for 1 January requires the price on 8 January.

My test loop trained on every day up to the forecast date. So to forecast from **2 January**, it
trained on examples including **1 January** — whose answer had been calculated using the price on
**8 January**, six days after the moment the model was supposedly standing in.

I had handed it worked examples whose answers were taken from the very week it was being asked to
predict. And since one day's market conditions look nearly identical to the next day's, it didn't
need to learn anything. It could recognise the near-duplicate and repeat the answer back.

The rule I'd broken, in one line: **you can only learn from examples that have already finished.**
On 2 January the most recent finished example is 26 December, because its week closed that
morning. Everything after it was still running.

The fix discards those last few days of training data at every step. Cost: you always give up the
most recent week. Benefit: the model can no longer see any part of the future.

**Removing the leak takes direction accuracy from 73.3% to 48.5%**, and the link between
prediction and outcome from 0.77 down to −0.03 — that second number means the forecasts and
reality are, statistically, unrelated. Not a weaker edge. No edge.

For the technically inclined: the target is `Target_Return_7d = close.shift(-7) / close - 1`, and
the walk-forward loop trained on `df.iloc[:i]` where it should have used `df.iloc[:i-7]`.

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
- **Weekly health monitoring** on pipeline liveness and prediction cadence — these alert on
  failure, since the dashboard is meant to stay live and current. Accuracy and retune age are
  both reported, not alarmed: ~50% is the expected state (a sudden jump *above* the band reads as
  a suspected leak, not good news), and retuning is paused until the tuning script's own leak is
  fixed, so flagging its age weekly isn't actionable yet.
- **20 unit tests run in CI on every push**, covering the date arithmetic that once caused a
  17-day silent data gap, the accounting-format parser, and the merge logic.

## What would have to change

Three fixes, in order of importance, before any future number here should be trusted. All three
are versions of the same mistake: **letting the test see something it shouldn't.**

1. **Stop training on unfinished examples.** That's the leak described above, and it's fixed in
   `scripts/evaluate_leakfree.py` — but the older scripts still contain it.

2. **Choose the inputs without peeking.** The model uses 52 pieces of market data, picked from
   271 candidates because they looked most useful *across the whole eight years* — then tested on
   those same eight years. So the choice of inputs was informed by the period being graded. The
   selection should happen inside the test, using only past data.

3. **Tune the settings on a period you then throw away.** The model's settings were chosen by
   trying 100 combinations, scoring each one across the entire history, and keeping the best.
   That's picking the winner of 100 attempts *on the exam you're about to sit* — so the reported
   score is a best-of-100, not an honest estimate. The standard fix is to tune on early years and
   report only on later years the tuning never touched.

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
