"""
evaluate_leakfree.py — Walk-forward evaluation without the 7-day target leak.

Why this exists
---------------
`Target_Return_7d = close.shift(-7) / close - 1` (features.py). The original
walk-forward loop in `evaluate.py` and `tune_xgboost.py` trains on `df.iloc[:i]`
and predicts row `i`. Rows i-6 .. i-1 are inside that training slice, and their
target values are computed from closing prices on days i .. i+5 — at and after
the date being predicted. The model is trained on the answer.

`evaluate_dl.py` already guards against this ("drop last 7 training rows (7-day
target peeks into test)"). The XGBoost path never did, which also means the
XGBoost-vs-LSTM/GRU comparison was never like for like.

This script runs both variants over the SAME non-overlapping windows so the cost
of the leak is measured rather than estimated:

    A "as-built"  — train on df.iloc[:i]        (reproduces the published number)
    B "leak-free" — train on df.iloc[:i - 7]    (purges the peeking rows)

Result as of 2026-08-12, 367 windows, 2019-06-19 → 2026-06-26:

    as-built    direction 73.3%   R2  0.571   corr(pred, actual)  0.774
    leak-free   direction 48.5%   R2 -0.308   corr(pred, actual) -0.030
    naive always-up          52.6%

The live prediction log independently agrees with the leak-free figure.

Caveat, stated on purpose
-------------------------
Variant B still uses hyperparameters that were tuned against the leaky objective,
and features selected on the full history. It isolates the leak; it is not a
clean out-of-sample estimate. The defensible claim is "no demonstrated edge",
not "proven unpredictable". Fixing feature selection and tuning is the remaining
work, and is unlikely to change the conclusion.

Usage:
    python3.11 scripts/evaluate_leakfree.py

Output:
    results/XGB_7d_walkforward_leakfree.csv  (date, actual, predicted)
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import TARGET_7D, RESULTS_DIR
from utils import load_featured_data

MIN_TRAIN_DAYS = 500
HORIZON = 7          # also the number of training rows to purge
STEP = 7             # non-overlapping windows

PARAMS_PATH = os.path.join(RESULTS_DIR, 'tuning', 'xgboost_best_params_v2.json')
OUT_PATH = os.path.join(RESULTS_DIR, 'XGB_7d_walkforward_leakfree.csv')
METADATA_KEYS = ('tuned_at',)


def report(df: pd.DataFrame, col: str, label: str):
    correct = ((df['actual'] > 0) == (df[col] > 0)).mean()
    ss_res = ((df['actual'] - df[col]) ** 2).sum()
    ss_tot = ((df['actual'] - df['actual'].mean()) ** 2).sum()
    corr = np.corrcoef(df[col], df['actual'])[0, 1]
    print(f"{label:>22}:  direction {correct*100:5.1f}%   "
          f"R2 {1 - ss_res/ss_tot:7.3f}   corr {corr:6.3f}   n={len(df)}")


def main():
    df = load_featured_data(TARGET_7D)
    features = [c for c in df.columns if c != TARGET_7D]

    with open(PARAMS_PATH) as f:
        params = {k: v for k, v in json.load(f).items() if k not in METADATA_KEYS}

    idxs = range(MIN_TRAIN_DAYS, len(df), STEP)
    print(f"rows={len(df)}  features={len(features)}  windows={len(list(idxs))}")

    rows = []
    start = time.time()
    for n, i in enumerate(range(MIN_TRAIN_DAYS, len(df), STEP), 1):
        test = df.iloc[i:i + 1]
        actual = float(test[TARGET_7D].values[0])
        if not np.isfinite(actual):
            continue

        as_built = XGBRegressor(**params)
        as_built.fit(df[features].iloc[:i], df[TARGET_7D].iloc[:i])

        leak_free = XGBRegressor(**params)
        leak_free.fit(df[features].iloc[:i - HORIZON], df[TARGET_7D].iloc[:i - HORIZON])

        rows.append({
            'date': test.index[0],
            'actual': actual,
            'as_built': float(as_built.predict(test[features])[0]),
            'predicted': float(leak_free.predict(test[features])[0]),
        })

        if n % 25 == 0:
            elapsed = time.time() - start
            print(f"  {n} windows, {elapsed/60:.1f}min elapsed", flush=True)

    results = pd.DataFrame(rows).set_index('date')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results[['actual', 'predicted']].to_csv(OUT_PATH)

    print("\n" + "=" * 72)
    report(results, 'as_built', 'A as-built (leaky)')
    report(results, 'predicted', 'B leak-free')
    print(f"{'naive always-up':>22}:  direction {(results['actual'] > 0).mean()*100:5.1f}%")
    print("=" * 72)
    print(f"saved: {OUT_PATH}")


if __name__ == '__main__':
    main()
