"""
predict.py — Load trained XGBoost model and make predictions.

Used by the Streamlit app to generate the current 7-day forecast.
Also handles logging predictions to the paper trading log.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from xgboost import XGBRegressor

APP_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'xgboost_production.joblib')
PARAMS_PATH = os.path.join(PROJECT_ROOT, 'results', 'tuning', 'xgboost_best_params.json')
TRADE_LOG_PATH = os.path.join(APP_DIR, 'data', 'trade_log.csv')

import sys
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from config import SELECTED_FEATURES, TARGET_7D, MASTER_DF_PATH


def load_model():
    """Load the production XGBoost model from disk."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def get_latest_features() -> pd.DataFrame:
    """Load the most recent row of features from master_df.

    On-chain data (CoinMetrics) typically lags 1-2 days.
    Forward-fill those columns so the prediction always uses the most recent date.
    """
    df = pd.read_csv(MASTER_DF_PATH, index_col=0, parse_dates=True,
                     usecols=['date'] + SELECTED_FEATURES)
    # Forward-fill sparse/lagged columns (on-chain data, etc.) then take last row
    df = df.ffill()
    latest = df.dropna().iloc[-1:]
    return latest


def predict_current() -> dict:
    """
    Make a 7-day return prediction using the latest available data.

    Returns dict with:
        prediction_date: date the prediction was made
        target_date: date the prediction is for (7 days later)
        predicted_return: predicted 7-day % return
        direction: 'UP' or 'DOWN'
        confidence: 'HIGH', 'MEDIUM', or 'LOW' based on magnitude
        btc_price: current BTC price
    """
    model = load_model()
    if model is None:
        return {'error': 'No trained model found. Run scripts/train_production.py first.'}

    latest = get_latest_features()
    if latest.empty:
        return {'error': 'No feature data available. Run scripts/update_data.py first.'}

    predicted_return = model.predict(latest[SELECTED_FEATURES])[0]
    abs_return = abs(predicted_return)

    # Confidence based on our Phase 3 finding:
    # high confidence (large moves) → 95% accuracy
    # low confidence (small moves) → 66% accuracy
    if abs_return > 0.05:
        confidence = 'HIGH'
    elif abs_return > 0.02:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'

    prediction_date = latest.index[0]

    # Load BTC price from master_df — must be present and non-NaN for the prediction date.
    # If it's missing the data pipeline has a gap; we return an error rather than logging
    # a wrong price that would silently corrupt future return calculations.
    price_df = pd.read_csv(MASTER_DF_PATH, index_col=0, parse_dates=True,
                           usecols=['date', 'Close_BTC'])
    if prediction_date not in price_df.index:
        return {'error': f'BTC price row missing for {prediction_date}. Run update_data.py.'}
    btc_price = price_df.loc[prediction_date, 'Close_BTC']
    if pd.isna(btc_price):
        return {'error': f'BTC price is NaN for {prediction_date}. master_df has a data gap.'}

    from datetime import date as _date
    days_stale = (_date.today() - prediction_date.date()).days

    return {
        'prediction_date': prediction_date.strftime('%Y-%m-%d'),
        'target_date': (prediction_date + timedelta(days=7)).strftime('%Y-%m-%d'),
        'predicted_return': float(predicted_return),
        'direction': 'UP' if predicted_return > 0 else 'DOWN',
        'confidence': confidence,
        'btc_price': float(btc_price),
        'predicted_price': float(btc_price * (1 + predicted_return)),
        'days_stale': days_stale,
    }


# ─────────────────────────────────────────────
# PAPER TRADING LOG
# ─────────────────────────────────────────────

def load_trade_log() -> pd.DataFrame:
    """Load the paper trading log, or create an empty one."""
    if os.path.exists(TRADE_LOG_PATH):
        # format='mixed': a hand-repaired row once stored its date with a time
        # component while every other row was date-only. Pandas infers the format
        # from the first value and raises on the rest — that single inconsistency
        # silently disabled the weekly monitor for five months.
        return pd.read_csv(
            TRADE_LOG_PATH,
            parse_dates=['prediction_date', 'target_date'],
            date_format='mixed',
        )
    return pd.DataFrame(columns=[
        'prediction_date', 'target_date', 'predicted_return', 'direction',
        'confidence', 'btc_price_at_prediction', 'predicted_price',
        'actual_return', 'actual_price', 'correct', 'logged_at',
    ])


def log_prediction(prediction: dict):
    """
    Append a new prediction to the trade log.
    Skips if a prediction for this date already exists.
    """
    log = load_trade_log()

    # Don't duplicate
    if not log.empty and prediction['prediction_date'] in log['prediction_date'].astype(str).values:
        return

    new_row = {
        'prediction_date': prediction['prediction_date'],
        'target_date': prediction['target_date'],
        'predicted_return': prediction['predicted_return'],
        'direction': prediction['direction'],
        'confidence': prediction['confidence'],
        'btc_price_at_prediction': prediction['btc_price'],
        'predicted_price': prediction['predicted_price'],
        'actual_return': None,
        'actual_price': None,
        'correct': None,
        'logged_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }

    log = pd.concat([log, pd.DataFrame([new_row])], ignore_index=True)
    os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
    log.to_csv(TRADE_LOG_PATH, index=False)


def update_outcomes():
    """
    For each past prediction whose target_date has passed, fill in actual_return,
    actual_price, and correct from master_df.

    Skips rows already resolved (actual_return is set).
    Skips rows whose target_date is not yet in master_df (still pending).
    """
    log = load_trade_log()
    if log.empty:
        return log

    price_df = pd.read_csv(MASTER_DF_PATH, index_col=0, parse_dates=True,
                           usecols=['date', 'Close_BTC'])

    updated = False
    for idx, row in log.iterrows():
        if pd.notna(row['actual_return']):
            continue

        entry_price = row['btc_price_at_prediction']
        if pd.isna(entry_price) or float(entry_price) == 0:
            # Should never happen — predict_current() now guards against this.
            # Log a warning but skip rather than computing a wrong return.
            print(f"WARNING: missing entry price for prediction {row['prediction_date']} — skipping outcome update.")
            continue

        target_date = pd.Timestamp(row['target_date'])
        available   = price_df.index[price_df.index >= target_date]
        if len(available) == 0:
            continue  # target date not yet in master_df — still pending

        actual_price  = float(price_df.loc[available[0], 'Close_BTC'])
        actual_return = (actual_price / float(entry_price)) - 1

        log.at[idx, 'actual_return'] = float(actual_return)
        log.at[idx, 'actual_price']  = float(actual_price)
        log.at[idx, 'correct']       = float((actual_return > 0) == (row['predicted_return'] > 0))
        updated = True

    if updated:
        log.to_csv(TRADE_LOG_PATH, index=False)

    return log


# ─────────────────────────────────────────────
# DRIFT DETECTION
# ─────────────────────────────────────────────

# Expected state is ~50%: the August 2026 audit found target leakage in the original
# evaluation, and leak-free the model has no demonstrated edge (48.5% against a 52.6%
# baseline, matched by the live log). So a band, not a floor.
#
# This previously used DRIFT_THRESHOLD = 0.60 as a "minimum acceptable accuracy", which
# after the audit meant the Forecast page permanently displayed "accuracy dropped...
# consider retraining". That advice was wrong: retraining cannot fix a null result, and
# a 60% floor implied a level this model has never genuinely reached. Same correction as
# the weekly monitor in .github/workflows/model_monitor.yml — ask whether something has
# changed mechanically, not whether the model is "good".
DRIFT_LOW, DRIFT_HIGH = 0.30, 0.70
DRIFT_WINDOW = 30       # number of resolved predictions to check


def check_drift() -> dict:
    """
    Check whether live accuracy has moved far enough from ~50% to suggest something
    broke, in either direction.

    A large *jump* is treated as suspicious rather than good news: given this project's
    history, accuracy well above chance is more likely a new data leak or a misaligned
    join than a real improvement.

    Returns dict with:
        status: 'OK', 'WARNING', or 'NO_DATA'
        accuracy: recent direction accuracy (if enough data)
        message: human-readable status
    """
    log = load_trade_log()
    resolved = log.dropna(subset=['correct'])

    if len(resolved) < DRIFT_WINDOW:
        return {
            'status': 'NO_DATA',
            'accuracy': None,
            'message': f'Need {DRIFT_WINDOW} resolved predictions to evaluate '
                       f'({len(resolved)} so far).',
        }

    recent = resolved.tail(DRIFT_WINDOW)
    accuracy = recent['correct'].mean()

    if accuracy > DRIFT_HIGH:
        return {
            'status': 'WARNING',
            'accuracy': accuracy,
            'message': f'Direction accuracy is {accuracy:.0%} over the last {DRIFT_WINDOW} '
                       f'predictions. This model has no established edge, so a result this '
                       f'good is more likely a data leak or a join bug than a genuine '
                       f'improvement. Investigate before celebrating.',
        }

    if accuracy < DRIFT_LOW:
        return {
            'status': 'WARNING',
            'accuracy': accuracy,
            'message': f'Direction accuracy is {accuracy:.0%} over the last {DRIFT_WINDOW} '
                       f'predictions. Consistently worse than chance usually means an '
                       f'inverted sign or a misaligned join rather than a bad model.',
        }

    return {
        'status': 'OK',
        'accuracy': accuracy,
        'message': f'Direction accuracy {accuracy:.0%} over the last {DRIFT_WINDOW} '
                   f'predictions, in line with the ~50% this model is expected to produce. '
                   f'Nothing has changed mechanically.',
    }
