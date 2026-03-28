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
    """Load the most recent row of features from master_df."""
    df = pd.read_csv(MASTER_DF_PATH, index_col=0, parse_dates=True,
                     usecols=['date'] + SELECTED_FEATURES)
    # Use the last row that has all features available
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

    # Load BTC price from master_df
    price_df = pd.read_csv(MASTER_DF_PATH, index_col=0, parse_dates=True,
                           usecols=['date', 'Close_BTC'])
    btc_price = price_df.loc[prediction_date, 'Close_BTC']

    return {
        'prediction_date': prediction_date.strftime('%Y-%m-%d'),
        'target_date': (prediction_date + timedelta(days=7)).strftime('%Y-%m-%d'),
        'predicted_return': float(predicted_return),
        'direction': 'UP' if predicted_return > 0 else 'DOWN',
        'confidence': confidence,
        'btc_price': float(btc_price),
        'predicted_price': float(btc_price * (1 + predicted_return)),
    }


# ─────────────────────────────────────────────
# PAPER TRADING LOG
# ─────────────────────────────────────────────

def load_trade_log() -> pd.DataFrame:
    """Load the paper trading log, or create an empty one."""
    if os.path.exists(TRADE_LOG_PATH):
        return pd.read_csv(TRADE_LOG_PATH, parse_dates=['prediction_date', 'target_date'])
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
    Check past predictions where target_date has passed,
    fill in actual returns from master_df.
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

        target_date = pd.Timestamp(row['target_date'])
        pred_date = pd.Timestamp(row['prediction_date'])

        # Check if we have data for the target date (or closest trading day)
        available = price_df.index[price_df.index >= target_date]
        if len(available) == 0:
            continue

        actual_date = available[0]
        actual_price = price_df.loc[actual_date, 'Close_BTC']
        pred_price = row['btc_price_at_prediction']
        actual_return = (actual_price / pred_price) - 1

        log.at[idx, 'actual_return'] = actual_return
        log.at[idx, 'actual_price'] = actual_price
        log.at[idx, 'correct'] = (actual_return > 0) == (row['predicted_return'] > 0)
        updated = True

    if updated:
        log.to_csv(TRADE_LOG_PATH, index=False)

    return log


# ─────────────────────────────────────────────
# DRIFT DETECTION
# ─────────────────────────────────────────────

DRIFT_THRESHOLD = 0.60  # minimum acceptable direction accuracy
DRIFT_WINDOW = 20       # number of resolved predictions to check

def check_drift() -> dict:
    """
    Check if model accuracy is degrading.
    Looks at the last DRIFT_WINDOW resolved predictions.

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
            'message': f'Need {DRIFT_WINDOW} resolved predictions to detect drift '
                       f'({len(resolved)} so far).',
        }

    recent = resolved.tail(DRIFT_WINDOW)
    accuracy = recent['correct'].mean()

    if accuracy < DRIFT_THRESHOLD:
        return {
            'status': 'WARNING',
            'accuracy': accuracy,
            'message': f'Direction accuracy dropped to {accuracy:.0%} over last '
                       f'{DRIFT_WINDOW} predictions (threshold: {DRIFT_THRESHOLD:.0%}). '
                       f'Consider retraining the model.',
        }

    return {
        'status': 'OK',
        'accuracy': accuracy,
        'message': f'Model performing well: {accuracy:.0%} direction accuracy '
                   f'over last {DRIFT_WINDOW} predictions.',
    }
