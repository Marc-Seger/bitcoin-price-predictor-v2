"""
train_production.py — Train and save the production XGBoost model.

Trains on ALL available data (no train/test split — we already validated
performance via walk-forward in Phase 3). Saves the model as a joblib file
for use by the Streamlit app.

Usage:
    python scripts/train_production.py

    Optionally loads tuned hyperparameters from results/tuning/xgboost_best_params.json.
    Falls back to baseline params if tuning hasn't been run yet.
"""

import os
import sys
import json
import joblib
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import SELECTED_FEATURES, TARGET_7D, RESULTS_DIR
from utils import load_featured_data

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_production.joblib')
PARAMS_PATH = os.path.join(RESULTS_DIR, 'tuning', 'xgboost_best_params_v2.json')

BASELINE_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 6,
    'verbosity': 0,
    'random_state': 42,
}


def load_params() -> dict:
    """Load tuned params if available, otherwise use baseline."""
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            params = json.load(f)
        print(f'Using tuned parameters from {PARAMS_PATH}')
        return params

    print('No tuned parameters found — using baseline params.')
    return BASELINE_PARAMS.copy()


def main():
    print('Training production XGBoost model...')

    df = load_featured_data(TARGET_7D)
    params = load_params()

    print(f'\nParameters:')
    for k, v in params.items():
        print(f'  {k}: {v}')

    model = XGBRegressor(**params)
    model.fit(df[SELECTED_FEATURES], df[TARGET_7D])

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f'\nModel trained on {len(df)} rows')
    print(f'Features: {len(SELECTED_FEATURES)}')
    print(f'Saved to: {MODEL_PATH}')
    print(f'File size: {os.path.getsize(MODEL_PATH) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
