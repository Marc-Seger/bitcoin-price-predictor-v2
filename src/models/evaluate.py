"""
evaluate.py — Walk-forward evaluation for tree models (RF, XGBoost).

Walk-forward validation:
    Train on all data up to day T, predict day T+1, record the error.
    Slide forward one day and repeat.

    For RF/XGBoost: retrain every day (fast enough, ~1s per model).
    For LSTM/GRU: see evaluate_dl.py (expanding window, less frequent retraining).

Target:
    Target_Return_1d (next-day % return), not absolute price.
    This avoids the extrapolation problem with tree-based models.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import SELECTED_FEATURES, TARGET_1D, RESULTS_DIR
from utils import load_featured_data, compute_metrics

TARGET = TARGET_1D


def walk_forward_evaluate(df: pd.DataFrame, model_class, model_params: dict,
                          min_train_days: int = 500) -> pd.DataFrame:
    """
    Walk-forward evaluation for a single model.

    Args:
        df: DataFrame with features + target, sorted by date.
        model_class: sklearn-compatible model class (RF or XGBoost).
        model_params: dict of hyperparameters to pass to model_class.
        min_train_days: minimum training window before starting predictions.

    Returns:
        DataFrame with columns: date, actual, predicted (one row per test day).
    """
    features = [c for c in df.columns if c != TARGET]
    results = []

    total_days = len(df) - min_train_days
    print(f'  Walk-forward: {total_days} predictions to make...', end='', flush=True)

    for i in range(min_train_days, len(df)):
        train = df.iloc[:i]
        test_row = df.iloc[i:i+1]

        X_train = train[features]
        y_train = train[TARGET]
        X_test = test_row[features]
        y_actual = test_row[TARGET].values[0]

        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)[0]

        results.append({
            'date': test_row.index[0],
            'actual': y_actual,
            'predicted': y_pred,
        })

        done = i - min_train_days + 1
        if done % 100 == 0:
            print(f' {done}/{total_days}', end='', flush=True)

    print(' done.')
    return pd.DataFrame(results).set_index('date')


def run_evaluation():
    """Run walk-forward evaluation for RF and XGBoost."""
    df = load_featured_data(TARGET)

    models = {
        'RandomForest': (RandomForestRegressor, {
            'n_estimators': 200,
            'max_depth': 15,
            'random_state': 42,
            'n_jobs': -1,
        }),
        'XGBoost': (XGBRegressor, {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'random_state': 42,
            'verbosity': 0,
        }),
    }

    all_metrics = {}
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for name, (model_class, params) in models.items():
        print(f'\nEvaluating {name}...')
        results = walk_forward_evaluate(df, model_class, params, min_train_days=500)

        metrics = compute_metrics(results, label=name)
        all_metrics[name] = metrics

        results_path = os.path.join(RESULTS_DIR, f'{name}_walkforward_results.csv')
        results.to_csv(results_path)
        print(f'  Saved: {results_path}')

    summary = pd.DataFrame(all_metrics).T
    summary_path = os.path.join(RESULTS_DIR, 'walkforward_summary.csv')
    summary.to_csv(summary_path)
    print(f'\nSummary saved: {summary_path}')
    print(summary)

    return all_metrics


if __name__ == '__main__':
    run_evaluation()
