"""
tune_xgboost.py — Bayesian hyperparameter tuning for XGBoost using Optuna.

Uses walk-forward validation (same as evaluate.py) to score each trial.
This ensures we're optimizing for real predictive performance, not overfitting
to a specific test period.

Usage:
    python scripts/tune_xgboost.py

    Takes ~2-4 hours for 100 trials. Progress is saved after each trial
    (SQLite database) — safe to interrupt and resume.

Output:
    results/tuning/xgboost_tuning_results.csv  — all trials ranked by score
    results/tuning/xgboost_best_params.json    — best parameters found
"""

import os
import sys
import json
import time
import optuna
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Suppress Optuna's verbose logging (show only progress)
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import SELECTED_FEATURES, TARGET_7D, RESULTS_DIR
from utils import load_featured_data

TARGET = TARGET_7D
N_TRIALS = 100
MIN_TRAIN_DAYS = 500

# Walk-forward uses every 7th prediction for non-overlapping 7-day evaluation
EVAL_STEP = 7


def walk_forward_score(df: pd.DataFrame, params: dict) -> float:
    """
    Walk-forward evaluation returning R² on non-overlapping 7-day predictions.
    Same logic as evaluate.py but returns a single score for Optuna to optimize.
    """
    features = [c for c in df.columns if c != TARGET]
    actuals, preds = [], []

    for i in range(MIN_TRAIN_DAYS, len(df)):
        train = df.iloc[:i]
        test_row = df.iloc[i:i+1]

        model = XGBRegressor(**params, verbosity=0, random_state=42)
        model.fit(train[features], train[TARGET])
        pred = model.predict(test_row[features])[0]

        actuals.append(test_row[TARGET].values[0])
        preds.append(pred)

    # Non-overlapping: every 7th prediction
    actuals_no = actuals[::EVAL_STEP]
    preds_no = preds[::EVAL_STEP]

    return r2_score(actuals_no, preds_no)


def objective(trial, df):
    """Optuna objective function — suggests params and returns R² score."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }

    score = walk_forward_score(df, params)
    return score


def main():
    start = time.time()
    print('=' * 60)
    print('XGBoost Hyperparameter Tuning (Optuna + Walk-Forward)')
    print(f'Target: {TARGET}')
    print(f'Trials: {N_TRIALS}')
    print(f'Min training window: {MIN_TRAIN_DAYS} days')
    print('=' * 60)

    df = load_featured_data(TARGET)
    total_preds = len(df) - MIN_TRAIN_DAYS
    print(f'Each trial evaluates {total_preds} predictions ({total_preds // EVAL_STEP} non-overlapping)')
    print(f'Estimated time: {N_TRIALS * 2}-{N_TRIALS * 4} minutes')
    print()

    # SQLite storage — allows interrupting and resuming
    tuning_dir = os.path.join(RESULTS_DIR, 'tuning')
    os.makedirs(tuning_dir, exist_ok=True)
    db_path = os.path.join(tuning_dir, 'xgboost_study_v2.db')
    storage = f'sqlite:///{db_path}'

    study = optuna.create_study(
        study_name='xgboost_7d_clean',
        direction='maximize',  # maximize R²
        storage=storage,
        load_if_exists=True,  # resume if interrupted
    )

    # Track progress
    completed = len(study.trials)
    remaining = max(0, N_TRIALS - completed)
    if completed > 0:
        print(f'Resuming from trial {completed + 1} ({completed} already completed)')
        print(f'Best so far: R²={study.best_value:.4f}')
    print(f'Running {remaining} trials...\n')

    def progress_callback(study, trial):
        n = trial.number + 1
        best = study.best_value
        elapsed_min = (time.time() - start) / 60
        avg_per_trial = elapsed_min / max(1, n - completed)
        remaining_min = avg_per_trial * (N_TRIALS - n)
        print(f'  Trial {n}/{N_TRIALS}: R²={trial.value:.4f}  '
              f'(best: {best:.4f})  '
              f'~{remaining_min:.0f}min remaining')

    if remaining > 0:
        study.optimize(
            lambda trial: objective(trial, df),
            n_trials=remaining,
            callbacks=[progress_callback],
        )

    # Save results
    print('\n' + '=' * 60)
    print('TUNING COMPLETE')
    print('=' * 60)

    best = study.best_params
    best['verbosity'] = 0
    best['random_state'] = 42
    print(f'\nBest R² (non-overlapping): {study.best_value:.4f}')
    print(f'Best parameters:')
    for k, v in best.items():
        print(f'  {k}: {v}')

    # Compare with baseline
    baseline_params = {
        'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6,
        'verbosity': 0, 'random_state': 42,
    }
    print(f'\nBaseline params for reference:')
    for k, v in baseline_params.items():
        print(f'  {k}: {v}')

    # Save best params as JSON
    params_path = os.path.join(tuning_dir, 'xgboost_best_params_v2.json')
    with open(params_path, 'w') as f:
        json.dump(best, f, indent=2)
    print(f'\nBest params saved: {params_path}')

    # Save all trials as CSV
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values('value', ascending=False)
    trials_path = os.path.join(tuning_dir, 'xgboost_tuning_results_v2.csv')
    trials_df.to_csv(trials_path, index=False)
    print(f'All trials saved: {trials_path}')

    elapsed = (time.time() - start) / 60
    print(f'\nTotal time: {elapsed:.1f} minutes')


if __name__ == '__main__':
    main()
