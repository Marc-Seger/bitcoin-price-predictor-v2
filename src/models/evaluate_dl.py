"""
evaluate_dl.py — Expanding-window evaluation for LSTM and GRU models.

Strategy:
    Expanding window with monthly retraining. Train on all data up to
    month M, predict every day in month M+1, slide forward one month.
    ~15-20 retraining cycles instead of 1700 daily retrains.

Safeguards (identified during pre-implementation review):
    1. Target leakage: drop last 7 training rows (7-day target peeks into test)
    2. Scaling: fit scaler on training data only, apply to train+test
    3. Evaluation: non-overlapping predictions (every 7th day) for honest metrics
    4. Early stopping: last 10% of training data as validation (chronological)

Target:
    Target_Return_7d (7-day % return).
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import SELECTED_FEATURES, TARGET_7D, RESULTS_DIR
from utils import load_featured_data, compute_metrics

TARGET = TARGET_7D
SEQUENCE_LENGTH = 30
TARGET_HORIZON = 7


# ─────────────────────────────────────────────
# SEQUENCE BUILDING
# ─────────────────────────────────────────────

def build_sequences(features: np.ndarray, targets: np.ndarray,
                    seq_len: int) -> tuple:
    """
    Converts flat arrays into (X, y) where X has shape
    (n_samples, seq_len, n_features).
    """
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# MODEL BUILDING
# ─────────────────────────────────────────────

def build_model(model_type: str, n_features: int, seq_len: int) -> Sequential:
    """
    Builds an LSTM or GRU model.

    Architecture rationale (for ~1500 training rows):
        - 48/24 units: enough capacity without overfitting
        - 0.25 dropout: slightly aggressive for noisy financial data
        - 2 layers: captures short-term within long-term patterns
    """
    layer_class = LSTM if model_type == 'LSTM' else GRU

    model = Sequential([
        layer_class(48, return_sequences=True,
                    input_shape=(seq_len, n_features)),
        Dropout(0.25),
        layer_class(24, return_sequences=False),
        Dropout(0.25),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ─────────────────────────────────────────────
# EXPANDING WINDOW EVALUATION
# ─────────────────────────────────────────────

def expanding_window_evaluate(df: pd.DataFrame, model_type: str,
                               min_train_days: int = 500) -> pd.DataFrame:
    """
    Expanding-window evaluation with monthly retraining.

    For each month M in the test period:
        1. Training data = all rows up to month M (minus last 7 for leakage)
        2. Fit scaler on training data, transform everything
        3. Build sequences, split last 10% of training for early stopping
        4. Train model with early stopping
        5. Predict every day in month M+1
        6. Record predictions
    """
    features_cols = [c for c in df.columns if c != TARGET]
    n_features = len(features_cols)

    all_dates = df.index
    first_test_date = all_dates[min_train_days]
    test_months = pd.period_range(
        start=first_test_date.to_period('M'),
        end=all_dates[-1].to_period('M'),
        freq='M'
    )

    all_results = []
    n_months = len(test_months)
    print(f'  {model_type}: {n_months} monthly retraining cycles')

    for cycle, month in enumerate(test_months):
        month_start = month.start_time
        month_end = month.end_time

        train_mask = df.index < month_start
        test_mask = (df.index >= month_start) & (df.index <= month_end)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) < min_train_days or len(test_df) == 0:
            continue

        # Safeguard #1: drop last TARGET_HORIZON rows from training
        train_df = train_df.iloc[:-TARGET_HORIZON]

        # Safeguard #2: fit scaler on training data only
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_df[features_cols])
        test_features = scaler.transform(test_df[features_cols])

        train_targets = train_df[TARGET].values
        test_targets = test_df[TARGET].values

        # Combine for boundary sequences (test day 1 needs 30 days before it)
        all_features = np.vstack([train_features, test_features])
        all_targets = np.concatenate([train_targets, test_targets])

        X_all, y_all = build_sequences(all_features, all_targets, SEQUENCE_LENGTH)

        n_train_sequences = len(train_features) - SEQUENCE_LENGTH
        if n_train_sequences <= 0:
            continue

        X_train = X_all[:n_train_sequences]
        y_train = y_all[:n_train_sequences]
        X_test = X_all[n_train_sequences:]
        y_test = y_all[n_train_sequences:]

        if len(X_test) == 0:
            continue

        # Safeguard #4: early stopping with last 10% of training as validation
        val_split = max(1, int(len(X_train) * 0.1))
        X_val = X_train[-val_split:]
        y_val = y_train[-val_split:]
        X_train_fit = X_train[:-val_split]
        y_train_fit = y_train[:-val_split]

        model = build_model(model_type, n_features, SEQUENCE_LENGTH)
        early_stop = EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )

        model.fit(
            X_train_fit, y_train_fit,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0,
        )

        predictions = model.predict(X_test, verbose=0).flatten()
        test_dates = test_df.index[-(len(predictions)):]

        for dt, actual, pred in zip(test_dates, y_test, predictions):
            all_results.append({
                'date': dt,
                'actual': actual,
                'predicted': pred,
            })

        stopped_at = early_stop.stopped_epoch if early_stop.stopped_epoch > 0 else 50
        print(f'    Cycle {cycle+1}/{n_months}: {month} — '
              f'{len(X_train_fit)} train, {len(X_test)} test, '
              f'{stopped_at} epochs', flush=True)

        tf.keras.backend.clear_session()

    return pd.DataFrame(all_results).set_index('date')


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run_evaluation():
    """Run expanding-window evaluation for LSTM and GRU."""
    df = load_featured_data(TARGET)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_metrics = {}

    for model_type in ['LSTM', 'GRU']:
        print(f'\nEvaluating {model_type}...')
        results = expanding_window_evaluate(df, model_type, min_train_days=500)

        metrics = compute_metrics(results, step=7, label=model_type)
        all_metrics[model_type] = metrics

        path = os.path.join(RESULTS_DIR, f'{model_type}_7d_walkforward_results.csv')
        results.to_csv(path)
        print(f'    Saved: {path}')

    summary = pd.DataFrame(all_metrics).T
    summary_path = os.path.join(RESULTS_DIR, 'dl_walkforward_summary.csv')
    summary.to_csv(summary_path)
    print(f'\nSummary saved: {summary_path}')
    print(summary)

    return all_metrics


if __name__ == '__main__':
    run_evaluation()
