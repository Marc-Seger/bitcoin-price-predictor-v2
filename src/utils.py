"""
utils.py — Shared utilities for data loading and model evaluation metrics.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import MASTER_DF_PATH, SELECTED_FEATURES


def load_featured_data(target: str) -> pd.DataFrame:
    """
    Load master_df and return only rows where all selected features
    and the specified target column are available.
    Uses usecols to avoid parsing unused columns.
    """
    cols_needed = SELECTED_FEATURES + [target]
    df = pd.read_csv(MASTER_DF_PATH, index_col=0, parse_dates=True,
                     usecols=['date'] + cols_needed)

    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns in master_df: {missing}')

    df = df.dropna()
    print(f'Data loaded: {len(df)} rows, {df.index.min().date()} → {df.index.max().date()}')
    return df


def compute_metrics(results: pd.DataFrame, step: int = 1,
                    label: str = '') -> dict:
    """
    Compute evaluation metrics from walk-forward results.

    Args:
        results: DataFrame with 'actual' and 'predicted' columns.
        step: sampling step for non-overlapping evaluation (1 = all, 7 = every 7th).
        label: if provided, print results to stdout.

    Returns:
        dict of metric values.
    """
    actual = results['actual']
    predicted = results['predicted']
    dir_all = ((actual > 0) == (predicted > 0)).mean()

    metrics = {
        'R²': r2_score(actual, predicted),
        'RMSE': mean_squared_error(actual, predicted) ** 0.5,
        'MAE': mean_absolute_error(actual, predicted),
        'Direction': dir_all,
        'N_predictions': len(results),
    }

    # Non-overlapping metrics (when step > 1)
    if step > 1:
        results_no = results.iloc[::step]
        act_no = results_no['actual']
        pred_no = results_no['predicted']
        metrics['R²_non_overlapping'] = r2_score(act_no, pred_no)
        metrics['Direction_non_overlapping'] = ((act_no > 0) == (pred_no > 0)).mean()
        metrics['N_independent'] = len(results_no)

    if label:
        print(f'\n  {label} Results:')
        print(f'    R²={metrics["R²"]:.4f}  Direction={metrics["Direction"]:.2%}  '
              f'({metrics["N_predictions"]} predictions)')
        if step > 1:
            print(f'    Non-overlapping: R²={metrics["R²_non_overlapping"]:.4f}  '
                  f'Direction={metrics["Direction_non_overlapping"]:.2%}  '
                  f'({metrics["N_independent"]} independent)')
        print(f'    RMSE={metrics["RMSE"]:.6f}  MAE={metrics["MAE"]:.6f}')

    return metrics
