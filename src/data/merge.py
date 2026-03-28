"""
merge.py — Merges new fetched data into the existing master_df.

Steps:
1. Load existing master_df.csv
2. Combine new dataframes from fetch.py into a single daily DataFrame
3. Append only truly new rows (no duplicates)
4. Forward-fill sparse columns (FRED macro data is monthly, not daily)
5. Save updated master_df.csv

Note on forward-filling:
    FRED macro series publish monthly. Between release dates we carry
    the last known value forward — this is standard practice (point-in-time).
    The same applies to any other non-daily source.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import MASTER_DF_PATH, FRED_SERIES

# Derived from FRED_SERIES keys — stays in sync automatically
FFILL_COLUMNS = list(FRED_SERIES.keys())


def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_DF_PATH, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = 'date'
    return df


def get_last_date(df: pd.DataFrame) -> str:
    return df.index.max().strftime('%Y-%m-%d')


def combine_new_data(fetched: dict) -> pd.DataFrame:
    """
    Combines the dict of DataFrames returned by fetch_all() into
    a single daily-indexed DataFrame, aligned on the date index.
    """
    frames = []

    for source, df in fetched.items():
        if df is None or df.empty:
            continue

        # Normalize index to date only (strip timezone if present)
        idx = pd.to_datetime(df.index).normalize()
        df.index = idx.tz_convert(None) if idx.tz is not None else idx
        df.index.name = 'date'
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    # Outer join on date — keeps all dates from all sources
    combined = pd.concat(frames, axis=1)
    combined = combined.sort_index()
    return combined


def append_new_rows(master: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    """
    Appends rows from new_data that are not already in master.
    Only dates strictly after the last date in master are added.
    """
    if new_data.empty:
        print('merge: no new rows to append.')
        return master

    last_date = master.index.max()
    new_rows = new_data[new_data.index > last_date]

    if new_rows.empty:
        print('merge: data is already up to date.')
        return master

    # Align columns — new_rows may have columns master doesn't (or vice versa)
    combined = pd.concat([master, new_rows], axis=0, sort=False)
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()

    print(f'merge: appended {len(new_rows)} new rows '
          f'({new_rows.index.min().date()} → {new_rows.index.max().date()}).')
    return combined


def forward_fill_sparse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fills monthly FRED series and other sparse columns.
    Only fills columns that actually exist in the DataFrame.
    """
    cols_to_fill = [c for c in FFILL_COLUMNS if c in df.columns]
    df[cols_to_fill] = df[cols_to_fill].ffill()
    return df


def save_master(df: pd.DataFrame):
    df.to_csv(MASTER_DF_PATH)
    print(f'merge: master_df saved ({len(df)} rows, {len(df.columns)} columns).')


def run_merge(fetched: dict) -> pd.DataFrame:
    """
    Loads master_df, appends new rows from fetched data,
    forward-fills sparse columns, and returns the result.
    Does not save — caller is responsible for saving the final result.
    """
    print('\nRunning merge...')

    master = load_master()
    print(f'merge: loaded master_df — {len(master)} rows, last date: {get_last_date(master)}')

    new_data = combine_new_data(fetched)
    updated = append_new_rows(master, new_data)
    updated = forward_fill_sparse(updated)

    return updated


if __name__ == '__main__':
    # Quick test — fetch last 10 days and merge
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from fetch import fetch_all
    from datetime import date, timedelta

    test_date = (date.today() - timedelta(days=10)).strftime('%Y-%m-%d')
    fetched = fetch_all(test_date)
    updated = run_merge(fetched)
    print(f'\nFinal shape: {updated.shape}')
    print(f'Last 3 rows (BTC close):\n{updated["Close_BTC"].tail(3)}')
