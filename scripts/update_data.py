"""
update_data.py — Single entry point to update the full data pipeline.

Usage:
    python scripts/update_data.py

Steps:
    1. Reads last date from master_df.csv
    2. Fetches new data from all sources since that date
    3. Merges new rows into master_df
    4. Recomputes all technical indicators on the full dataset
    5. Saves the updated master_df.csv
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'data'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fetch import fetch_all
from merge import load_master, get_last_date, append_new_rows, combine_new_data, forward_fill_sparse, save_master
from features import compute_features


def main():
    start_time = datetime.now()
    print('=' * 50)
    print('Bitcoin Price Predictor — Data Update')
    print(f'Started: {start_time.strftime("%Y-%m-%d %H:%M")}')
    print('=' * 50)

    master = load_master()
    last_date = get_last_date(master)
    print(f'\nCurrent data: {len(master)} rows, last date: {last_date}')

    fetched = fetch_all(last_date)

    has_new_data = any(not df.empty for df in fetched.values())
    if not has_new_data:
        print('\nAll sources up to date. Nothing to do.')
        return

    # Merge using the already-loaded master (avoids re-reading CSV)
    new_data = combine_new_data(fetched)
    updated = append_new_rows(master, new_data)
    updated = forward_fill_sparse(updated)

    final = compute_features(updated)
    save_master(final)

    elapsed = (datetime.now() - start_time).seconds
    print('\n' + '=' * 50)
    print(f'Done in {elapsed}s')
    print(f'Rows: {len(master)} → {len(final)} (+{len(final) - len(master)})')
    print(f'Last date: {final.index.max().date()}')
    print('=' * 50)


if __name__ == '__main__':
    main()
