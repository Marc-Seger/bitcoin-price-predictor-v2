"""
Unit tests for the three pipeline functions most prone to silent bugs:
  - _date_range()        (fetch.py)
  - fetch_yfinance()     (fetch.py) — date math only, yf.download is mocked
  - _parse_accounting()  (fetch.py)
  - append_new_rows()    (merge.py)
"""

import sys
import os
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# Allow importing from src/data and src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'data'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fetch import _date_range, _parse_accounting, fetch_yfinance
from merge import append_new_rows


# ─────────────────────────────────────────────
# _date_range()
# ─────────────────────────────────────────────

class TestDateRange:
    def test_already_up_to_date(self):
        today = date.today().isoformat()
        start, end = _date_range(today)
        assert start is None
        assert end is None

    def test_one_day_behind(self):
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        start, end = _date_range(yesterday)
        assert start == date.today()
        assert end == date.today()

    def test_ten_days_behind(self):
        ten_ago = (date.today() - timedelta(days=10)).isoformat()
        start, end = _date_range(ten_ago)
        assert start == (date.today() - timedelta(days=9))
        assert end == date.today()


# ─────────────────────────────────────────────
# fetch_yfinance() — date math (no real API calls)
# ─────────────────────────────────────────────

def _make_empty_yf_download():
    return pd.DataFrame()


def _make_fake_yf_download(tickers_str, start, end, **kwargs):
    """Returns a minimal multi-level DataFrame mimicking yf.download output."""
    dates = pd.date_range(start=start, end=end, freq='B')
    if len(dates) == 0:
        return pd.DataFrame()
    cols = pd.MultiIndex.from_tuples(
        [('Close', 'BTC-USD'), ('Open', 'BTC-USD')],
        names=[None, None],
    )
    data = pd.DataFrame(1.0, index=dates, columns=cols)
    data.index.name = 'Date'
    return data


class TestFetchYfinanceDateMath:
    def test_end_is_today_plus_one(self):
        last_date = (date.today() - timedelta(days=5)).isoformat()
        captured = {}

        def fake_download(tickers_str, start, end, **kwargs):
            captured['end'] = end
            captured['start'] = start
            return _make_empty_yf_download()

        with patch('fetch.yf.download', side_effect=fake_download):
            fetch_yfinance(last_date)

        expected_end = str(date.today() + timedelta(days=1))
        assert captured.get('end') == expected_end, (
            f"Expected end={expected_end}, got {captured.get('end')}"
        )

    def test_start_is_last_date_minus_two(self):
        last_date = (date.today() - timedelta(days=5)).isoformat()
        captured = {}

        def fake_download(tickers_str, start, end, **kwargs):
            captured['start'] = start
            return _make_empty_yf_download()

        with patch('fetch.yf.download', side_effect=fake_download):
            fetch_yfinance(last_date)

        expected_start = str((pd.to_datetime(last_date) - timedelta(days=2)).date())
        assert captured.get('start') == expected_start, (
            f"Expected start={expected_start}, got {captured.get('start')}"
        )

    def test_empty_download_returns_empty_dataframe(self):
        last_date = (date.today() - timedelta(days=3)).isoformat()
        with patch('fetch.yf.download', return_value=pd.DataFrame()):
            result = fetch_yfinance(last_date)
        assert result.empty

    def test_already_up_to_date_skips_download(self):
        today = date.today().isoformat()
        with patch('fetch.yf.download') as mock_dl:
            result = fetch_yfinance(today)
        mock_dl.assert_not_called()
        assert result.empty


# ─────────────────────────────────────────────
# _parse_accounting()
# ─────────────────────────────────────────────

class TestParseAccounting:
    def test_negative_parentheses(self):
        assert _parse_accounting('(95.1)') == pytest.approx(-95.1)

    def test_dash_is_zero(self):
        assert _parse_accounting('-') == 0.0

    def test_empty_string_is_zero(self):
        assert _parse_accounting('') == 0.0

    def test_comma_thousands(self):
        assert _parse_accounting('1,234.5') == pytest.approx(1234.5)

    def test_nan_string_is_zero(self):
        assert _parse_accounting('nan') == 0.0

    def test_positive_plain(self):
        assert _parse_accounting('655.3') == pytest.approx(655.3)

    def test_none_string_is_zero(self):
        assert _parse_accounting('None') == 0.0


# ─────────────────────────────────────────────
# append_new_rows()
# ─────────────────────────────────────────────

def _make_df(data: dict, dates) -> pd.DataFrame:
    df = pd.DataFrame(data, index=pd.to_datetime(dates))
    df.index.name = 'date'
    return df


class TestAppendNewRows:
    def test_nan_in_master_filled_by_new_data(self):
        master = _make_df({'Close_BTC': [50000.0, np.nan]}, ['2026-04-01', '2026-04-02'])
        new_data = _make_df({'Close_BTC': [51000.0]}, ['2026-04-02'])
        result = append_new_rows(master, new_data)
        assert result.loc['2026-04-02', 'Close_BTC'] == pytest.approx(51000.0)

    def test_real_value_in_master_not_overwritten(self):
        master = _make_df({'Close_BTC': [50000.0, 51000.0]}, ['2026-04-01', '2026-04-02'])
        new_data = _make_df({'Close_BTC': [99999.0]}, ['2026-04-02'])
        result = append_new_rows(master, new_data)
        assert result.loc['2026-04-02', 'Close_BTC'] == pytest.approx(51000.0)

    def test_new_dates_appended(self):
        master = _make_df({'Close_BTC': [50000.0]}, ['2026-04-01'])
        new_data = _make_df({'Close_BTC': [51000.0, 52000.0]}, ['2026-04-02', '2026-04-03'])
        result = append_new_rows(master, new_data)
        assert pd.Timestamp('2026-04-02') in result.index
        assert pd.Timestamp('2026-04-03') in result.index
        assert len(result) == 3

    def test_no_duplicate_dates(self):
        master = _make_df({'Close_BTC': [50000.0, 51000.0]}, ['2026-04-01', '2026-04-02'])
        new_data = _make_df({'Close_BTC': [51000.0, 52000.0]}, ['2026-04-02', '2026-04-03'])
        result = append_new_rows(master, new_data)
        assert result.index.duplicated().sum() == 0

    def test_empty_new_data_returns_master_unchanged(self):
        master = _make_df({'Close_BTC': [50000.0]}, ['2026-04-01'])
        result = append_new_rows(master, pd.DataFrame())
        pd.testing.assert_frame_equal(result, master)
