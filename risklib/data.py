"""
risklib/data.py
===============
Price ingestion and return construction.

Pipeline: load_prices -> clean_prices -> to_returns

Each stage guards a specific downstream failure rather than validating in the
abstract. The notes on each function say which one.
"""

from __future__ import annotations

import warnings

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["load_prices", "clean_prices", "to_returns"]


def load_prices(csv_path: Any, date_col: str | None = None) -> pd.DataFrame:
    """
    Load a wide price table (one column per instrument) into a date-indexed frame.

    Parameters
    ----------
    csv_path : path, file-like object, or buffer
    date_col : name of the date column; if omitted, the first column is used

    Notes
    -----
    A sorted DatetimeIndex is a hard requirement for everything downstream —
    .rolling(window), .shift(1), .loc[start:end] and the expanding-window
    variance in the FHS backtest all depend on it. A CSV exported newest-first
    would otherwise run every rolling window backwards through time and silently
    invert the backtest, so the sort is not defensive decoration.

    `errors="coerce"` turns unparseable dates into NaT instead of raising, so a
    single malformed row does not kill an upload.

    Real exports carry values like "$1,234.56". Excel-origin CSVs read those as
    object dtype, and an object column silently poisons .cov() and .mean().
    Stripping the separators and then coercing forces every column to float64
    or NaN.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Price file is empty.")

    dcol = date_col if (date_col and date_col in df.columns) else df.columns[0]

    # Mixed/unknown date formats are expected from arbitrary user uploads, so the
    # "could not infer format" notice is not actionable for the caller.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df[df[dcol].notna()]
    if df.empty:
        raise ValueError(
            f"No parseable dates in column {dcol!r}. Check the date format, or pass "
            "the correct column name explicitly."
        )

    df = df.set_index(dcol).sort_index()

    df.columns = [str(c).strip() for c in df.columns]

    # Coerce every remaining column to numeric.
    #
    # The test is "not already numeric" rather than `dtype == "object"`. pandas 3.0
    # introduced a dedicated `str` dtype, so text columns no longer report as
    # object and an equality check silently stopped stripping currency symbols —
    # the values then failed to parse and the whole column was dropped as
    # non-numeric. is_numeric_dtype is stable across pandas 1.x, 2.x and 3.x.
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = (
                df[c].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Dropped AFTER the loop. Previously this sat inside it, mutating the frame
    # while iterating over its own (stale) column index — a KeyError waiting for
    # the first all-NaN column, and an O(n^2) rescan in the meantime.
    df = df.dropna(axis=1, how="all")

    if df.empty or df.shape[1] == 0:
        raise ValueError(
            "No numeric price columns survived parsing. Check that the file has a "
            "date column plus at least one column of numeric prices."
        )
    return df


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Make a price frame safe for return computation.

    - Duplicate dates break .loc[start:end] slicing and double-count
      observations inside the rolling window.
    - Non-positive prices are set to NaN: log(P_t / P_{t-1}) is undefined for
      them, and a single zero would emit -inf into the return series and then
      into .cov(), turning the entire covariance matrix into NaN.
    - Columns with fewer than two valid observations cannot produce even one
      return.
    """
    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.where(prices > 0)
    return prices.loc[:, prices.count() >= 2]


def to_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Convert prices to returns.

    method="log"    : r = ln(P_t / P_{t-1})   [default]
    method="simple" : r = P_t / P_{t-1} - 1

    Log returns are the default because they are time-additive — an h-day return
    is the sum of h daily returns — which is exactly the property that makes
    sqrt(h) scaling and the i.i.d. assumption coherent. Simple returns are
    offered because portfolio aggregation (w . r) and P&L attribution are
    strictly correct for simple returns and only approximate for log returns.
    Both are exposed; the trade-off is the user's to make.

    dropna(how="all") removes only the first row, which is all-NaN by
    construction from .shift(1). Dropping rows containing ANY NaN would silently
    shorten the sample whenever one instrument has a holiday the others do not.
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("`prices` is empty or not a DataFrame.")

    prices = clean_prices(prices)
    if prices.empty or (prices.count() < 2).all():
        raise ValueError(
            "Not enough valid price data per column (need at least 2 rows). "
            "Check the file for dates, numeric values, and blank cells."
        )

    if method == "log":
        rets = np.log(prices / prices.shift(1))
    elif method == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'log' or 'simple'.")

    rets = rets.replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(how="all").dropna(axis=1, how="all")

    if rets.empty:
        raise ValueError(
            "All returns were dropped. The file likely has only one row of prices, "
            "or no numeric columns."
        )
    return rets
