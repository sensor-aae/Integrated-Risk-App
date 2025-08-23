from __future__ import annotations
import pandas as pd
import numpy as np

# Loading the data
def load_prices(csv_path:str,data_col:str | None = None) -> pd.DataFrame:
    """ load wide price table (columns are asset tickers)."""
    
    df = pd.read_csv(csv_path)
    
    # Detect or use provided column
    
    if data_col and data_col in df.columns:
        df[data_col] = pd.to_datetime(df[data_col])
        dcol = data_col
    else:
        dcol = df.columns[0]

    # parse date
    df[dcol] = pd.to_datetime(df[dcol] , errors="coerce")
    df = df.set_index(dcol).sort_index()

    # Keep only non-date columns
    df = df[[c for c in df.columns if c != dcol]]
    # strip spaces in headers
    df.columns = [str(c).strip() for c in df.columns]
    # Coerce to numeric
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = ( df[c].astype(str).str.replace(",", "",regex=False).str.replace("$", "", regex= False).str.strip())
        df[c] = pd.to_numeric(df[c], errors="coerce")
        # Drop columns that are entirely NaN
        df = df.dropna(axis=1 , how ="all")
    return df
                
                
def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure prices are usable:
    - Remove duplicate dates
    - Drop columns with <2 valid observations
    - Drop non-positive prices (cannot compute log returns)
    """
    # Remove duplicate index (keep first)
    prices = prices[~prices.index.duplicated(keep="first")]
    # Drop non-positive entries
    prices = prices.where(prices > 0)
    # Drop columns with fewer than 2 valid points
    valid_counts = prices.count()
    prices = prices.loc[:, valid_counts >= 2]
    return prices                         

def to_returns(prices: pd.DataFrame, method:str = "log") -> pd.DataFrame:
    """ Compute the returns from prices."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("Input 'prices' is empty or not a DataFrame.")
    prices = clean_prices(prices)
    if prices.empty or (prices.count()<2).all():
        raise ValueError( "Not enough valid price data per column (need ≥2 rows). "
            "Check your CSV: dates, numeric values, no blanks.")
        
    if method == "log":
        rets = np.log(prices / prices.shift(1))
    elif method == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'log' or 'simple'.")
    
    # Drop rows where all assets are Nan 
    rets = rets.dropna(how="all")
    # Drop columns that never produced returns
    rets = rets.dropna(axis=1, how="all")
    
    if rets.empty:
        raise ValueError("All returns were dropped. Likely only one row of prices or all non-numeric.")

    return rets 
   
