"""
Common data processing utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path


def load_csv_data(file_path: str, symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Load CSV data with common preprocessing.
    
    Args:
        file_path: Path to CSV file
        symbol: Optional symbol to filter by
        
    Returns:
        Processed DataFrame
    """
    df = pd.read_csv(file_path)
    
    # Common preprocessing
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    # Ensure numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate returns for given periods."""
    return prices.pct_change(periods)


def calculate_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling volatility."""
    returns = calculate_returns(prices)
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized


def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to different timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        timeframe: Target timeframe (e.g., '1H', '1D')
        
    Returns:
        Resampled DataFrame
    """
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    return df.resample(timeframe).agg(ohlc_dict).dropna()
