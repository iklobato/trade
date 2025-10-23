"""
Polygon.io flat files loader and processor.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import gzip
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PolygonLoader:
    """
    Load and process Polygon.io crypto flat files.
    """
    
    def __init__(self, data_root: str = "./app/data/cache"):
        self.data_root = Path(data_root)
        
        # Expected schema for Polygon flat files
        self.schema = [
            'ticker', 'volume', 'open', 'close', 'high', 'low', 
            'window_start', 'transactions'
        ]
        
        # Target symbols to filter
        self.target_symbols = ['BTCUSD', 'ETHUSD']
        
    def _load_single_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single Polygon flat file.
        
        Args:
            file_path: Path to the CSV.gz file
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Read compressed CSV
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f, names=self.schema)
            
            # Filter target symbols
            df = df[df['ticker'].isin(self.target_symbols)]
            
            if len(df) == 0:
                logger.warning(f"No target symbols found in {file_path.name}")
                return pd.DataFrame()
            
            # Convert timestamp from nanoseconds to datetime
            df['timestamp'] = pd.to_datetime(df['window_start'], unit='ns', utc=True)
            
            # Convert OHLCV to numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'transactions']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with invalid data
            df = df.dropna(subset=numeric_cols)
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            logger.debug(f"Loaded {len(df)} records from {file_path.name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return pd.DataFrame()
    
    def load_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load Polygon data for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            symbols: List of symbols to load (None for all target symbols)
            
        Returns:
            Combined DataFrame with OHLCV data
        """
        if symbols is None:
            symbols = self.target_symbols
        
        all_data = []
        
        # Generate date range
        current_date = start_date
        while current_date <= end_date:
            file_path = self.data_root / f"{current_date.strftime('%Y-%m-%d')}.csv.gz"
            
            if file_path.exists():
                df = self._load_single_file(file_path)
                if len(df) > 0:
                    all_data.append(df)
            else:
                logger.warning(f"File not found: {file_path}")
            
            current_date += timedelta(days=1)
        
        if not all_data:
            logger.error("No data loaded")
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=False)
        
        # Sort by timestamp
        combined_df.sort_index(inplace=True)
        
        # Remove duplicates (keep last occurrence)
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        
        logger.info(f"Loaded {len(combined_df)} total records from {start_date.date()} to {end_date.date()}")
        return combined_df
    
    def get_symbol_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Extract data for a specific symbol.
        
        Args:
            data: Combined DataFrame with ticker column
            symbol: Symbol to extract
            
        Returns:
            DataFrame with OHLCV data for the symbol
        """
        symbol_data = data[data['ticker'] == symbol].copy()
        
        if len(symbol_data) == 0:
            logger.warning(f"No data found for symbol {symbol}")
            return pd.DataFrame()
        
        # Drop ticker column as it's redundant
        symbol_data = symbol_data.drop('ticker', axis=1)
        
        # Ensure we have the required OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in symbol_data.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Validate OHLC relationships
        invalid_rows = (
            (symbol_data['high'] < symbol_data['low']) |
            (symbol_data['high'] < symbol_data['open']) |
            (symbol_data['high'] < symbol_data['close']) |
            (symbol_data['low'] > symbol_data['open']) |
            (symbol_data['low'] > symbol_data['close'])
        )
        
        if invalid_rows.any():
            logger.warning(f"Found {invalid_rows.sum()} rows with invalid OHLC relationships")
            symbol_data = symbol_data[~invalid_rows]
        
        logger.info(f"Extracted {len(symbol_data)} records for {symbol}")
        return symbol_data
    
    def resample_data(
        self,
        data: pd.DataFrame,
        timeframe: str = '1min',
        agg_method: str = 'ohlc'
    ) -> pd.DataFrame:
        """
        Resample data to different timeframe.
        
        Args:
            data: Input DataFrame with DatetimeIndex
            timeframe: Target timeframe (e.g., '1min', '5min', '1h')
            agg_method: Aggregation method ('ohlc' or 'last')
            
        Returns:
            Resampled DataFrame
        """
        if len(data) == 0:
            return data
        
        if agg_method == 'ohlc':
            # Standard OHLCV aggregation
            resampled = data.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'transactions': 'sum'
            })
        else:
            # Last value aggregation
            resampled = data.resample(timeframe).last()
        
        # Remove rows with NaN values
        resampled = resampled.dropna()
        
        logger.info(f"Resampled {len(data)} records to {timeframe}: {len(resampled)} records")
        return resampled
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return statistics.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with quality metrics
        """
        if len(data) == 0:
            return {'valid': False, 'error': 'Empty dataset'}
        
        stats = {
            'total_records': len(data),
            'date_range': (data.index.min(), data.index.max()),
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_timestamps': data.index.duplicated().sum(),
            'negative_prices': (data[['open', 'high', 'low', 'close']] <= 0).any().any(),
            'zero_volume': (data['volume'] == 0).sum(),
            'invalid_ohlc': 0,
            'valid': True
        }
        
        # Check OHLC relationships
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        stats['invalid_ohlc'] = invalid_ohlc.sum()
        
        # Overall validation
        if stats['duplicate_timestamps'] > 0:
            stats['valid'] = False
            stats['error'] = 'Duplicate timestamps found'
        elif stats['negative_prices']:
            stats['valid'] = False
            stats['error'] = 'Negative prices found'
        elif stats['invalid_ohlc'] > 0:
            stats['valid'] = False
            stats['error'] = f'{stats["invalid_ohlc"]} invalid OHLC relationships'
        
        return stats
