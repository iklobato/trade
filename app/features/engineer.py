"""
Causal feature engineering for ML trading model.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Generate causal technical features for ML model.
    """
    
    def __init__(self):
        self.feature_names = []
        
    def calculate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical features.
        
        Args:
            data: DataFrame with OHLCV data and DatetimeIndex
            
        Returns:
            DataFrame with all features added
        """
        df = data.copy()
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate price-based features
        df = self._add_price_features(df)
        
        # Calculate momentum features
        df = self._add_momentum_features(df)
        
        # Calculate volatility features
        df = self._add_volatility_features(df)
        
        # Calculate volume features
        df = self._add_volume_features(df)
        
        # Calculate trend features
        df = self._add_trend_features(df)
        
        # Calculate candle features
        df = self._add_candle_features(df)
        
        # Calculate transaction features
        df = self._add_transaction_features(df)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col not in required_cols]
        
        logger.info(f"Generated {len(self.feature_names)} features")
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based technical indicators."""
        
        # EMAs
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # Price ratios
        df['price_ema9_ratio'] = df['close'] / df['ema_9']
        df['price_ema21_ratio'] = df['close'] / df['ema_21']
        df['price_ema50_ratio'] = df['close'] / df['ema_50']
        
        # EMA crossovers
        df['ema9_ema21_cross'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)
        df['ema21_ema50_cross'] = np.where(df['ema_21'] > df['ema_50'], 1, -1)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # MACD
        macd, macd_signal, macd_hist = self._calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        
        # ROC (Rate of Change)
        df['roc_5'] = df['close'].pct_change(5)
        df['roc_15'] = df['close'].pct_change(15)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = self._calculate_stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'])
        
        # Rolling volatility
        df['volatility_60'] = df['close'].pct_change().rolling(60).std()
        
        # Price volatility ratios
        df['atr_price_ratio'] = df['atr'] / df['close']
        df['volatility_price_ratio'] = df['volatility_60'] / df['close']
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        
        # Volume moving averages
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_sma_60'] = df['volume'].rolling(60).mean()
        
        # Volume ratios
        df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']
        df['volume_ratio_60'] = df['volume'] / df['volume_sma_60']
        
        # Volume delta
        df['volume_delta_20'] = df['volume'] - df['volume'].rolling(20).mean()
        df['volume_delta_60'] = df['volume'] - df['volume'].rolling(60).mean()
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
        
        # On-Balance Volume
        df['obv'] = self._calculate_obv(df['close'], df['volume'])
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend strength indicators."""
        
        # ADX
        df['adx'] = self._calculate_adx(df['high'], df['low'], df['close'])
        
        # Trend direction
        df['trend_direction'] = np.where(df['ema_21'] > df['ema_50'], 1, -1)
        
        # Trend strength
        df['trend_strength'] = abs(df['ema_21'] - df['ema_50']) / df['ema_50']
        
        return df
    
    def _add_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features."""
        
        # Body and wick ratios
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Ratios
        df['body_ratio'] = df['body_size'] / df['total_range']
        df['upper_wick_ratio'] = df['upper_wick'] / df['total_range']
        df['lower_wick_ratio'] = df['lower_wick'] / df['total_range']
        
        # Candle type
        df['candle_type'] = np.where(df['close'] > df['open'], 1, -1)  # 1 for bullish, -1 for bearish
        
        # Doji detection
        df['is_doji'] = (df['body_ratio'] < 0.1).astype(int)
        
        return df
    
    def _add_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add transaction-based features."""
        
        if 'transactions' not in df.columns:
            # Create dummy transaction features if not available
            df['transactions'] = df['volume'] / 1000  # Rough estimate
        
        # Transaction moving averages
        df['transactions_sma_20'] = df['transactions'].rolling(20).mean()
        df['transactions_sma_60'] = df['transactions'].rolling(60).mean()
        
        # Transaction ratios
        df['transactions_ratio_20'] = df['transactions'] / df['transactions_sma_20']
        df['transactions_ratio_60'] = df['transactions'] / df['transactions_sma_60']
        
        # Transaction trend
        df['transactions_trend'] = df['transactions'].rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan
        )
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                             k_period: int = 14, d_period: int = 3) -> tuple:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = high - low
        high_close_prev = (high - close.shift()).abs()
        low_close_prev = (low - close.shift()).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = np.where(close > close.shift(), volume, 
                      np.where(close < close.shift(), -volume, 0))
        return pd.Series(obv, index=close.index).cumsum()
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        tr = self._calculate_atr(high, low, close, period)
        
        plus_dm = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
        minus_dm = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
        
        plus_di = 100 * (pd.Series(plus_dm, index=high.index).rolling(period).mean() / tr)
        minus_di = 100 * (pd.Series(minus_dm, index=high.index).rolling(period).mean() / tr)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        return adx
    
    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self.feature_names.copy()
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate feature quality.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with validation results
        """
        stats = {
            'total_features': len(self.feature_names),
            'missing_values': df[self.feature_names].isnull().sum().to_dict(),
            'infinite_values': np.isinf(df[self.feature_names]).sum().to_dict(),
            'constant_features': [],
            'correlated_features': []
        }
        
        # Check for constant features
        for feature in self.feature_names:
            if df[feature].nunique() <= 1:
                stats['constant_features'].append(feature)
        
        # Check for highly correlated features
        corr_matrix = df[self.feature_names].corr()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.95:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        stats['correlated_features'] = high_corr_pairs
        
        return stats
