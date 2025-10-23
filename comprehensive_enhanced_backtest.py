"""
Comprehensive Enhanced Strategies Backtest
==========================================

Backtest all enhanced strategies using ALL available data files
for a complete performance analysis.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StrategyConfig:
    """Configuration for trading strategies."""
    name: str
    parameters: Dict[str, Any]
    risk_params: Dict[str, float]
    enabled: bool = True


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
        self.parameters = config.parameters
        self.risk_params = config.risk_params
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        pass


class TechnicalIndicators:
    """Technical indicators implementation using pandas/numpy."""
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        df = data.copy()
        
        # Price-based indicators
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()
        df['EMA_12'] = df['close'].ewm(span=12).mean()
        df['EMA_26'] = df['close'].ewm(span=26).mean()
        
        # Momentum indicators
        df['RSI'] = TechnicalIndicators._calculate_rsi(df['close'], 14)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = TechnicalIndicators._calculate_macd(df['close'])
        df['STOCH_K'], df['STOCH_D'] = TechnicalIndicators._calculate_stochastic(df['high'], df['low'], df['close'])
        
        # Volatility indicators
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = TechnicalIndicators._calculate_bollinger_bands(df['close'])
        df['ATR'] = TechnicalIndicators._calculate_atr(df['high'], df['low'], df['close'])
        
        # Volume indicators
        df['OBV'] = TechnicalIndicators._calculate_obv(df['close'], df['volume'])
        
        # Trend indicators
        df['ADX'] = TechnicalIndicators._calculate_adx(df['high'], df['low'], df['close'])
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    @staticmethod
    def _calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def _calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    @staticmethod
    def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = high - low
        high_close_prev = (high - close.shift()).abs()
        low_close_prev = (low - close.shift()).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def _calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = np.where(close > close.shift(), volume, 
                      np.where(close < close.shift(), -volume, 0))
        return pd.Series(obv, index=close.index).cumsum()
    
    @staticmethod
    def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (simplified)."""
        tr = TechnicalIndicators._calculate_atr(high, low, close, period)
        plus_dm = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
        minus_dm = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
        
        plus_di = 100 * (pd.Series(plus_dm, index=high.index).rolling(period).mean() / tr)
        minus_di = 100 * (pd.Series(minus_dm, index=high.index).rolling(period).mean() / tr)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        return adx


class EnhancedTrendFollowingStrategy(BaseStrategy):
    """Enhanced trend following strategy with multiple indicators."""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced trend following signals."""
        df = data.copy()
        
        # Calculate indicators
        df = TechnicalIndicators.calculate_all_indicators(df)
        
        # Trend conditions
        trend_up = df['SMA_50'] > df['SMA_200']
        price_above_sma = df['close'] > df['SMA_50']
        
        # Momentum conditions
        rsi_ok = (df['RSI'] > 30) & (df['RSI'] < 70)
        macd_bullish = df['MACD'] > df['MACD_signal']
        
        # Volatility conditions
        atr_normal = df['ATR'] < df['ATR'].rolling(20).mean() * 2
        
        # Trend strength
        adx_strong = df['ADX'] > 25
        
        # Entry conditions
        df['entry_signal'] = (
            trend_up & 
            price_above_sma & 
            rsi_ok & 
            macd_bullish & 
            atr_normal & 
            adx_strong
        ).astype(int)
        
        # Exit conditions
        df['exit_signal'] = (
            (df['RSI'] > 80) |
            (df['MACD'] < df['MACD_signal']) |
            (df['close'] < df['SMA_50'] * 0.98)
        ).astype(int)
        
        return df


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands and RSI."""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals."""
        df = data.copy()
        
        # Calculate indicators
        df = TechnicalIndicators.calculate_all_indicators(df)
        
        # Oversold conditions
        oversold = df['close'] < df['BB_lower']
        rsi_oversold = df['RSI'] < 30
        
        # Overbought conditions
        overbought = df['close'] > df['BB_upper']
        rsi_overbought = df['RSI'] > 70
        
        # Entry conditions (buy oversold)
        df['entry_signal'] = (oversold & rsi_oversold).astype(int)
        
        # Exit conditions (sell overbought or return to mean)
        df['exit_signal'] = (
            overbought | 
            rsi_overbought | 
            (df['close'] > df['BB_middle'])
        ).astype(int)
        
        return df


class MomentumStrategy(BaseStrategy):
    """Momentum strategy using MACD and Stochastic."""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum signals."""
        df = data.copy()
        
        # Calculate indicators
        df = TechnicalIndicators.calculate_all_indicators(df)
        
        # MACD momentum
        macd_bullish = df['MACD'] > df['MACD_signal']
        macd_cross_up = (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))
        
        # Stochastic momentum
        stoch_bullish = df['STOCH_K'] > df['STOCH_D']
        stoch_oversold = (df['STOCH_K'] < 20) & (df['STOCH_D'] < 20)
        
        # RSI confirmation
        rsi_ok = (df['RSI'] > 40) & (df['RSI'] < 80)
        
        # Entry conditions
        df['entry_signal'] = (
            macd_cross_up & 
            stoch_bullish & 
            stoch_oversold & 
            rsi_ok
        ).astype(int)
        
        # Exit conditions
        df['exit_signal'] = (
            (df['MACD'] < df['MACD_signal']) |
            (df['STOCH_K'] > 80) |
            (df['RSI'] > 80)
        ).astype(int)
        
        return df


class VolatilityBreakoutStrategy(BaseStrategy):
    """Volatility breakout strategy using ATR and Bollinger Bands."""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility breakout signals."""
        df = data.copy()
        
        # Calculate indicators
        df = TechnicalIndicators.calculate_all_indicators(df)
        
        # Volatility expansion
        atr_expansion = df['ATR'] > df['ATR'].rolling(20).mean() * 1.5
        bb_width = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        bb_expansion = bb_width > bb_width.rolling(20).mean() * 1.2
        
        # Breakout conditions
        breakout_up = df['close'] > df['BB_upper']
        breakout_down = df['close'] < df['BB_lower']
        
        # RSI filter
        rsi_not_extreme = (df['RSI'] > 20) & (df['RSI'] < 80)
        
        # Entry conditions (breakout up)
        df['entry_signal'] = (
            atr_expansion & 
            bb_expansion & 
            breakout_up & 
            rsi_not_extreme
        ).astype(int)
        
        # Exit conditions
        df['exit_signal'] = (
            (df['close'] < df['BB_middle']) |
            (df['RSI'] > 80) |
            (df['ATR'] < df['ATR'].rolling(20).mean())
        ).astype(int)
        
        return df


class SimpleBacktester:
    """Simple backtesting engine."""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 1000.0):
        self.data = data
        self.initial_balance = initial_balance
    
    def backtest_strategy(self, strategy: BaseStrategy) -> Dict[str, Any]:
        """Backtest a strategy."""
        signals = strategy.generate_signals(self.data)
        
        # Initialize variables
        cash = self.initial_balance
        quantity = 0.0
        in_position = False
        entry_price = 0.0
        trades = []
        balance_series = []
        
        # Main trading loop
        for i in range(1, len(signals) - 1):
            close_price = signals['close'].iloc[i]
            
            # Update portfolio value
            if in_position:
                balance_series.append(quantity * close_price)
            else:
                balance_series.append(cash)
            
            # Entry logic
            if not in_position and signals['entry_signal'].iloc[i]:
                next_open = signals['open'].iloc[i + 1]
                buy_price = next_open * 1.001  # Include fees
                quantity = cash / buy_price
                cash = 0.0
                in_position = True
                entry_price = buy_price
            
            # Exit logic
            elif in_position and signals['exit_signal'].iloc[i]:
                next_open = signals['open'].iloc[i + 1]
                sell_price = next_open * 0.999  # Include fees
                trade_return = (sell_price - entry_price) / entry_price
                cash = quantity * sell_price
                quantity = 0.0
                in_position = False
                trades.append(trade_return)
        
        # Final mark-to-market
        last_close = signals['close'].iloc[-1]
        if in_position:
            balance_series.append(quantity * last_close)
            final_price = last_close * 0.999
            trade_return = (final_price - entry_price) / entry_price
            trades.append(trade_return)
            cash = quantity * final_price
        else:
            balance_series.append(cash)
        
        return {
            'trades': trades,
            'equity_curve': pd.Series(balance_series, index=signals.index[1:len(balance_series)+1]),
            'initial_balance': self.initial_balance,
            'final_balance': cash
        }
    
    def calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics."""
        trades = results['trades']
        equity_curve = results['equity_curve']
        initial_balance = results['initial_balance']
        final_balance = results['final_balance']
        
        if not trades:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'num_trades': 0,
                'profit_factor': 0.0
            }
        
        # Basic metrics
        total_return = (final_balance / initial_balance) - 1
        num_trades = len(trades)
        win_rate = (len([t for t in trades if t > 0]) / num_trades) if num_trades > 0 else 0.0
        
        # Profit factor
        sum_profits = sum(t for t in trades if t > 0)
        sum_losses = -sum(t for t in trades if t <= 0)
        profit_factor = sum_profits / sum_losses if sum_losses > 0 else np.inf
        
        # Risk metrics
        hourly_returns = equity_curve.pct_change().fillna(0)
        mean_ret = hourly_returns.mean()
        std_ret = hourly_returns.std()
        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(365 * 24) if std_ret != 0 else 0.0
        
        # Drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max) - 1
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'profit_factor': profit_factor
        }


def load_crypto_data(file_path: str) -> pd.DataFrame:
    """Load cryptocurrency data from CSV file."""
    column_names = [
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ]
    
    data = pd.read_csv(file_path, names=column_names)
    
    # Fix timestamps
    data['open_time'] = data['open_time'].apply(lambda x: x / 1000 if x > 10**15 else x)
    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    data.set_index('open_time', inplace=True)
    
    # Convert to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = pd.to_numeric(data[col])
    
    return data


def load_all_data():
    """Load and combine all available BTC/USDT data."""
    data_files = [
        'BTCUSDT_1h.csv',           # 2023-01 data
        'BTCUSDT-1h-2024-02.csv',   # 2024-02 data
        'BTCUSDT-1h-2024-03.csv',   # 2024-03 data
        'BTCUSDT-1h-2025-06.csv',   # 2025-06 data
        'BTCUSDT-1h-2025-08.csv',   # 2025-08 data
        'BTCUSDT-1h-2025-09.csv'    # 2025-09 data
    ]
    
    all_data = []
    
    for file in data_files:
        if os.path.exists(file):
            print(f"Loading {file}...")
            try:
                data = load_crypto_data(file)
                all_data.append(data)
                print(f"  Loaded {len(data)} records from {data.index[0].date()} to {data.index[-1].date()}")
            except Exception as e:
                print(f"  Error loading {file}: {e}")
    
    if not all_data:
        print("No data files found!")
        return None
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=False)
    combined_data = combined_data.sort_index()
    combined_data = combined_data.drop_duplicates()
    
    print(f"\nCombined dataset:")
    print(f"  Total records: {len(combined_data)}")
    print(f"  Date range: {combined_data.index[0].date()} to {combined_data.index[-1].date()}")
    print(f"  Price range: ${combined_data['close'].min():.2f} to ${combined_data['close'].max():.2f}")
    
    return combined_data


def run_comprehensive_backtest():
    """Run comprehensive backtest with all available data."""
    print("🚀 Comprehensive Enhanced Strategies Backtest")
    print("=" * 70)
    
    # Load all available data
    data = load_all_data()
    if data is None:
        return
    
    # Define strategies
    strategies = [
        EnhancedTrendFollowingStrategy(StrategyConfig(
            name="Enhanced Trend Following",
            parameters={},
            risk_params={'max_risk': 0.02}
        )),
        MeanReversionStrategy(StrategyConfig(
            name="Mean Reversion",
            parameters={},
            risk_params={'max_risk': 0.02}
        )),
        MomentumStrategy(StrategyConfig(
            name="Momentum",
            parameters={},
            risk_params={'max_risk': 0.02}
        )),
        VolatilityBreakoutStrategy(StrategyConfig(
            name="Volatility Breakout",
            parameters={},
            risk_params={'max_risk': 0.02}
        ))
    ]
    
    # Backtest strategies
    backtester = SimpleBacktester(data)
    
    print("\n📊 Comprehensive Strategy Performance Analysis:")
    print("-" * 70)
    
    strategy_results = {}
    for strategy in strategies:
        try:
            print(f"\n🔄 Testing {strategy.name}...")
            results = backtester.backtest_strategy(strategy)
            metrics = backtester.calculate_metrics(results)
            strategy_results[strategy.name] = metrics
            
            print(f"✅ {strategy.name} Results:")
            print(f"   Total Return: {metrics['total_return']*100:.2f}%")
            print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"   Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            print(f"   Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"   Number of Trades: {metrics['num_trades']}")
            print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
            
        except Exception as e:
            print(f"❌ Error testing {strategy.name}: {e}")
    
    # Create performance comparison table
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON TABLE")
    print("=" * 70)
    
    comparison_data = []
    for strategy_name, metrics in strategy_results.items():
        comparison_data.append({
            'Strategy': strategy_name,
            'Total Return (%)': f"{metrics['total_return']*100:.2f}",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
            'Max Drawdown (%)': f"{metrics['max_drawdown']*100:.2f}",
            'Win Rate (%)': f"{metrics['win_rate']*100:.1f}",
            'Num Trades': metrics['num_trades'],
            'Profit Factor': f"{metrics['profit_factor']:.2f}"
        })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # Find best strategies
    if strategy_results:
        print("\n" + "=" * 50)
        print("BEST PERFORMING STRATEGIES")
        print("=" * 50)
        
        # Best by different metrics
        best_sharpe = max(strategy_results.items(), key=lambda x: x[1]['sharpe_ratio'])
        best_return = max(strategy_results.items(), key=lambda x: x[1]['total_return'])
        lowest_drawdown = min(strategy_results.items(), key=lambda x: x[1]['max_drawdown'])
        most_trades = max(strategy_results.items(), key=lambda x: x[1]['num_trades'])
        
        print(f"🏆 Best Sharpe Ratio: {best_sharpe[0]}")
        print(f"   Sharpe: {best_sharpe[1]['sharpe_ratio']:.3f}, Return: {best_sharpe[1]['total_return']*100:.2f}%, Trades: {best_sharpe[1]['num_trades']}")
        
        print(f"\n💰 Best Total Return: {best_return[0]}")
        print(f"   Return: {best_return[1]['total_return']*100:.2f}%, Sharpe: {best_return[1]['sharpe_ratio']:.3f}, Trades: {best_return[1]['num_trades']}")
        
        print(f"\n🛡️ Lowest Drawdown: {lowest_drawdown[0]}")
        print(f"   Drawdown: {lowest_drawdown[1]['max_drawdown']*100:.2f}%, Return: {lowest_drawdown[1]['total_return']*100:.2f}%, Trades: {lowest_drawdown[1]['num_trades']}")
        
        print(f"\n📈 Most Active: {most_trades[0]}")
        print(f"   Trades: {most_trades[1]['num_trades']}, Return: {most_trades[1]['total_return']*100:.2f}%, Sharpe: {most_trades[1]['sharpe_ratio']:.3f}")
        
        # Overall recommendation
        print("\n" + "=" * 50)
        print("FINAL RECOMMENDATION")
        print("=" * 50)
        
        # Score strategies based on multiple criteria
        scored_strategies = []
        for name, metrics in strategy_results.items():
            if metrics['num_trades'] > 0:  # Only consider strategies with trades
                score = (
                    metrics['sharpe_ratio'] * 0.4 +  # Risk-adjusted returns
                    metrics['total_return'] * 10 * 0.3 +  # Absolute returns
                    (1 + metrics['max_drawdown']) * 0.2 +  # Drawdown control
                    metrics['win_rate'] * 0.1  # Win rate
                )
                scored_strategies.append((name, score, metrics))
        
        if scored_strategies:
            best_strategy = max(scored_strategies, key=lambda x: x[1])
            
            print(f"🥇 BEST OVERALL STRATEGY: {best_strategy[0]}")
            print(f"   Overall Score: {best_strategy[1]:.3f}")
            
            metrics = best_strategy[2]
            print(f"\n   Performance Summary:")
            print(f"   • Total Return: {metrics['total_return']*100:.2f}%")
            print(f"   • Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"   • Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            print(f"   • Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"   • Number of Trades: {metrics['num_trades']}")
            print(f"   • Profit Factor: {metrics['profit_factor']:.2f}")
            
            # Trading recommendations
            print(f"\n   Trading Recommendations:")
            if metrics['sharpe_ratio'] > 2.0:
                print("   ✅ Excellent for live trading")
            elif metrics['sharpe_ratio'] > 1.0:
                print("   ✅ Good for live trading")
            else:
                print("   ⚠️ Consider additional optimization")
            
            if metrics['max_drawdown'] > -0.1:
                print("   ✅ Low drawdown risk")
            elif metrics['max_drawdown'] > -0.2:
                print("   ✅ Moderate drawdown risk")
            else:
                print("   ⚠️ High drawdown risk - use strict risk management")
            
            if metrics['num_trades'] > 50:
                print("   ✅ Good trading frequency")
            elif metrics['num_trades'] > 20:
                print("   ✅ Adequate trading frequency")
            else:
                print("   ⚠️ Low trading frequency")
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE BACKTEST COMPLETE")
    print("=" * 70)
    print("Ready for live trading implementation with the best performing strategy!")


if __name__ == "__main__":
    run_comprehensive_backtest()

