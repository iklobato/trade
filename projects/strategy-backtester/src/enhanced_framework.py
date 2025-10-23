"""
Enhanced Cryptocurrency Trading Framework
========================================

A comprehensive trading framework using advanced libraries for better organization,
more strategies, and improved profitability.

Libraries used:
- TA-Lib: Technical analysis indicators
- Optuna: Hyperparameter optimization
- Vectorbt: Fast backtesting and analysis
- CCXT: Exchange integration
- Freqtrade: Strategy framework
"""

import pandas as pd
import numpy as np
import talib
import optuna
import vectorbt as vbt
import ccxt
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Configure vectorbt for better performance
vbt.settings.set_theme("dark")
vbt.settings['plotting']['layout']['width'] = 1000
vbt.settings['plotting']['layout']['height'] = 600


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
    
    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """Get list of required technical indicators."""
        pass


class TechnicalIndicators:
    """Comprehensive technical indicators using TA-Lib."""
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        df = data.copy()
        
        # Price-based indicators
        df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
        df['SMA_200'] = talib.SMA(df['close'], timeperiod=200)
        df['EMA_12'] = talib.EMA(df['close'], timeperiod=12)
        df['EMA_26'] = talib.EMA(df['close'], timeperiod=26)
        
        # Momentum indicators
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'])
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['high'], df['low'], df['close'])
        df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'])
        
        # Volatility indicators
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'])
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volume indicators
        df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        df['OBV'] = talib.OBV(df['close'], df['volume'])
        df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
        
        # Trend indicators
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['AROON_UP'], df['AROON_DOWN'] = talib.AROON(df['high'], df['low'], timeperiod=14)
        df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Pattern recognition
        df['DOJI'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['HAMMER'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['ENGULFING'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        
        return df


class EnhancedTrendFollowingStrategy(BaseStrategy):
    """Enhanced trend following strategy with multiple indicators."""
    
    def get_required_indicators(self) -> List[str]:
        return ['SMA_50', 'SMA_200', 'RSI', 'MACD', 'ATR', 'ADX']
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced trend following signals."""
        df = data.copy()
        
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
    
    def get_required_indicators(self) -> List[str]:
        return ['BB_upper', 'BB_lower', 'BB_middle', 'RSI', 'ATR']
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals."""
        df = data.copy()
        
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
    
    def get_required_indicators(self) -> List[str]:
        return ['MACD', 'MACD_signal', 'STOCH_K', 'STOCH_D', 'RSI', 'ATR']
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum signals."""
        df = data.copy()
        
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
    
    def get_required_indicators(self) -> List[str]:
        return ['ATR', 'BB_upper', 'BB_lower', 'BB_middle', 'RSI']
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility breakout signals."""
        df = data.copy()
        
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


class StrategyOptimizer:
    """Optimize strategies using Optuna."""
    
    def __init__(self, strategy_class, data: pd.DataFrame):
        self.strategy_class = strategy_class
        self.data = data
        self.best_params = None
        self.best_score = -np.inf
    
    def objective(self, trial):
        """Objective function for optimization."""
        # Define parameter ranges
        params = {}
        
        if hasattr(self.strategy_class, 'get_parameter_ranges'):
            param_ranges = self.strategy_class.get_parameter_ranges()
            for param_name, param_range in param_ranges.items():
                if param_range['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_range['low'], 
                        param_range['high']
                    )
                elif param_range['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, 
                        param_range['low'], 
                        param_range['high']
                    )
                elif param_range['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, 
                        param_range['choices']
                    )
        
        # Create strategy with trial parameters
        config = StrategyConfig(
            name=f"{self.strategy_class.__name__}_optimized",
            parameters=params,
            risk_params={'max_risk': 0.02}
        )
        
        strategy = self.strategy_class(config)
        
        # Generate signals
        signals = strategy.generate_signals(self.data)
        
        # Calculate performance
        score = self._calculate_score(signals)
        
        return score
    
    def _calculate_score(self, signals: pd.DataFrame) -> float:
        """Calculate optimization score."""
        # Simple scoring based on signal quality
        entry_signals = signals['entry_signal'].sum()
        exit_signals = signals['exit_signal'].sum()
        
        if entry_signals == 0:
            return -1000
        
        # Score based on signal frequency and quality
        signal_ratio = exit_signals / entry_signals if entry_signals > 0 else 0
        score = entry_signals * signal_ratio
        
        return score
    
    def optimize(self, n_trials: int = 100) -> Dict[str, Any]:
        """Run optimization."""
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': n_trials
        }


class VectorbtBacktester:
    """Fast backtesting using Vectorbt."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.prices = vbt.YFData.download('BTC-USD').get('Close')
    
    def backtest_strategy(self, strategy: BaseStrategy) -> Dict[str, Any]:
        """Backtest a strategy using Vectorbt."""
        # Generate signals
        signals = strategy.generate_signals(self.data)
        
        # Convert to vectorbt format
        entries = signals['entry_signal'].fillna(0)
        exits = signals['exit_signal'].fillna(0)
        
        # Run backtest
        pf = vbt.Portfolio.from_signals(
            self.data['close'],
            entries=entries,
            exits=exits,
            fees=0.001,
            freq='1H'
        )
        
        # Calculate metrics
        metrics = {
            'total_return': pf.total_return(),
            'sharpe_ratio': pf.sharpe_ratio(),
            'max_drawdown': pf.max_drawdown(),
            'win_rate': pf.trades.win_rate(),
            'profit_factor': pf.trades.profit_factor(),
            'num_trades': pf.trades.count(),
            'avg_trade_duration': pf.trades.duration.mean(),
            'total_fees': pf.trades.fees.sum()
        }
        
        return metrics
    
    def plot_results(self, strategy: BaseStrategy, **kwargs):
        """Plot backtest results."""
        signals = strategy.generate_signals(self.data)
        entries = signals['entry_signal'].fillna(0)
        exits = signals['exit_signal'].fillna(0)
        
        pf = vbt.Portfolio.from_signals(
            self.data['close'],
            entries=entries,
            exits=exits,
            fees=0.001,
            freq='1H'
        )
        
        return pf.plot(**kwargs)


class MultiStrategyPortfolio:
    """Portfolio of multiple strategies."""
    
    def __init__(self, strategies: List[BaseStrategy], weights: Optional[List[float]] = None):
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
        
        if len(self.weights) != len(strategies):
            raise ValueError("Number of weights must match number of strategies")
    
    def generate_combined_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate combined signals from all strategies."""
        combined_signals = pd.DataFrame(index=data.index)
        combined_signals['entry_signal'] = 0
        combined_signals['exit_signal'] = 0
        
        for strategy, weight in zip(self.strategies, self.weights):
            signals = strategy.generate_signals(data)
            combined_signals['entry_signal'] += signals['entry_signal'] * weight
            combined_signals['exit_signal'] += signals['exit_signal'] * weight
        
        # Normalize signals
        combined_signals['entry_signal'] = (combined_signals['entry_signal'] > 0.5).astype(int)
        combined_signals['exit_signal'] = (combined_signals['exit_signal'] > 0.5).astype(int)
        
        return combined_signals
    
    def backtest_portfolio(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Backtest the entire portfolio."""
        signals = self.generate_combined_signals(data)
        
        # Use vectorbt for fast backtesting
        entries = signals['entry_signal'].fillna(0)
        exits = signals['exit_signal'].fillna(0)
        
        pf = vbt.Portfolio.from_signals(
            data['close'],
            entries=entries,
            exits=exits,
            fees=0.001,
            freq='1H'
        )
        
        metrics = {
            'total_return': pf.total_return(),
            'sharpe_ratio': pf.sharpe_ratio(),
            'max_drawdown': pf.max_drawdown(),
            'win_rate': pf.trades.win_rate(),
            'profit_factor': pf.trades.profit_factor(),
            'num_trades': pf.trades.count(),
            'total_fees': pf.trades.fees.sum()
        }
        
        return metrics


class ExchangeConnector:
    """Connect to cryptocurrency exchanges using CCXT."""
    
    def __init__(self, exchange_name: str = 'binance'):
        self.exchange_name = exchange_name
        self.exchange = getattr(ccxt, exchange_name)()
    
    def get_historical_data(self, symbol: str, timeframe: str = '1h', limit: int = 1000) -> pd.DataFrame:
        """Get historical data from exchange."""
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['last']


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Load and prepare data with all technical indicators."""
    # Load data
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
    
    # Calculate all technical indicators
    data = TechnicalIndicators.calculate_all_indicators(data)
    
    return data


def run_enhanced_analysis():
    """Run comprehensive analysis with multiple strategies."""
    print("🚀 Enhanced Cryptocurrency Trading Framework")
    print("=" * 60)
    
    # Load data
    try:
        data = load_and_prepare_data('BTCUSDT_1h.csv')
        print(f"✅ Loaded {len(data)} records with technical indicators")
    except FileNotFoundError:
        print("❌ Data file not found. Please ensure BTCUSDT_1h.csv exists.")
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
    
    # Backtest individual strategies
    backtester = VectorbtBacktester(data)
    
    print("\n📊 Individual Strategy Performance:")
    print("-" * 60)
    
    strategy_results = {}
    for strategy in strategies:
        try:
            metrics = backtester.backtest_strategy(strategy)
            strategy_results[strategy.name] = metrics
            
            print(f"\n{strategy.name}:")
            print(f"  Total Return: {metrics['total_return']*100:.2f}%")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"  Number of Trades: {metrics['num_trades']}")
            
        except Exception as e:
            print(f"❌ Error testing {strategy.name}: {e}")
    
    # Test portfolio
    print("\n🎯 Multi-Strategy Portfolio:")
    print("-" * 60)
    
    portfolio = MultiStrategyPortfolio(strategies)
    portfolio_metrics = portfolio.backtest_portfolio(data)
    
    print(f"Portfolio Performance:")
    print(f"  Total Return: {portfolio_metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {portfolio_metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {portfolio_metrics['win_rate']*100:.1f}%")
    print(f"  Number of Trades: {portfolio_metrics['num_trades']}")
    
    # Find best strategy
    if strategy_results:
        best_strategy = max(strategy_results.items(), key=lambda x: x[1]['sharpe_ratio'])
        print(f"\n🏆 Best Strategy: {best_strategy[0]}")
        print(f"   Sharpe Ratio: {best_strategy[1]['sharpe_ratio']:.3f}")
        print(f"   Total Return: {best_strategy[1]['total_return']*100:.2f}%")
    
    print("\n✅ Enhanced analysis complete!")
    print("Ready for live trading implementation with multiple strategies.")


if __name__ == "__main__":
    run_enhanced_analysis()

